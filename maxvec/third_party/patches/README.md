# Third-Party Patches

Patches for upstream dependencies (raft/cuVS headers, Faiss source) that affect CAGRA vector search on ATS/unified-memory systems (NVIDIA GH200, DGX-Spark GB10).

## Patch Files

| File | Target | Build-arg | What it does |
|------|--------|-----------|-------------|
| `raft-ats-runtime.patch` | `$CONDA_PREFIX/include/` | `APPLY_RAFT_ATS_PATCH=true` | Runtime data=0/data=1 control via atomic flag |
| `faiss-cagra-graph.patch` | `/home/faiss/` | `APPLY_FAISS_CAGRA_PATCH=true` | Graph cache: `copyFrom_graph()` + uint32 constructor |
| `faiss-ivf-host-view.patch` | `/home/faiss/` | `APPLY_FAISS_IVF_PATCH=true` | IVF host view: `referenceFrom()` + `referenceInvertedListsFrom()` |

Legacy Python scripts (`.py` files) are kept as backup/reference but are NOT used by the Dockerfile or Makefile.

---

## `raft-ats-runtime.patch` — Runtime ATS Data Placement Control

### The Bug

`make_strided_dataset()` in raft/cuVS uses `device_ptr != nullptr` to decide if memory is device-accessible. On ATS systems, `cudaPointerGetAttributes()` returns non-null `devicePointer` for ALL memory, so the check always returns true → creates a non-owning view of host memory. This prevents cuVS from copying host data to GPU when needed.

### The Fix

Adds a `std::atomic<bool> cagra_ats_force_copy_enabled` flag. When set to true (by Maximus for `cagra_data_on_gpu=1`), the check uses `ptr_attrs.type == cudaMemoryTypeDevice` instead, causing host data to be copied to GPU. When false (default), original ATS behavior preserved (non-owning host view for `cagra_data_on_gpu=0`).

### Files patched

| File | Change |
|------|--------|
| `raft/neighbors/dataset.hpp` | Add `#include <atomic>`, atomic flag namespace, conditional check in `make_strided_dataset()` |
| `cuvs/neighbors/common.hpp` | Add include for flag, conditional check in `make_strided_dataset()` |
| `raft/neighbors/detail/cagra/utils.hpp` | Add include for flag, conditional check in `device_matrix_view_from_host()` |

### How to regenerate

If upstream raft/cuVS headers change (e.g., new conda package version):

```bash
# In a container with UNPATCHED headers:
# 1. Save originals
cp -r $CONDA_PREFIX/include/raft/neighbors /tmp/raft_neighbors_orig
cp -r $CONDA_PREFIX/include/cuvs/neighbors /tmp/cuvs_neighbors_orig

# 2. Apply the Python script (kept as reference)
python3 third_party/patches/patch_raft_cuvs_ats.py

# 3. Generate new .patch file
cd $CONDA_PREFIX/include
diff -ruN /tmp/raft_neighbors_orig raft/neighbors > raft-ats-runtime.patch
diff -ruN /tmp/cuvs_neighbors_orig cuvs/neighbors >> raft-ats-runtime.patch

# 4. Fix paths in the patch (replace /tmp/*_orig/ with the correct relative paths)
sed -i 's|/tmp/raft_neighbors_orig/|raft/neighbors/|g' raft-ats-runtime.patch
sed -i 's|/tmp/cuvs_neighbors_orig/|cuvs/neighbors/|g' raft-ats-runtime.patch
```

---

## `faiss-cagra-graph.patch` — CAGRA Graph Cache

### What it adds

- `CuvsCagra` constructor taking `const uint32_t*` graph (skips int64→uint32 conversion)
- `GpuIndexCagra::copyFrom_graph()` method (constructs GPU CAGRA from pre-extracted uint32 graph + dataset pointer)

This enables `cagra_cache_graph=1`: cache the CAGRA graph as uint32 during `to_cpu()`, use `copyFrom_graph()` in `to_gpu()` to skip HNSW graph extraction. Reduces per-rep `index_movement` from ~280ms to ~100ms.

### Files patched

| File | Change |
|------|--------|
| `faiss/gpu/impl/CuvsCagra.cuh` | uint32 constructor declaration |
| `faiss/gpu/impl/CuvsCagra.cu` | uint32 constructor implementation |
| `faiss/gpu/GpuIndexCagra.h` | `copyFrom_graph()` declaration |
| `faiss/gpu/GpuIndexCagra.cu` | `copyFrom_graph()` implementation |

### How to regenerate

If upstream Faiss version changes:

```bash
# In a container with UNPATCHED Faiss source at /home/faiss:
# 1. Save originals
cp faiss/gpu/impl/CuvsCagra.cuh /tmp/CuvsCagra.cuh.orig
cp faiss/gpu/impl/CuvsCagra.cu /tmp/CuvsCagra.cu.orig
cp faiss/gpu/GpuIndexCagra.h /tmp/GpuIndexCagra.h.orig
cp faiss/gpu/GpuIndexCagra.cu /tmp/GpuIndexCagra.cu.orig

# 2. Apply the Python script (kept as reference)
python3 third_party/patches/patch_faiss_cagra_graph.py /home/faiss

# 3. Generate new .patch file
cd /home/faiss
diff -u /tmp/CuvsCagra.cuh.orig faiss/gpu/impl/CuvsCagra.cuh > faiss-cagra-graph.patch
diff -u /tmp/CuvsCagra.cu.orig faiss/gpu/impl/CuvsCagra.cu >> faiss-cagra-graph.patch
diff -u /tmp/GpuIndexCagra.h.orig faiss/gpu/GpuIndexCagra.h >> faiss-cagra-graph.patch
diff -u /tmp/GpuIndexCagra.cu.orig faiss/gpu/GpuIndexCagra.cu >> faiss-cagra-graph.patch

# 4. Fix paths
sed -i 's|/tmp/CuvsCagra.cuh.orig|faiss/gpu/impl/CuvsCagra.cuh|g' faiss-cagra-graph.patch
sed -i 's|/tmp/CuvsCagra.cu.orig|faiss/gpu/impl/CuvsCagra.cu|g' faiss-cagra-graph.patch
sed -i 's|/tmp/GpuIndexCagra.h.orig|faiss/gpu/GpuIndexCagra.h|g' faiss-cagra-graph.patch
sed -i 's|/tmp/GpuIndexCagra.cu.orig|faiss/gpu/GpuIndexCagra.cu|g' faiss-cagra-graph.patch
```

---

## `faiss-ivf-host-view.patch` — IVF Host View (Zero-Copy)

### What it adds

- `IVFBase::referenceInvertedListsFrom()`: sets `deviceListDataPointers_` and `deviceListIndexPointers_` to host addresses instead of allocating GPU memory and copying data. The GPU reads inverted list codes and IDs directly from host memory via ATS page walks.
- `GpuIndexIVFFlat::referenceFrom()`: wrapper that copies the coarse quantizer to GPU, creates an `IVFFlat` with `interleavedLayout=false`, then calls `referenceInvertedListsFrom()`.

This enables `index_data_on_gpu=0` for IVF indexes: the `to_gpu()` path in Maximus uses `referenceFrom()` instead of `copyFrom()`, skipping the full H2D copy of all inverted lists.

### Requirements

- `use_cuvs=0` — cuVS IVF forces `interleavedLayout=true`, incompatible with host view
- `interleavedLayout=false` — forced by `referenceFrom()`, matches CPU data format
- `INDICES_64_BIT` — default, allows direct pointer reference for index IDs
- ATS system (GH200, DGX-Spark) — GPU must be able to dereference host pointers

### Files patched

| File | Change |
|------|--------|
| `faiss/gpu/impl/IVFBase.cuh` | `referenceInvertedListsFrom()` declaration |
| `faiss/gpu/impl/IVFBase.cu` | `referenceInvertedListsFrom()` implementation (~35 lines) |
| `faiss/gpu/GpuIndexIVFFlat.h` | `referenceFrom()` declaration |
| `faiss/gpu/GpuIndexIVFFlat.cu` | `referenceFrom()` implementation (~40 lines) |

### How to regenerate

```bash
# In a container with UNPATCHED Faiss source at /home/faiss:
# 1. Save originals
cp faiss/gpu/impl/IVFBase.cuh /tmp/IVFBase.cuh.orig
cp faiss/gpu/impl/IVFBase.cu /tmp/IVFBase.cu.orig
cp faiss/gpu/GpuIndexIVFFlat.h /tmp/GpuIndexIVFFlat.h.orig
cp faiss/gpu/GpuIndexIVFFlat.cu /tmp/GpuIndexIVFFlat.cu.orig

# 2. Apply changes manually (see patch file for exact additions)

# 3. Generate new .patch file
cd /home/faiss
diff -u /tmp/IVFBase.cuh.orig faiss/gpu/impl/IVFBase.cuh > faiss-ivf-host-view.patch
diff -u /tmp/IVFBase.cu.orig faiss/gpu/impl/IVFBase.cu >> faiss-ivf-host-view.patch
diff -u /tmp/GpuIndexIVFFlat.h.orig faiss/gpu/GpuIndexIVFFlat.h >> faiss-ivf-host-view.patch
diff -u /tmp/GpuIndexIVFFlat.cu.orig faiss/gpu/GpuIndexIVFFlat.cu >> faiss-ivf-host-view.patch

# 4. Fix paths
sed -i 's|/tmp/IVFBase.cuh.orig|faiss/gpu/impl/IVFBase.cuh|g' faiss-ivf-host-view.patch
sed -i 's|/tmp/IVFBase.cu.orig|faiss/gpu/impl/IVFBase.cu|g' faiss-ivf-host-view.patch
sed -i 's|/tmp/GpuIndexIVFFlat.h.orig|faiss/gpu/GpuIndexIVFFlat.h|g' faiss-ivf-host-view.patch
sed -i 's|/tmp/GpuIndexIVFFlat.cu.orig|faiss/gpu/GpuIndexIVFFlat.cu|g' faiss-ivf-host-view.patch
```

---

## Usage

### Build-time (baked into Docker image)

```bash
make build-image APPLY_RAFT_ATS_PATCH=true APPLY_FAISS_CAGRA_PATCH=true APPLY_FAISS_IVF_PATCH=true
```

### Runtime (patch a running container)

```bash
make apply-raft-cuvs-ats-patch    # patches headers + rebuilds Faiss + Maximus
make apply-faiss-cagra-patch      # patches Faiss source + rebuilds Faiss + Maximus
make apply-faiss-ivf-patch        # patches Faiss source + rebuilds Faiss + Maximus
```

### Non-ATS machines

```bash
make build-image APPLY_RAFT_ATS_PATCH=false   # skip raft patch (not needed on x86 dGPU)
```
