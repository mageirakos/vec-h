# Engine Limitations: LIST Column Types

This document details the current limitations of the Maximus execution engines (cuDF and Acero) regarding the support for `LIST` data types (e.g., `list<item: float>`, embeddings).

## Verified by Tests

The following test suite systematically verified LIST type support:
- **Test file**: [`tests/list_type_support.cpp`](file:///local/home/vmageirakos/projects/Maximus/tests/list_type_support.cpp)
- **Run**: `make run CMD="./build/Debug/tests/test.list_type_support"`

| Test | Status | Notes |
|------|--------|-------|
| LoadListArrayFromVSDS_CPU | ✅ PASS | `list<item: float>` loads correctly |
| LoadLargeListArrayFromVSDS_CPU | ✅ PASS | `large_list<item: float>` loads correctly |
| CPUFilterWithListColumn | ✅ PASS | Filter passes LIST through unchanged |
| CPUProjectPassthroughWithListColumn | ✅ PASS | Simple projection works |
| CPUHashJoinWorkaround_StripListColumn | ✅ PASS | Workaround: strip LIST before join |
| ExhaustiveVectorJoin_CPU | ✅ PASS | ENN works with embeddings |
| IndexedVectorJoin_CPU | ✅ PASS | ANN works with embeddings |
| DISABLED_CPUHashJoinWithListPayload | ⛔ DISABLED | Expected FAIL: LIST as join payload |
| ArrowToCuDFListConversion | ⛔ FAIL | Schema assertion failure during conversion |
| GPUFilterWithListColumn | ⛔ FAIL | Blocked by Arrow↔cuDF interop |
| CuDFToArrowListConversion | ⛔ FAIL | Critical: `id_to_arrow_type` missing LIST |

---

## 1. cuDF (GPU Engine)

### a. Interop (CRITICAL) - No Results back to CPU
**Status**: ⛔ UNSUPPORTED  
**Impact**: Results of type `LIST` cannot be retrieved from the GPU back to the CPU. This affects **any** operation that produces List output on GPU.

**Question**: But I think it should be fine if we drop the embeddings column before we bring it to the CPU. We have a flag to keep_vector_column=false, that should be on by default on all GPU operations, and then when we do table_sink() to the final result back to the CPU we should just not bring the vectors. Just drop the embeddings after you compute vector search operation, since we do not use them for anything else afterwards. 

**Limitation** Indeed this is a limitation if for some reason to move embeddings/vectors from GPU to CPU.

**Root Cause**: `id_to_arrow_type` missing case for `cudf::type_id::LIST`.

**Reference Files**:
- [`cudf/cpp/src/interop/arrow_utilities.cpp`](file:///local/home/vmageirakos/projects/_deps_code/Maximus/cudf/cpp/src/interop/arrow_utilities.cpp): `id_to_arrow_type()`
- [`src/maximus/gpu/cudf/cudf_types.cpp`](file:///local/home/vmageirakos/projects/Maximus/src/maximus/gpu/cudf/cudf_types.cpp): `to_arrow_type()`, `to_maximus_type()`

**Possible Fix**:
```cpp
// In cudf/cpp/src/interop/arrow_utilities.cpp: id_to_arrow_type()
case cudf::type_id::LIST: return NANOARROW_TYPE_LIST;

// In src/maximus/gpu/cudf/cudf_types.cpp: to_arrow_type()
case cudf::type_id::LIST: {
    // Use cudf::lists_column_view to get child type, then recursively convert
    return arrow::list(to_arrow_type(child_type));
}
```

### b. Project (AST Expressions)
**Status**: ⛔ UNSUPPORTED  
**Impact**: Cannot use LIST columns in computed expressions.

**Reference**: [`src/maximus/gpu/cudf/cudf_expr.cpp`](file:///local/home/vmageirakos/projects/Maximus/src/maximus/gpu/cudf/cudf_expr.cpp)

**Possible Fix**: Add LIST case in `get_expr_type()`. Note: This may require significant work since cuDF AST has limited LIST support.

### c. GroupBy (Aggregations producing LIST)
**Status**: ⛔ UNSUPPORTED  
**Impact**: Aggregations like `collect_list` blocked by Interop limitation.

**Possible Fix**: Fix Interop (1.a) first, then implement list aggregation wrappers.

---

## 2. Acero (CPU Engine)

### a. Hash Join - Non-Key Payloads
**Status**: ⛔ UNSUPPORTED  
**Impact**: LIST columns cannot be passed through joins as payload (non-key) columns.

**Observed Error**:
```
Maximus Error: 4; Message: Data type list<item: float> is not supported in join non-key field
```

**Reference**: [`src/maximus/operators/acero/proxy_operator.hpp`](file:///local/home/vmageirakos/projects/Maximus/src/maximus/operators/acero/proxy_operator.hpp) ~L245

**Workaround** (Verified in tests):
```cpp
// 1. Strip LIST columns before join
auto reviews_no_list = table_source(db, "reviews", schema, {"rv_reviewkey", "rv_partkey"}, device);
// 2. Perform join without LIST
auto joined = inner_join(reviews_no_list, other_table, keys, keys, "", "", device);
// 3. Re-join to get LIST column back if needed
```

**Possible Fix**: This is an Arrow Acero limitation. Options:
1. Use the workaround pattern (implemented in `prejoin_reviews`)
2. Implement custom join operator that handles LIST columns natively

---

## 3. Summary: What Works vs What Doesn't

| Operation | CPU (Acero) | GPU (cuDF) |
|-----------|-------------|------------|
| Load LIST/LARGE_LIST | ✅ | ✅ (Arrow→cuDF) |
| Filter with LIST column | ✅ | ⛔ (interop blocks result) |
| Project passthrough | ✅ | ⛔ (AST limitation) |
| Hash Join with LIST key | ✅ | ⛔ |
| Hash Join with LIST payload | ⛔ | ⛔ |
| Vector Join (FAISS) | ✅ | ⛔ (interop blocks result) |
| Return LIST to CPU | N/A | ⛔ CRITICAL |

---

## 4. Priority Fixes

1. **HIGH**: Fix `id_to_arrow_type` for LIST in cudf interop - unblocks all GPU LIST operations
2. **MEDIUM**: Add LIST passthrough support in GPU project operator
3. **LOW**: Acero join workaround already documented and tested

