#include "faiss_gpu_index.hpp"

#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>

// Include ALL the heavy GPU headers here
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_factory.h>
#include <faiss/gpu/GpuCloner.h>

#include <cuda_runtime.h>

#include <maximus/gpu/gtable/cuda/cuda_context.hpp>
#include <maximus/operators/faiss/interop.hpp>
#include <maximus/utils/utils.hpp>

#include "gpu_resources.hpp"

namespace maximus::faiss {

FaissGPUIndex::FaissGPUIndex(std::shared_ptr<MaximusContext>& ctx,
                             int d,
                             std::unique_ptr<::faiss::gpu::GpuIndex> index)
        : FaissIndex(ctx, d, std::move(index)) {
    device_type = DeviceType::GPU;
};

FaissGPUIndex::FaissGPUIndex(std::shared_ptr<MaximusContext>& ctx,
                             int d,
                             std::unique_ptr<::faiss::gpu::GpuIndex> index,
                             std::shared_ptr<::faiss::gpu::GpuResourcesProvider> provider)
        : FaissIndex(ctx, d, std::move(index)) {
    device_type    = DeviceType::GPU;
    this->provider = std::move(provider);
};

FaissGPUIndex::FaissGPUIndex(std::shared_ptr<MaximusContext>& ctx,
                             int d,
                             const std::string& description,
                             MetricType metric,
                             bool use_cuvs_param)
        : FaissIndex(ctx, d, metric) {
    this->use_cuvs    = use_cuvs_param;
    device_type       = DeviceType::GPU;
    this->description = description;

    auto [provider, res] = make_gpu_provider_and_resources(ctx);
    this->provider       = provider;

    // 1. Handle Cagra (Special non-standard factory type)
    if (starts_with(description, "GPU,Cagra")) {
        auto config = parse_cagra_config(description);
        faiss_index =
            std::make_unique<::faiss::gpu::GpuIndexCagra>(this->provider.get(), d, metric, config);
    }
    // 2. Generic FAISS Factory path (GPU,...)
    // This allows using standard strings like "GPU,IVF1024,Flat" or "GPU,Flat"
    else if (starts_with(description, "GPU,")) {
        std::string cpu_desc = description.substr(4); // Remove "GPU," prefix

        std::unique_ptr<::faiss::Index> cpu_index;
        try {
            cpu_index.reset(::faiss::index_factory(d, cpu_desc.c_str(), metric));
        } catch (const std::exception& e) {
             throw std::invalid_argument("Failed to parse FAISS factory string '" + cpu_desc + "': " + e.what());
        }

        ::faiss::gpu::GpuClonerOptions options;
        options.use_cuvs = this->use_cuvs;
        // IVFPQ GPU lookup tables: M*256*4 bytes must fit in shared memory (49152 bytes).
        // useFloat16 halves the footprint (M*256*2) and is a no-op for non-PQ indexes.
        if (cpu_desc.find("PQ") != std::string::npos) {
            options.useFloat16 = true;
        }

        auto gpu_index_raw = ::faiss::gpu::index_cpu_to_gpu(
            provider.get(),
            0, // Default to device 0
            cpu_index.get(),
            &options
        );

        faiss_index.reset(gpu_index_raw);
    } else {
        throw std::invalid_argument("Unsupported GPU index type: " + description);
    }
    assert(faiss_index->d == this->D);
}

FaissGPUIndex::~FaissGPUIndex() {
    // GpuIndexCagra may hold zero-copy views of host data backed by
    // device_training_data_ or source_cpu_ref_. Destroy the GPU index
    // first to release cuVS internal references before freeing storage.
    if (device_training_data_ || source_cpu_ref_) {
        faiss_index.reset();
    }
    if (device_training_data_) {
        ctx->pool_mr.deallocate(device_training_data_, device_training_data_size_,
                                rmm::cuda_stream_default);
        device_training_data_ = nullptr;
        device_training_data_size_ = 0;
    }
    // source_cpu_ref_ destroyed automatically by member dtor after
    // faiss_index is already null — no dangling view.
}

// ---------------------------------------------------------------------------
// CAGRA Dataset GPU Copy
//
// On ATS systems (GH200, DGX-Spark), cuVS's make_strided_dataset() may create
// a non-owning view of host memory. If the source buffer is freed before search
// (e.g., Arrow's get_table() clone going out of scope), the view dangles and
// CAGRA search crashes with cudaErrorIllegalAddress.
//
// Copying to GPU (via RMM pool) before train() solves this and gives best search
// perf (data in HBM, not accessed over NVLink-C2C).
//
// Controlled at runtime via copy_data_to_gpu_ flag (set from index_data_on_gpu
// query parameter in build_indexes). When false, data stays in host memory —
// the caller must ensure the buffer outlives the index (use get_table_nocopy).
// ---------------------------------------------------------------------------
const float* FaissGPUIndex::copy_data_to_gpu_if_needed(
        const float* host_ptr, int64_t n) {
    if (!copy_data_to_gpu_)
        return host_ptr;
    if (!dynamic_cast<::faiss::gpu::GpuIndexCagra*>(faiss_index.get()))
        return host_ptr;

    cudaPointerAttributes attrs;
    if (cudaPointerGetAttributes(&attrs, host_ptr) == cudaSuccess &&
        attrs.type == cudaMemoryTypeDevice)
        return host_ptr;  // Already on GPU

    size_t nbytes = static_cast<size_t>(n) * D * sizeof(float);
    void* d_ptr = nullptr;
    try {
        d_ptr = ctx->pool_mr.allocate(nbytes, rmm::cuda_stream_default);
    } catch (const std::exception& e) {
        std::cerr << "[FaissGPUIndex] RMM allocate(" << (nbytes >> 20)
                  << " MiB) failed: " << e.what()
                  << ". Proceeding with host data.\n";
        return host_ptr;
    }
    auto err = cudaMemcpy(d_ptr, host_ptr, nbytes, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        ctx->pool_mr.deallocate(d_ptr, nbytes, rmm::cuda_stream_default);
        std::cerr << "[FaissGPUIndex] cudaMemcpy failed. Proceeding with host data.\n";
        return host_ptr;
    }
    device_training_data_ = d_ptr;
    device_training_data_size_ = nbytes;
    std::cout << "[FaissGPUIndex] Copied " << (nbytes >> 20)
              << " MiB dataset to GPU for CAGRA\n";
    return static_cast<const float*>(d_ptr);
}

static void print_source_data_ref(const std::shared_ptr<arrow::Array>& ref) {
    if (ref && ref->data() && ref->data()->buffers.size() > 1 && ref->data()->buffers[1]) {
        printf("[source_data_ref_] type=%s, length=%ld, buffer_size=%.2f GiB\n",
               ref->type()->ToString().c_str(), (long)ref->length(),
               ref->data()->buffers[1]->size() / (1024.0*1024.0*1024.0));
    }
}

void FaissGPUIndex::train(EmbeddingsArray& vectors) {
    auto ptr = copy_data_to_gpu_if_needed(
            raw_ptr_from_array(vectors), vectors.length());
    if (!device_training_data_) {
        source_data_ref_ = vectors.values();  // keep buffer alive for non-owning view
        print_source_data_ref(source_data_ref_);
    }
    faiss_index->train(vectors.length(), ptr);
}

void FaissGPUIndex::train(arrow::FixedSizeListArray& vectors) {
    auto ptr = copy_data_to_gpu_if_needed(
            raw_ptr_from_array(vectors), vectors.length());
    if (!device_training_data_) {
        source_data_ref_ = vectors.values();
        print_source_data_ref(source_data_ref_);
    }
    faiss_index->train(vectors.length(), ptr);
}

void FaissGPUIndex::train(arrow::LargeListArray& vectors) {
    auto ptr = copy_data_to_gpu_if_needed(
            raw_ptr_from_array(vectors), vectors.length());
    if (!device_training_data_) {
        source_data_ref_ = vectors.values();
        print_source_data_ref(source_data_ref_);
    }
    faiss_index->train(vectors.length(), ptr);
}

std::pair<std::shared_ptr<::faiss::gpu::GpuResourcesProvider>,
          std::shared_ptr<::faiss::gpu::GpuResources>>
    FaissGPUIndex::make_gpu_provider_and_resources(std::shared_ptr<MaximusContext>& ctx) {
    std::shared_ptr<::faiss::gpu::GpuResources> res;
    if (ctx) {
#if defined(MAXIMUS_WITH_CUDA) && defined(MAXIMUS_WITH_FAISS_GPUCUVS)
        // One MaximusFaissGpuResources per context, created on first GPU index
        // operation and kept alive for the process lifetime so cuBLAS/RAFT
        // handles and the pinned buffer are never re-initialized.
        static std::unordered_map<MaximusContext*, std::shared_ptr<::faiss::gpu::GpuResources>>
            s_cache;
        auto& entry = s_cache[ctx.get()];
        if (!entry) {
            entry = std::make_shared<MaximusFaissGpuResources>(ctx);
        }
        res = entry;
#else
        res = std::make_shared<MaximusFaissGpuResources>(ctx);
#endif
    } else {
        res = std::make_shared<::faiss::gpu::StandardGpuResourcesImpl>();
    }

    // GpuResourcesProviderFromInstance is a trivial shared_ptr wrapper; cheap
    // to create per call.  All indexes built from the same ctx share the same
    // underlying GpuResources — this is the intended FAISS design (GpuIndex
    // calls getResources() and stores the shared_ptr internally).
    auto provider = std::make_shared<::faiss::gpu::GpuResourcesProviderFromInstance>(res);
    return std::make_pair(provider, res);
}


::faiss::gpu::GpuIndexCagraConfig parse_cagra_config(const std::string& input) {
    std::stringstream ss(input);
    std::string token;
    std::vector<std::string> parts;

    while (std::getline(ss, token, ',')) {
        parts.push_back(token);
    }

    assert(parts.size() >= 5 && "Input must have at least 5 comma-separated parts");
    assert(parts[0] == "GPU" && "Input must start with 'GPU'");
    assert(parts[1] == "Cagra" && "Second part must be 'Cagra'");

    ::faiss::gpu::GpuIndexCagraConfig config;
    config.intermediate_graph_degree = std::stoul(parts[2]);
    config.graph_degree              = std::stoul(parts[3]);
    if (parts[4] == "IVF_PQ") {
        config.build_algo = ::faiss::gpu::graph_build_algo::IVF_PQ;
    } else if (parts[4] == "NN_DESCENT") {
        config.build_algo = ::faiss::gpu::graph_build_algo::NN_DESCENT;
    } else {
        throw std::invalid_argument("Unknown build_algo: " + parts[4]);
    }
    std::cout << "Parsed cagra config: " << "intermediate_graph_degree="
              << config.intermediate_graph_degree << ", graph_degree=" << config.graph_degree
              << ", build_algo="
              << (config.build_algo == ::faiss::gpu::graph_build_algo::IVF_PQ ? "IVF_PQ"
                                                                              : "NN_DESCENT")
              << std::endl;

    return config;
}


void FaissGPUIndex::on_load(std::unique_ptr<::faiss::Index> loaded_cpu_index) {
    int device_id        = 0;  // assuming device 0 for now
    auto [provider, res] = make_gpu_provider_and_resources(ctx);
    this->provider       = provider;

    // CASE A: HNSW CPU from cache — keep as CPU index.
    // The cached file is already a CPU IndexHNSWCagra (saved via copyTo during
    // the cold build). Loading it as CPU avoids the copyFrom_ex dangling-pointer
    // bug (faiss #4742) and the unnecessary GPU→CPU round-trip. The operator's
    // to_gpu() will use index_cpu_to_gpu() at query time — same path as the
    // cold build with index_storage_device=cpu.
    auto hnsw_cpu = dynamic_cast<::faiss::IndexHNSWCagra*>(loaded_cpu_index.get());
    if (hnsw_cpu) {
        // Extract and cache the CAGRA graph for fast to_gpu() if enabled
        if (this->cache_cagra_graph_) {
            auto& hnsw = hnsw_cpu->hnsw;
            int graph_degree = hnsw.nb_neighbors(0);
            int64_t n = hnsw_cpu->ntotal;

            auto cache = std::make_unique<CagraGraphCache>();
            cache->n_vectors = n;
            cache->graph_degree = graph_degree;
            cache->graph.resize(n * graph_degree);

            #pragma omp parallel for
            for (int64_t i = 0; i < n; ++i) {
                size_t begin, end;
                hnsw.neighbor_range(i, 0, &begin, &end);
                for (size_t j = begin; j < end; j++) {
                    cache->graph[i * graph_degree + (j - begin)] =
                        static_cast<uint32_t>(hnsw.neighbors[j]);
                }
            }
            this->cagra_graph_cache_ = std::move(cache);
            printf("[MEM] CAGRA graph cache: %ld vectors, degree=%d, size=%zu MiB\n",
                   (long)n, graph_degree, (size_t)(n * graph_degree * 4) >> 20);
        }

        FaissIndex::on_load(std::move(loaded_cpu_index));
        this->device_type = DeviceType::CPU;
        return;
    }

    // CASE A2: IVF with host view — keep as CPU for referenceFrom() in to_gpu()
    // Same pattern as CAGRA CASE A: keep CPU index, set device_type=CPU.
    // FaissGPUIndex::to_gpu() delegates to FaissIndex::to_gpu() when device_type==CPU,
    // which hits the IVF host view path (referenceFrom).
    auto* ivf_cpu = dynamic_cast<::faiss::IndexIVFFlat*>(loaded_cpu_index.get());
    if (ivf_cpu && !this->copy_data_to_gpu_ && !this->use_cuvs) {
        std::cout << "[FaissGPUIndex] IVF host view: keeping loaded index as CPU\n";
        FaissIndex::on_load(std::move(loaded_cpu_index));
        this->device_type = DeviceType::CPU;
        return;
    }

    // CASE B: STANDARD CPU -> GPU
    if (!this->ctx) throw std::runtime_error("Context is null during GPU load");

    ::faiss::gpu::GpuClonerOptions options;
    options.use_cuvs = this->use_cuvs;
    // IVFPQ GPU lookup tables: M*256*4 bytes must fit in shared memory (49152 bytes).
    // useFloat16 halves the footprint (M*256*2) and is a no-op for non-PQ indexes.
    if (this->description.find("PQ") != std::string::npos) {
        options.useFloat16 = true;
    }
    auto gpu_index_ptr =
        ::faiss::gpu::index_cpu_to_gpu(provider.get(), device_id, loaded_cpu_index.get(), &options);

    this->faiss_index.reset(gpu_index_ptr);
}

::faiss::Index* FaissGPUIndex::prepare_index_for_save(
    std::unique_ptr<::faiss::Index>& temp_storage) {
    // CASE A: CAGRA GPU -> HNSW CPU
    auto cagra_index = dynamic_cast<::faiss::gpu::GpuIndexCagra*>(this->faiss_index.get());
    if (cagra_index) {
        std::cout << "[WARNING] : Storing works (~somewhat), but lossy going from Cagra -> HnswCagra. "
                     "Also loading has issues in DGX-Sparks. Need to test on other machines."
                  << std::endl;
        auto hnsw_cpu =
            std::make_unique<::faiss::IndexHNSWCagra>(cagra_index->d, cagra_index->metric_type);
        cagra_index->copyTo(hnsw_cpu.get());

        temp_storage = std::move(hnsw_cpu);  // Give ownership to caller's holder
        return temp_storage.get();           // Return raw ptr to write
    }

    // CASE B: STANDARD GPU -> CPU
    temp_storage.reset(::faiss::gpu::index_gpu_to_cpu(this->faiss_index.get()));
    return temp_storage.get();
}

std::shared_ptr<FaissIndex> FaissGPUIndex::to_gpu() {
    if (device_type == DeviceType::CPU) {
        // Index was loaded from cache as CPU (e.g. IndexHNSWCagra).
        // Use the base class index_cpu_to_gpu() path.
        return FaissIndex::to_gpu();
    }
    return shared_from_this(); // Already on GPU, return self
}

std::shared_ptr<FaissIndex> FaissGPUIndex::to_cpu() {
    std::cout << "[FaissGPUIndex] Moving GPU index to CPU..." << std::endl;
    
    // CASE A: CAGRA GPU -> HNSW CPU
    // NOTE: You might lose some quality here because it's not a 1-1 conversion, it's lossy going from Cagra -> HNSWCagra
    auto cagra_index = dynamic_cast<::faiss::gpu::GpuIndexCagra*>(this->faiss_index.get());
    if (cagra_index) {
        std::cout << "[FaissGPUIndex] Converting Cagra -> HNSWCagra (some loss expected)" << std::endl;

        // Cache the CAGRA graph as uint32 before converting to HNSWCagra.
        // This allows to_gpu() to use copyFrom_graph() instead of copyFrom_ex(),
        // skipping HNSW graph extraction and int64→uint32 conversion each rep.
        std::unique_ptr<CagraGraphCache> graph_cache;
        if (this->cache_cagra_graph_) {
            PE("cache_cagra_graph");
            auto knn_graph_int64 = cagra_index->get_knngraph();  // D2H copy
            int graph_degree = static_cast<int>(knn_graph_int64.size() / cagra_index->ntotal);
            graph_cache = std::make_unique<CagraGraphCache>();
            graph_cache->n_vectors = cagra_index->ntotal;
            graph_cache->graph_degree = graph_degree;
            graph_cache->graph.resize(knn_graph_int64.size());
            // int64→uint32 conversion (one-time)
            for (size_t i = 0; i < knn_graph_int64.size(); ++i) {
                graph_cache->graph[i] = static_cast<uint32_t>(knn_graph_int64[i]);
            }
            printf("[MEM] CAGRA graph cache: %ld vectors, degree=%d, size=%zu MiB\n",
                   (long)graph_cache->n_vectors, graph_degree,
                   (graph_cache->graph.size() * 4) >> 20);
            PL("cache_cagra_graph");
        }

        auto hnsw_cpu = std::make_unique<::faiss::IndexHNSWCagra>(D, faiss_metric);
        PE("faiss");
        cagra_index->copyTo(hnsw_cpu.get());
        PL("faiss");

        // Eagerly free GPU resources — the HNSWCagra is self-contained after
        // copyTo. This reclaims the CAGRA graph + device_training_data_ (~N*D*4
        // bytes) immediately instead of waiting for this object's destructor.
        // Critical on ATS/unified-memory systems where GPU and CPU share the
        // same physical memory pool.
        cagra_index = nullptr;  // dangling after reset below
        faiss_index.reset();
        if (device_training_data_) {
            ctx->pool_mr.deallocate(device_training_data_, device_training_data_size_,
                                    rmm::cuda_stream_default);
            device_training_data_ = nullptr;
            device_training_data_size_ = 0;
        }

        auto result = std::make_shared<FaissIndex>(ctx, D, std::move(hnsw_cpu));
        result->description       = this->description;
        result->use_cuvs          = this->use_cuvs;
        result->copy_data_to_gpu_ = this->copy_data_to_gpu_;
        result->cache_cagra_graph_ = this->cache_cagra_graph_;
        result->cagra_graph_cache_ = std::move(graph_cache);
        std::cout << "[FaissGPUIndex] Index moved to CPU (as HNSWCagra)" << std::endl;
        return result;
    }
    
    // CASE A2: IVF host view → return source CPU index (data was never on GPU)
    // The GPU index has non-owning pointers into CPU InvertedLists held by
    // source_cpu_ref_. Just return the source CPU index — no gpu_to_cpu needed.
    if (source_cpu_ref_) {
        bool is_ivf_flat = this->description.find("IVF") != std::string::npos
                        && this->description.find("Flat") != std::string::npos;
        if (is_ivf_flat && !this->copy_data_to_gpu_) {
            std::cout << "[FaissGPUIndex] IVF host view: returning source CPU index\n";
            faiss_index.reset();  // free GPU coarse quantizer + pointer arrays
            auto result = source_cpu_ref_;
            source_cpu_ref_.reset();
            return result;
        }
    }

    // CASE B: STANDARD GPU -> CPU
    PE("faiss");
    auto cpu_index_raw = ::faiss::gpu::index_gpu_to_cpu(this->faiss_index.get());
    PL("faiss");
    auto result = std::make_shared<FaissIndex>(ctx, D, std::unique_ptr<::faiss::Index>(cpu_index_raw));
    
    std::string cpu_desc = this->description;
    if (starts_with(cpu_desc, "GPU,")) {
        cpu_desc = cpu_desc.substr(4);
    }
    result->description       = cpu_desc;
    result->use_cuvs          = this->use_cuvs;
    result->copy_data_to_gpu_ = this->copy_data_to_gpu_;

    std::cout << "[FaissGPUIndex] Index moved to CPU successfully" << std::endl;
    return result;
}

}  // namespace maximus::faiss
