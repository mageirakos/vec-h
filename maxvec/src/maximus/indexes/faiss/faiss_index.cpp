#include <faiss/index_factory.h>

#include <maximus/indexes/faiss/faiss_index.hpp>
#include <maximus/operators/faiss/interop.hpp>
#include <maximus/utils/utils.hpp>
#include <maximus/profiler/profiler.hpp>
#include <string>


#if defined(MAXIMUS_WITH_CUDA) && defined(MAXIMUS_WITH_FAISS_GPUCUVS)
#include <maximus/indexes/faiss/faiss_gpu_index.hpp>
#include <maximus/operators/faiss/gpu/faiss_kernels.hpp>
#include <maximus/utils/cudf_helpers.hpp>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <atomic>
// Runtime ATS flag declared in raft/neighbors/dataset.hpp (via APPLY_RAFT_ATS_PATCH).
// When the patch is not applied, this extern resolves to nothing (linker ignores unused).
namespace raft::neighbors::detail {
    extern std::atomic<bool> cagra_ats_force_copy_enabled;
}
#endif
#ifdef MAXIMUS_WITH_CUDA
#include <cuda_runtime.h>
#endif
#include <faiss/index_io.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexFlatCodes.h>
#include <faiss/IndexHNSW.h>

#include <filesystem>


namespace maximus::faiss {

// Helper: get the flat codes storage (embedding data) from a FAISS index.
// Returns nullptr for IVF indexes (data scattered across inverted lists).
static ::faiss::IndexFlatCodes* get_flat_codes_storage(::faiss::Index* index) {
    if (!index) return nullptr;
    auto* flat = dynamic_cast<::faiss::IndexFlatCodes*>(index);
    if (flat) return flat;
    auto* hnsw = dynamic_cast<::faiss::IndexHNSW*>(index);
    if (hnsw && hnsw->storage) {
        return dynamic_cast<::faiss::IndexFlatCodes*>(hnsw->storage);
    }
    return nullptr;
}

FaissIndex::~FaissIndex() {
    unpin();
}

void FaissIndex::pin() {
#ifdef MAXIMUS_WITH_CUDA
    if (is_pinned_ || !faiss_index) return;
    if (device_type != DeviceType::CPU) return;

    auto* flat = get_flat_codes_storage(faiss_index.get());
    if (!flat) {
        // NOTE: in theory you could pin each inverted list but complex, unecessary to make our point at the moment
        std::cout << "[FaissIndex] Warning: cpu-pinned requested but index type '"
                  << description
                  << "' has no flat codes buffer to pin (e.g. IVF stores data in inverted lists). "
                  << "Proceeding with pageable memory.\n";
        return;
    }
    if (flat->ntotal == 0) return;  // empty index, nothing to pin

    void* ptr = flat->codes.data();
    size_t size = flat->codes.byte_size();

    auto err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
    if (err == cudaSuccess) {
        is_pinned_ = true;
        std::cout << "[FaissIndex] Pinned " << (size / (1024 * 1024))
                  << " MiB of index data for fast H2D transfer\n";
    } else {
        std::cout << "[FaissIndex] cudaHostRegister failed: " << cudaGetErrorString(err)
                  << " (proceeding with pageable memory)\n";
    }
#endif
}

void FaissIndex::unpin() {
#ifdef MAXIMUS_WITH_CUDA
    if (!is_pinned_ || !faiss_index) return;

    auto* flat = get_flat_codes_storage(faiss_index.get());
    if (flat && flat->codes.data()) {
        cudaHostUnregister(flat->codes.data());
    }
    is_pinned_ = false;
#endif
}


bool FaissIndex::cache_exists(const std::string& index_path) {
    // Check if both files exist and are non-empty
    return std::filesystem::exists(index_path) && std::filesystem::file_size(index_path) > 0;
}


// Returns true if loaded successfully, false if cache file not found.
bool FaissIndex::load_from_disk(const std::string& cache_key, const std::string& cache_dir) {
    if (!this->faiss_index) {
        throw std::runtime_error("Attempted to load an uninitialized index.");
    }
    std::filesystem::path cache_path = std::filesystem::absolute(cache_dir);
    std::filesystem::path bin_path_p = cache_path / (cache_key + ".index");
    std::string bin_path             = bin_path_p.string();

    // 1. Check existence
    if (!cache_exists(bin_path)) {
        // if cache exists and storing is atomic (w/ the tmp path) you've essentially validated the index
        // description, vector column, meric, dimension, data length, and exact undelrying data
        return false;
    }

    // 2. Load the raw index from disk (always CPU resident initially)
    std::unique_ptr<::faiss::Index> loaded_cpu_index;
    try {
        loaded_cpu_index.reset(::faiss::read_index(bin_path.c_str()));
    } catch (const std::exception& e) {
        std::cout << "[CACHE DEBUG] faiss::read_index threw exception: " << e.what() << std::endl;
        return false;
    }

    // Delegate to child class for final loading steps
    this->on_load(std::move(loaded_cpu_index));
    std::cout << "Loaded FAISS index from cache: '" << bin_path << "'" << std::endl;
    return true;
}

// Generic Save Workflow
void FaissIndex::save_to_disk(const std::string& cache_key, const std::string& cache_dir) {
    if (!this->faiss_index) throw std::runtime_error("Attempted to save uninitialized index.");

    try {
        std::filesystem::create_directories(cache_dir);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create cache dir: " + std::string(e.what()));
    }

    std::filesystem::path cache_path = std::filesystem::absolute(cache_dir);

    try {
        std::filesystem::create_directories(cache_path);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create cache dir: " + std::string(e.what()));
    }

    std::filesystem::path bin_path_p  = cache_path / (cache_key + ".index");
    std::filesystem::path temp_path_p = bin_path_p;
    temp_path_p += ".tmp";

    std::string bin_path  = bin_path_p.string();
    std::string temp_path = temp_path_p.string();

    // Holder for potential GPU->CPU temporary object
    std::unique_ptr<::faiss::Index> temp_storage;

    // DELEGATE to child to get the writable pointer
    ::faiss::Index* index_to_write = this->prepare_index_for_save(temp_storage);
    assert(index_to_write != nullptr);

    try {
        ::faiss::write_index(index_to_write, temp_path.c_str());
        std::filesystem::rename(temp_path, bin_path);
    } catch (...) {
        if (std::filesystem::exists(temp_path)) std::filesystem::remove(temp_path);
        throw;
    }
    std::cout << "Saved FAISS index to cache: '" << bin_path << "'" << std::endl;
}


void FaissIndex::on_load(std::unique_ptr<::faiss::Index> loaded_cpu_index) {
    // CPU default: Just take the pointer
    std::cout << "Loading CPU index from cache..." << std::endl;
    this->faiss_index = std::move(loaded_cpu_index);
}

::faiss::Index* FaissIndex::prepare_index_for_save(std::unique_ptr<::faiss::Index>& temp_storage) {
    // CPU default: Just save myself
    return this->faiss_index.get();
}


std::shared_ptr<FaissIndex> FaissIndex::build(std::shared_ptr<MaximusContext>& ctx,
                                              DeviceTablePtr& data_table,
                                              const std::string& vector_column,
                                              const std::string& description,
                                              VectorDistanceMetric metric,
                                              bool use_cache,
                                              const std::string& cache_dir,
                                              bool use_cuvs,
                                              bool copy_to_gpu,
                                              bool cache_graph) {
    // 1. Schema & Data Prep
    std::shared_ptr<Schema> schema;
    if (data_table.is_table_batch()) {
        schema = data_table.as_table_batch()->get_schema();
    } else if (data_table.is_table()) {
        schema = data_table.as_table()->get_schema();
    } else if (data_table.is_gtable()) {
        schema = data_table.as_gtable()->get_schema();
    } else {
        throw std::runtime_error("Unsupported DeviceTablePtr type");
    }
    //  vector_column must exist in base schema - no renames
    if (!schema->get_schema() || schema->get_schema()->GetFieldIndex(vector_column) < 0) {
        throw std::invalid_argument("Vector column '" + vector_column +
                                    "' not found in table schema");
    }

    data_table.convert_to<TableBatchPtr>(ctx, schema);

    auto tb           = data_table.as_table_batch();
    auto vector_array = tb->get_table_batch()->GetColumnByName(vector_column);

    // std::cout << "Data table \n" << tb->to_string() << std::endl;

    int64_t data_length = vector_array->length();
    // std::cout << "data length: " << data_length << std::endl;
    int dimension       = maximus::embedding_dimension(vector_array);

    // std::cout << "Dimension = " << dimension << std::endl;

    if (description.empty()) throw std::invalid_argument("Index description cannot be empty");

    ::faiss::MetricType faiss_metric = to_faiss_metric(metric);

    // 2. Create the "Shell" Object
    // This creates the correct C++ wrapper (FaissGPUIndex or FaissIndex)
    // and initializes the underlying FAISS pointer (GpuIndexCagra, GpuIndexFlat, etc.)
    auto index = FaissIndex::factory_make(ctx, dimension, description, faiss_metric, use_cuvs);

#if defined(MAXIMUS_WITH_CUDA) && defined(MAXIMUS_WITH_FAISS_GPUCUVS)
    // Set GPU copy flag for CAGRA indexes (copies dataset to GPU before train)
    if (copy_to_gpu) {
        auto* gpu_idx = dynamic_cast<FaissGPUIndex*>(index.get());
        if (gpu_idx) gpu_idx->copy_data_to_gpu_ = true;
    }
#endif

    // Set graph cache flag before load_from_disk so on_load() can populate it
    index->cache_cagra_graph_ = cache_graph;

    // 3. Handle Cache (only compute expensive hash if caching is enabled)
    std::string cache_key;
    if (use_cache) {
        std::string data_digest  = compute_arrow_data_hash(vector_array, dimension);
        std::string build_config = description + "_" + vector_column + "_" +
                                   maximus::metric_short_name(faiss_metric) + "_d" +
                                   std::to_string(dimension) + "_nb" + std::to_string(data_length) +
                                   (use_cuvs ? "_cuvs" : "_nocuvs");
        cache_key = build_config + "_" + data_digest;

        // Attempt Load
        if (index->load_from_disk(cache_key, cache_dir)) {
            return index;
        }
        std::cout << "Cache miss. Building new FAISS index: " << cache_key << std::endl;
    }

    // 4. Warn about GPU + LargeListArray combination
    bool is_large_list = (vector_array->type()->id() == arrow::Type::LARGE_LIST);
    if (is_large_list && index->device_type == DeviceType::GPU) {
        std::cout << "[WARNING] Building GPU index from LargeListArray (int64 offsets). "
                  << "cuDF only supports int32 offsets. Drop the '" << vector_column
                  << "' column after build; the index owns the data." << std::endl;
    }

    // 5. Train & Add
    PE("faiss_index_build");
    if (vector_array->type()->id() == arrow::Type::LIST) {
        auto list_array = std::static_pointer_cast<EmbeddingsArray>(vector_array);
        index->train(*list_array);
        index->add(*list_array);
    } else if (vector_array->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
        auto fsl_array = std::static_pointer_cast<arrow::FixedSizeListArray>(vector_array);
        index->train(*fsl_array);
        index->add(*fsl_array);
    } else if (is_large_list) {
        auto large_list_array = std::static_pointer_cast<arrow::LargeListArray>(vector_array);
        index->train(*large_list_array);
        index->add(*large_list_array);
    } else {
        throw std::runtime_error("Unsupported vector column type");
    }
    PL("faiss_index_build");

    // 6. Save for next time
    if (use_cache) {
        index->save_to_disk(cache_key, cache_dir);
    }

    // Debug output
    ::faiss::Index* raw_faiss_index = index->faiss_index.get();
    std::cout << "=== Index Built ===" << std::endl;
    std::cout << "Type: " << index->to_string() << std::endl;
    std::cout << "Vectors: " << raw_faiss_index->ntotal << std::endl;
    std::cout << "Trained: " << raw_faiss_index->is_trained << std::endl;

    return index;
}

std::shared_ptr<FaissIndex> FaissIndex::factory_make(std::shared_ptr<MaximusContext>& ctx,
                                                     int d,
                                                     const std::string& description,
                                                     MetricType metric,
                                                     bool use_cuvs) {
    if (description.empty()) {
        throw std::invalid_argument("Description cannot be empty");
    }

#if defined(MAXIMUS_WITH_CUDA) && defined(MAXIMUS_WITH_FAISS_GPUCUVS)
    if (starts_with(description, "GPU")) {
        return std::make_shared<FaissGPUIndex>(ctx, d, description, metric, use_cuvs);
    }
#else
    else if (starts_with(description, "GPU")) {
        throw std::invalid_argument("GPU indexes not available");
    }
#endif
    else {
        auto raw_faiss_index =
            std::unique_ptr<::faiss::Index>(::faiss::index_factory(d, description.data(), metric));
        auto faiss_index         = std::make_shared<FaissIndex>(ctx, d, std::move(raw_faiss_index));
        faiss_index->description = description;
        faiss_index->use_cuvs    = use_cuvs;
        return faiss_index;
    }
};

void FaissIndex::train(EmbeddingsArray& vectors) {
    faiss_index->train(vectors.length(), raw_ptr_from_array(vectors));
};

void FaissIndex::add(EmbeddingsArray& vectors) {
    faiss_index->add(vectors.length(), raw_ptr_from_array(vectors));
};

void FaissIndex::search(EmbeddingsArray& vectors,
                        int k,
                        arrow::FloatArray& distances,
                        arrow::Int64Array& labels,
                        const ::faiss::SearchParameters* params) {
    assert(labels.length() >= vectors.length() * k);
    assert(distances.length() >= vectors.length() * k);
    assert(faiss_index->metric_type == this->faiss_metric);
    faiss_index->search(vectors.length(),
                        raw_ptr_from_array(vectors),
                        k,
                        (float*) distances.raw_values(),
                        (int64_t*) labels.raw_values(),
                        params);
};


void FaissIndex::train(arrow::FixedSizeListArray& vectors) {
    faiss_index->train(vectors.length(), raw_ptr_from_array(vectors));
};

void FaissIndex::add(arrow::FixedSizeListArray& vectors) {
    faiss_index->add(vectors.length(), raw_ptr_from_array(vectors));
};

void FaissIndex::train(arrow::LargeListArray& vectors) {
    faiss_index->train(vectors.length(), raw_ptr_from_array(vectors));
};

void FaissIndex::add(arrow::LargeListArray& vectors) {
    faiss_index->add(vectors.length(), raw_ptr_from_array(vectors));
};

void FaissIndex::search(arrow::LargeListArray& vectors,
                        int k,
                        arrow::FloatArray& distances,
                        arrow::Int64Array& labels,
                        const ::faiss::SearchParameters* params) {
    // 1. Check GPU compatibility
    if (this->device_type == DeviceType::GPU) {
        throw std::runtime_error("GPU search with LargeListArray (int64 offsets) is unsupported. "
                                 "cuDF requires int32 offsets.");
    }

    assert(labels.length() >= vectors.length() * k);
    assert(distances.length() >= vectors.length() * k);
    assert(faiss_index->metric_type == this->faiss_metric);
    faiss_index->search(vectors.length(),
                        raw_ptr_from_array(vectors),
                        k,
                        (float*) distances.raw_values(),
                        (int64_t*) labels.raw_values(),
                        params);
};

void FaissIndex::search(arrow::FixedSizeListArray& vectors,
                        int k,
                        arrow::FloatArray& distances,
                        arrow::Int64Array& labels,
                        const ::faiss::SearchParameters* params) {
    assert(labels.length() >= vectors.length() * k);
    assert(distances.length() >= vectors.length() * k);
    assert(faiss_index->metric_type == this->faiss_metric);
    faiss_index->search(vectors.length(),
                        raw_ptr_from_array(vectors),
                        k,
                        (float*) distances.raw_values(),
                        (int64_t*) labels.raw_values(),
                        params);
};

void FaissIndex::range_search(int n, const float* x) {
    // TODO
    throw std::runtime_error("range search is not implemented for FaissIndex");
};

#ifdef MAXIMUS_WITH_CUDA
void FaissIndex::train(::cudf::column_view& vectors) {
    assert(D > 0);
    auto ptr        = gpu::get_embedding_ptr(vectors, D);
    int num_vectors = gpu::get_num_vectors(vectors);
    faiss_index->train(num_vectors, ptr);
};

void FaissIndex::add(::cudf::column_view& vectors) {
    assert(D > 0);
    auto ptr        = gpu::get_embedding_ptr(vectors, D);
    int num_vectors = gpu::get_num_vectors(vectors);
    faiss_index->add(num_vectors, ptr);
};

void FaissIndex::search(::cudf::column_view const& vectors,
                        int k,
                        ::cudf::column& distances,
                        ::cudf::column& labels,
                        const ::faiss::SearchParameters* params) {
    assert(vectors.type().id() == ::cudf::type_id::LIST);
    assert(distances.type().id() == ::cudf::type_id::FLOAT32);
    assert(labels.type().id() == ::cudf::type_id::INT64);

    assert(labels.size() >= vectors.size() * k);
    assert(distances.size() >= vectors.size() * k);
    assert(faiss_index->metric_type == this->faiss_metric);

    auto num_vectors = gpu::get_num_vectors(vectors);
    auto ptr         = gpu::get_embedding_ptr(vectors, D);

    faiss_index->search(num_vectors,
                        ptr,
                        k,
                        get_device_data_ptr<float>(distances),
                        get_device_data_ptr<int64_t>(labels),
                        params);
};
#endif

std::string FaissIndex::to_string() {
    return description + ":" + maximus::metric_short_name(this->faiss_metric);
};

std::shared_ptr<FaissIndex> FaissIndex::to_cpu() {
    // Already on CPU, return self
    return shared_from_this();
}

std::shared_ptr<FaissIndex> FaissIndex::to_gpu() {
#if defined(MAXIMUS_WITH_CUDA) && defined(MAXIMUS_WITH_FAISS_GPUCUVS)
    std::cout << "[FaissIndex] Moving CPU index to GPU..." << std::endl;

    // Set the raft ATS runtime flag for CAGRA data=1. This makes cuVS's
    // make_strided_dataset() use ptr_attrs.type check instead of devicePointer
    // check, causing host data to be copied to GPU internally by cuVS.
    // Requires APPLY_RAFT_ATS_PATCH=true (runtime patch variant).
    bool is_cagra = this->description.find("Cagra") != std::string::npos;
    bool set_ats_flag = is_cagra && this->copy_data_to_gpu_;

    // =========================================================================
    // FAST PATH: Cached CAGRA graph → copyFrom_graph() (skips copyFrom_ex)
    // =========================================================================
    if (cagra_graph_cache_ && cache_cagra_graph_ && is_cagra) {
        std::cout << "[FaissIndex] Using cached graph path (copyFrom_graph)"
                  << (set_ats_flag ? " [data=1: ATS force copy]" : " [data=0: host view]")
                  << "\n";

        auto [provider, res] = FaissGPUIndex::make_gpu_provider_and_resources(ctx);
        auto config = parse_cagra_config(this->description);

        // Get dataset from HNSWCagra's IndexFlat storage
        auto* hnsw_cagra = dynamic_cast<::faiss::IndexHNSWCagra*>(faiss_index.get());
        if (!hnsw_cagra || !hnsw_cagra->storage) {
            throw std::runtime_error("[FaissIndex] cached graph path: index is not IndexHNSWCagra");
        }
        auto* flat_storage = dynamic_cast<::faiss::IndexFlat*>(hnsw_cagra->storage);
        if (!flat_storage) {
            throw std::runtime_error("[FaissIndex] cached graph path: storage is not IndexFlat");
        }
        const float* dataset_ptr = flat_storage->get_xb();

        // Set raft flag BEFORE construction — cuVS will check it in
        // make_strided_dataset() and either copy (flag=true) or view (flag=false).
        if (set_ats_flag) {
            raft::neighbors::detail::cagra_ats_force_copy_enabled.store(true, std::memory_order_relaxed);
        }

        PE("faiss");
        auto gpu_cagra = std::make_unique<::faiss::gpu::GpuIndexCagra>(
            provider.get(), D, faiss_metric, config);
        gpu_cagra->copyFrom_graph(
            cagra_graph_cache_->graph.data(),
            dataset_ptr,
            cagra_graph_cache_->n_vectors,
            cagra_graph_cache_->graph_degree);
        PL("faiss");

        // Reset flag immediately after construction
        if (set_ats_flag) {
            raft::neighbors::detail::cagra_ats_force_copy_enabled.store(false, std::memory_order_relaxed);
        }

        auto result = std::make_shared<FaissGPUIndex>(
            ctx, D,
            std::unique_ptr<::faiss::gpu::GpuIndex>(gpu_cagra.release()),
            provider);
        result->description        = this->description;
        result->use_cuvs           = this->use_cuvs;
        result->copy_data_to_gpu_  = this->copy_data_to_gpu_;
        result->cache_cagra_graph_ = this->cache_cagra_graph_;

        // Keep host data alive: copyFrom_graph() created a zero-copy view of
        // the HNSWCagra's IndexFlat buffer. Hold a shared_ptr back to the
        // source FaissIndex so its faiss_index (and the buffer) stays valid.
        // Non-destructive: the source index remains intact for reuse.
        if (!this->copy_data_to_gpu_) {
            result->source_cpu_ref_ = shared_from_this();
        }

        std::cout << "[FaissIndex] Index moved to GPU (cached graph path)" << std::endl;
        return result;
    }

    // =========================================================================
    // IVF HOST VIEW PATH: referenceFrom() — ATS zero-copy, no data H2D
    // Requires: use_cuvs=0, IVFFlat index, copy_data_to_gpu_=false,
    //           APPLY_FAISS_IVF_PATCH (adds referenceFrom/referenceInvertedListsFrom)
    // =========================================================================
    bool is_ivf_flat = this->description.find("IVF") != std::string::npos
                    && this->description.find("Flat") != std::string::npos;
    if (is_ivf_flat && !this->copy_data_to_gpu_ && !this->use_cuvs) {
        auto* cpu_ivf = dynamic_cast<::faiss::IndexIVFFlat*>(faiss_index.get());
        if (cpu_ivf) {
            std::cout << "[FaissIndex] Using IVF host view path (referenceFrom)\n";

            auto [provider, res] = FaissGPUIndex::make_gpu_provider_and_resources(ctx);

            // Config: interleavedLayout=false (GPU reads CPU's flat layout),
            // use_cuvs=false (cuVS forces interleaved), INDICES_64_BIT (direct ptr ref)
            ::faiss::gpu::GpuIndexIVFFlatConfig config;
            config.interleavedLayout = false;
            config.indicesOptions = ::faiss::gpu::INDICES_64_BIT;
            config.use_cuvs = false;

            PE("faiss");
            auto gpu_ivf = std::make_unique<::faiss::gpu::GpuIndexIVFFlat>(
                provider.get(), cpu_ivf->d, cpu_ivf->nlist,
                cpu_ivf->metric_type, config);
            gpu_ivf->referenceFrom(cpu_ivf);
            PL("faiss");

            auto result = std::make_shared<FaissGPUIndex>(
                ctx, D,
                std::unique_ptr<::faiss::gpu::GpuIndex>(gpu_ivf.release()),
                provider);
            result->description       = this->description;
            result->use_cuvs          = false;
            result->copy_data_to_gpu_ = false;
            // Keep CPU index alive — its InvertedLists are referenced by GPU
            result->source_cpu_ref_   = shared_from_this();

            std::cout << "[FaissIndex] Index moved to GPU (IVF host view)\n";
            return result;
        }
    }

    // =========================================================================
    // NORMAL PATH: index_cpu_to_gpu → copyFrom_ex (standard Faiss path)
    // =========================================================================
    std::string gpu_desc = description;
    if (!starts_with(gpu_desc, "GPU,")) {
        gpu_desc = "GPU," + gpu_desc;
    }

    auto [provider, res] = FaissGPUIndex::make_gpu_provider_and_resources(ctx);

    ::faiss::gpu::GpuClonerOptions options;
    options.use_cuvs = this->use_cuvs;
    if (this->description.find("PQ") != std::string::npos) {
        options.useFloat16 = true;
    }

    // Set raft flag for data=1 — cuVS will copy host data inside copyFrom_ex()
    if (set_ats_flag) {
        std::cout << "[FaissIndex] Normal path with data=1: setting ATS force copy flag\n";
        raft::neighbors::detail::cagra_ats_force_copy_enabled.store(true, std::memory_order_relaxed);
    }

    PE("faiss");
    auto gpu_index_raw = ::faiss::gpu::index_cpu_to_gpu(
        provider.get(), 0, this->faiss_index.get(), &options);
    PL("faiss");

    if (set_ats_flag) {
        raft::neighbors::detail::cagra_ats_force_copy_enabled.store(false, std::memory_order_relaxed);
    }

    auto result = std::make_shared<FaissGPUIndex>(
        ctx, D,
        std::unique_ptr<::faiss::gpu::GpuIndex>(
            dynamic_cast<::faiss::gpu::GpuIndex*>(gpu_index_raw)),
        provider);
    result->description       = this->description;
    result->use_cuvs          = this->use_cuvs;
    result->copy_data_to_gpu_ = this->copy_data_to_gpu_;
    result->cache_cagra_graph_ = this->cache_cagra_graph_;

    // Keep host data alive for CAGRA with data_on_gpu=0 (same as fast path).
    if (is_cagra && !this->copy_data_to_gpu_) {
        result->source_cpu_ref_ = shared_from_this();
    }

    std::cout << "[FaissIndex] Index moved to GPU successfully" << std::endl;
    return result;
#else
    throw std::runtime_error("GPU indexes not available - CUDA/cuVS not compiled");
#endif
}

}  // namespace maximus::faiss