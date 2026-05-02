#pragma once

#include <faiss/Index.h>

#include <maximus/indexes/index.hpp>
#include <maximus/types/types.hpp>
#include <memory>
#include <string>

#ifdef MAXIMUS_WITH_CUDA
#include <cudf/column/column_view.hpp>
#endif

namespace maximus::faiss {

using MetricType           = ::faiss::MetricType;
using VectorDistanceMetric = maximus::VectorDistanceMetric;

/**
 * FaissIndex - Wrapper for FAISS CPU indexes
 *
 * This class wraps FAISS CPU indexes for use with Maximus operators.
 *
 * USAGE RECOMMENDATIONS:
 * - CPU JoinIndexedOperator + CPU FaissIndex: ✅ Best: no implicit copies
 * - GPU JoinIndexedOperator + CPU FaissIndex: ⚠️ Anti-pattern: implicit D2H copy
 *
 * While FAISS may handle some cross-device operations internally, it's better to:
 * 1. Match index type to operator type (CPU index with CPU operator)
 * 2. Use DeviceTablePtr conversions to explicitly manage data placement
 * 3. Avoid relying on implicit H2D/D2H copies within the FAISS library
 *
 * For GPU operators, prefer FaissGPUIndex to avoid unnecessary data transfers.
 */

class FaissIndex : public Index, public std::enable_shared_from_this<FaissIndex> {
public:
    FaissIndex(std::shared_ptr<MaximusContext>& ctx, int d, MetricType metric)
            : Index(ctx), D(d), faiss_metric(metric) {
        engine_type = EngineType::FAISS;
        device_type = DeviceType::CPU;
    }

    FaissIndex(std::shared_ptr<MaximusContext>& ctx,
               int d,
               std::unique_ptr<::faiss::Index> raw_faiss_index)
            : FaissIndex(ctx, d, raw_faiss_index->metric_type) {
        faiss_index = std::move(raw_faiss_index);
        assert(faiss_index->d == this->D);
        assert(faiss_index->metric_type == this->faiss_metric);
    }

    ~FaissIndex() override;

    static std::shared_ptr<FaissIndex> factory_make(std::shared_ptr<MaximusContext>& ctx,
                                                    int d,
                                                    const std::string& description,
                                                    MetricType faiss_metric = MetricType::METRIC_L2,
                                                    bool use_cuvs = true);


    static std::shared_ptr<FaissIndex> build(std::shared_ptr<MaximusContext>& ctx,
                                             DeviceTablePtr& data_table,
                                             const std::string& vector_column,
                                             const std::string& description,
                                             VectorDistanceMetric metric,
                                             bool use_cache,
                                             const std::string& cache_dir = "./index_cache",
                                             bool use_cuvs = true,
                                             bool copy_to_gpu = false,
                                             bool cache_graph = false);

    virtual void train(EmbeddingsArray& vectors);

    virtual void add(EmbeddingsArray& vectors);

    virtual void search(EmbeddingsArray& vectors,
                        int k,
                        arrow::FloatArray& distances,
                        arrow::Int64Array& labels,
                        const ::faiss::SearchParameters* params = nullptr);

    virtual void train(arrow::FixedSizeListArray& vectors);

    virtual void add(arrow::FixedSizeListArray& vectors);

    virtual void train(arrow::LargeListArray& vectors);

    virtual void add(arrow::LargeListArray& vectors);

    virtual void search(arrow::LargeListArray& vectors,
                        int k,
                        arrow::FloatArray& distances,
                        arrow::Int64Array& labels,
                        const ::faiss::SearchParameters* params = nullptr);

    virtual void search(arrow::FixedSizeListArray& vectors,
                        int k,
                        arrow::FloatArray& distances,
                        arrow::Int64Array& labels,
                        const ::faiss::SearchParameters* params = nullptr);

    virtual void range_search(int n, const float* x);

#ifdef MAXIMUS_WITH_CUDA
    virtual void train(::cudf::column_view& vectors);

    virtual void add(::cudf::column_view& vectors);

    virtual void search(::cudf::column_view const& vectors,
                        int k,
                        ::cudf::column& distances,
                        ::cudf::column& labels,
                        const ::faiss::SearchParameters* params = nullptr);
#endif

    std::string to_string();

    bool is_trained() const override {
        return faiss_index && faiss_index->is_trained;
    }

    // Move index between devices
    // to_gpu: Returns GPU version of this index. Creates FaissGPUIndex from CPU index.
    //         Returns nullptr if already on GPU or if CUDA not available.
    virtual std::shared_ptr<FaissIndex> to_gpu();

    // to_cpu: Returns CPU version of this index. Creates FaissIndex from GPU index.
    //         Returns self if already on CPU.
    virtual std::shared_ptr<FaissIndex> to_cpu();

    /// Pin the index's embedding data (flat codes buffer) in host memory via cudaHostRegister
    /// for faster H2D transfer. Only the raw vector data is pinned — graph structures (HNSW
    /// neighbors, CAGRA graph cache), centroids, and other metadata remain in pageable memory.
    /// Supported: Flat, HNSW (via storage), PQ. Not supported: IVF (data in inverted lists).
    /// No-op if already pinned, on GPU, or if CUDA is not available.
    void pin();

    /// Unpin previously pinned memory. Called automatically by destructor.
    void unpin();

protected:
    static bool cache_exists(const std::string& index_path);
    bool load_from_disk(const std::string& cache_key, const std::string& cache_dir);
    void save_to_disk(const std::string& cache_key, const std::string& cache_dir);
    // hooks:
    virtual void on_load(std::unique_ptr<::faiss::Index> loaded_cpu_index);
    virtual ::faiss::Index* prepare_index_for_save(std::unique_ptr<::faiss::Index>& temp_storage);

public:
    std::unique_ptr<::faiss::Index> faiss_index;
    int D;
    MetricType faiss_metric = MetricType::METRIC_L2;
    bool use_cuvs = true;  // Runtime cuVS toggle (only relevant for GPU indexes)
    bool is_pinned_ = false;  // True if index codes are page-locked via cudaHostRegister
    /// When true, index data is explicitly copied to GPU before search.
    /// For CAGRA: forces GPU-resident dataset via cuVS ATS flag (data=1 vs data=0).
    /// For IVF: uses copyInvertedListsFrom() instead of referenceFrom() host view.
    /// On discrete GPUs: redundant (always copies, no ATS).
    /// Survives to_cpu()/to_gpu() round-trip. Set via index_data_on_gpu query param.
    bool copy_data_to_gpu_ = false;

    /// When true, to_gpu() uses cached graph path (copyFrom_graph) instead of
    /// copyFrom_ex(). Architecture-independent optimization — just avoids HNSW
    /// graph extraction + int64->uint32 conversion on each to_gpu() call.
    /// Requires APPLY_FAISS_CAGRA_PATCH. Set via cagra_cache_graph param.
    bool cache_cagra_graph_ = false;

    /// Cached CAGRA graph in uint32 format. Populated during to_cpu() when
    /// cache_cagra_graph_ is true. Used by to_gpu() to skip copyFrom_ex().
    struct CagraGraphCache {
        std::vector<uint32_t> graph;  // Dense [n_vectors x graph_degree]
        int graph_degree = 0;
        int64_t n_vectors = 0;
    };
    std::unique_ptr<CagraGraphCache> cagra_graph_cache_;
    
    VectorDistanceMetric metric() const override {
        return (faiss_metric == MetricType::METRIC_INNER_PRODUCT) ? 
               VectorDistanceMetric::INNER_PRODUCT : VectorDistanceMetric::L2;
    }
    
};

class FaissSearchParameters : public IndexParameters {
public:
    FaissSearchParameters(std::shared_ptr<::faiss::SearchParameters> params)
            : params(std::move(params)) {
        assert(this->params != nullptr);
    }
    std::shared_ptr<::faiss::SearchParameters> params = nullptr;
};

using FaissIndexPtr = std::shared_ptr<FaissIndex>;

}  // namespace maximus::faiss