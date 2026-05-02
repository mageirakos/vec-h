#pragma once

#include <arrow/array.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexCagra.h>

#include "faiss_index.hpp"

namespace maximus::faiss {

using MetricType = ::faiss::MetricType;

/**
 * FaissGPUIndex - Wrapper for FAISS GPU indexes (GpuIndexFlat, GpuIndexCagra, etc.)
 *
 * This class wraps FAISS GPU indexes for use with Maximus GPU operators.
 *
 * NOTE: We've handled the following in the operators. GPU operator uses GPU index. CPU operator uses CPU index. Handles index movement as well 
 * USAGE RECOMMENDATIONS:
 * - GPU JoinIndexedOperator + GPU FaissGPUIndex: ✅ ✅ Best: no implicit copies
 * - CPU JoinIndexedOperator + GPU FaissGPUIndex: ⚠️ Anti-pattern: implicit copies (H2D queries + D2H results)
 *
 * While FAISS GPU indexes can accept host pointers and perform implicit H2D/D2H copies, it's better to:
 * 1. Match index type to operator type (GPU index with GPU operator)
 * 2. Use DeviceTablePtr conversions to control data placement explicitly
 * 3. Let Maximus's memory management handle transfers, not FAISS internals
 *
 * For CPU operators, prefer FaissIndex (CPU) to avoid per-query copy overhead.
 */
class FaissGPUIndex : public FaissIndex {
public:
    FaissGPUIndex(std::shared_ptr<MaximusContext>& ctx,
                  int d,
                  std::unique_ptr<::faiss::gpu::GpuIndex> index);

    FaissGPUIndex(std::shared_ptr<MaximusContext>& ctx,
                  int d,
                  const std::string& description,
                  MetricType metric,
                  bool use_cuvs = true);

    FaissGPUIndex(std::shared_ptr<MaximusContext>& ctx,
                  int d,
                  std::unique_ptr<::faiss::gpu::GpuIndex> index,
                  std::shared_ptr<::faiss::gpu::GpuResourcesProvider> provider);

    ~FaissGPUIndex() override;

    // Override train() to handle ATS/unified memory for CAGRA
    void train(EmbeddingsArray& vectors) override;
    void train(arrow::FixedSizeListArray& vectors) override;
    void train(arrow::LargeListArray& vectors) override;

    std::shared_ptr<FaissIndex> to_gpu() override;
    std::shared_ptr<FaissIndex> to_cpu() override;

    static std::pair<std::shared_ptr<::faiss::gpu::GpuResourcesProvider>,
                     std::shared_ptr<::faiss::gpu::GpuResources>>
        make_gpu_provider_and_resources(std::shared_ptr<MaximusContext>& ctx);

protected:
    void on_load(std::unique_ptr<::faiss::Index> loaded_cpu_index) override;
    ::faiss::Index* prepare_index_for_save(std::unique_ptr<::faiss::Index>& temp_storage) override;

protected:
    // For Cagra index, we need to hold a provider
    std::shared_ptr<::faiss::gpu::GpuResourcesProvider> provider;

public:
    /// Device copy of training data (allocated from ctx->pool_mr, freed in destructor).
    /// Must outlive the faiss index since CuvsCagra stores storage_ = x.
    /// Public so FaissIndex::to_gpu() can set it for the cached graph path.
    void* device_training_data_ = nullptr;
    size_t device_training_data_size_ = 0;

    // CAGRA with data_on_gpu=0: cuVS holds a zero-copy host view of the embedding
    // buffer. The backing memory must stay alive for the GPU index's lifetime.
    // source_data_ref_ covers the build-from-scratch path (Arrow array owns buffer).
    // source_cpu_ref_ covers the to_gpu() path: keeps the source FaissIndex alive
    // via shared_from_this(), so its faiss_index (HNSWCagra with IndexFlat storage)
    // remains valid. Non-destructive — the source index stays intact for reuse.
    std::shared_ptr<arrow::Array> source_data_ref_;
    std::shared_ptr<FaissIndex> source_cpu_ref_;

private:
    /// Copy dataset to GPU if copy_data_to_gpu_ is true and index is CAGRA.
    const float* copy_data_to_gpu_if_needed(const float* host_ptr, int64_t n);
};

::faiss::gpu::GpuIndexCagraConfig parse_cagra_config(const std::string& input);

}  // namespace maximus::faiss