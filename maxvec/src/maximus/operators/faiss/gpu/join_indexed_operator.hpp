#pragma once

#include <maximus/indexes/faiss/faiss_gpu_index.hpp>
#include <maximus/operators/abstract_vector_join_operator.hpp>
#include <maximus/operators/gpu/gpu_operator.hpp>
#include <maximus/indexes/faiss/faiss_index.hpp>

namespace maximus::faiss::gpu {

class JoinIndexedOperator
        : public maximus::AbstractVectorJoinOperator
        , public maximus::gpu::GpuOperator {
public:
    JoinIndexedOperator(std::shared_ptr<MaximusContext> &ctx,
                            std::vector<std::shared_ptr<Schema>> input_schemas,
                            std::shared_ptr<VectorJoinIndexedProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;

    void run_kernel(std::shared_ptr<MaximusContext>& ctx,
                    std::vector<CudfTablePtr>& input_tables,
                    std::vector<CudfTablePtr>& output_tables) override;

private:
    std::shared_ptr<VectorJoinIndexedProperties> properties;
    std::shared_ptr<::faiss::SearchParameters> search_parameters;
    std::shared_ptr<FaissIndex> index;
};
}  // namespace maximus::faiss
