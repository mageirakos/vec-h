#pragma once

#include <maximus/operators/abstract_vector_join_operator.hpp>
#include <maximus/operators/gpu/gpu_operator.hpp>

namespace maximus::faiss::gpu {

class JoinExhaustiveOperator
        : public maximus::AbstractVectorJoinOperator
        , public maximus::gpu::GpuOperator {
public:
    JoinExhaustiveOperator(std::shared_ptr<MaximusContext> &ctx,
                            std::vector<std::shared_ptr<Schema>> input_schemas,
                            std::shared_ptr<VectorJoinExhaustiveProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;

    void run_kernel(std::shared_ptr<MaximusContext>& ctx,
                    std::vector<CudfTablePtr>& input_tables,
                    std::vector<CudfTablePtr>& output_tables) override;

private:
    std::shared_ptr<VectorJoinExhaustiveProperties> properties;
};
}  // namespace maximus::faiss
