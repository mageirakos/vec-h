#pragma once

#include <maximus/operators/abstract_vector_project_distance_operator.hpp>
#include <maximus/operators/gpu/gpu_operator.hpp>

namespace maximus::faiss::gpu {

class ProjectDistanceOperator
        : public maximus::AbstractVectorProjectDistanceOperator
        , public maximus::gpu::GpuOperator {
public:
    ProjectDistanceOperator(std::shared_ptr<MaximusContext> &ctx,
                            std::vector<std::shared_ptr<Schema>> input_schemas,
                            std::shared_ptr<VectorProjectDistanceProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;

    void run_kernel(std::shared_ptr<MaximusContext>& ctx,
                    std::vector<CudfTablePtr>& input_tables,
                    std::vector<CudfTablePtr>& output_tables) override;
};
}  // namespace maximus::faiss
