#pragma once

#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/abstract_limit_per_group_operator.hpp>
#include <maximus/operators/gpu/gpu_operator.hpp>

namespace maximus::cudf {

class LimitPerGroupOperator
        : public maximus::AbstractLimitPerGroupOperator
        , public maximus::gpu::GpuOperator {
public:
    LimitPerGroupOperator(std::shared_ptr<MaximusContext>& ctx,
                          std::shared_ptr<Schema> input_schema,
                          std::shared_ptr<LimitPerGroupProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;

    void run_kernel(std::shared_ptr<MaximusContext>& ctx,
                    std::vector<CudfTablePtr>& input_tables,
                    std::vector<CudfTablePtr>& output_tables) override;
};

}  // namespace maximus::cudf
