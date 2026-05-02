#pragma once

#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/abstract_fused_operator.hpp>
#include <maximus/operators/gpu/gpu_operator.hpp>

namespace maximus::cudf {

class FusedOperator
        : public maximus::AbstractFusedOperator
        , public maximus::gpu::GpuOperator {
public:
    FusedOperator(std::shared_ptr<MaximusContext> &ctx,
                  std::shared_ptr<Schema> input_schema,
                  std::shared_ptr<FusedProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;

private:
    std::vector<std::shared_ptr<maximus::gpu::GpuOperator>> operators;
    bool streaming, input_finished_flag = false;
};

}  // namespace maximus::cudf