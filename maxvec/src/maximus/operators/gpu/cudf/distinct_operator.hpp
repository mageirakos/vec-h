#pragma once

#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/abstract_distinct_operator.hpp>
#include <maximus/operators/gpu/gpu_operator.hpp>

namespace maximus::cudf {

class DistinctOperator
        : public maximus::AbstractDistinctOperator
        , public maximus::gpu::GpuOperator {
public:
    DistinctOperator(std::shared_ptr<MaximusContext>& ctx,
                     std::shared_ptr<Schema> input_schema,
                     std::shared_ptr<DistinctProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;

    void run_kernel(std::shared_ptr<MaximusContext>& ctx,
                    std::vector<CudfTablePtr>& input_tables,
                    std::vector<CudfTablePtr>& output_tables) override;

private:
    std::vector<int> key_indices;
};

}  // namespace maximus::cudf