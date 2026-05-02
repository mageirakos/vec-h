#pragma once

#include <cudf/sorting.hpp>
#include <maximus/operators/abstract_order_by_operator.hpp>
#include <maximus/operators/gpu/gpu_operator.hpp>

namespace maximus::cudf {

class OrderByOperator
        : public maximus::AbstractOrderByOperator
        , public maximus::gpu::GpuOperator {
public:
    OrderByOperator(std::shared_ptr<MaximusContext>& ctx,
                    std::shared_ptr<Schema> input_schema,
                    std::shared_ptr<OrderByProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;

    void run_kernel(std::shared_ptr<MaximusContext>& ctx,
                    std::vector<CudfTablePtr>& input_tables,
                    std::vector<CudfTablePtr>& output_tables) override;

private:
    std::vector<int> key_indices;
    std::vector<::cudf::order> key_orders;
    std::vector<::cudf::null_order> null_orders;
};

}  // namespace maximus::cudf
