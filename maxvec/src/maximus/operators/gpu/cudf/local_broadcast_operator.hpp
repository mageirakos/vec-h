#pragma once

#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/abstract_local_broadcast_operator.hpp>
#include <maximus/operators/gpu/gpu_operator.hpp>

namespace maximus::cudf {

class LocalBroadcastOperator
        : public maximus::AbstractLocalBroadcastOperator
        , public maximus::gpu::GpuOperator {
public:
    LocalBroadcastOperator(std::shared_ptr<MaximusContext>& ctx,
                           std::shared_ptr<Schema> input_schema,
                           std::shared_ptr<LocalBroadcastProperties> properties);

    void on_add_input(DeviceTablePtr device_input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking, int port) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl(int port) override;

    void run_kernel(std::shared_ptr<MaximusContext>& ctx,
                    std::vector<CudfTablePtr>& input_tables,
                    std::vector<CudfTablePtr>& output_tables) override;

private:
    bool input_finished_flag = false, replicate_flag = false;
    std::vector<int> broadcasted_ports;
    int num_broadcast_ports = 1, num_inputs = 0;
    std::vector<int> port_progress;
    std::vector<DeviceTablePtr> transferred_output_tables;
};

}  // namespace maximus::cudf
