#pragma once

#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/abstract_gather_operator.hpp>
#include <maximus/operators/gpu/gpu_operator.hpp>

namespace maximus::cudf {

/**
 * @brief GPU-based gather operator using cudf.
 *
 * Collects data from multiple input ports and concatenates them into a single
 * output using cudf::concatenate. This is the inverse of ScatterOperator.
 */
class GatherOperator
        : public maximus::AbstractGatherOperator
        , public maximus::gpu::GpuOperator {
public:
    GatherOperator(std::shared_ptr<MaximusContext>& ctx,
                   std::shared_ptr<Schema> input_schema,
                   std::shared_ptr<GatherProperties> properties);

    void on_add_input(DeviceTablePtr device_input, int port) override;

    void on_no_more_input(int port) override;

    void run_kernel(std::shared_ptr<MaximusContext>& ctx,
                    std::vector<CudfTablePtr>& input_tables,
                    std::vector<CudfTablePtr>& output_tables) override;

private:
    int num_inputs_ = 0;
    std::vector<bool> port_finished_;
    std::vector<std::vector<CudfTablePtr>> port_batches_;
    bool concatenation_done_ = false;
    CudfTablePtr concatenated_result_;
    bool result_exported_ = false;
};

}  // namespace maximus::cudf
