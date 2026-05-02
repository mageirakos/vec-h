#pragma once

#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/abstract_scatter_operator.hpp>
#include <maximus/operators/gpu/gpu_operator.hpp>

namespace maximus::cudf {

/**
 * @brief GPU-based scatter operator using cudf.
 *
 * Scatters input data by a key column, distributing rows to different output ports
 * based on a key-agnostic mapping: unique key values are sorted and assigned
 * sequential partition indices (smallest key -> port 0, next -> port 1, etc.).
 *
 * This follows the LocalBroadcast pattern for multi-port output but instead of
 * replicating data to all ports, it splits data by key value.
 */
class ScatterOperator
        : public maximus::AbstractScatterOperator
        , public maximus::gpu::GpuOperator {
public:
    ScatterOperator(std::shared_ptr<MaximusContext>& ctx,
                    std::shared_ptr<Schema> input_schema,
                    std::shared_ptr<ScatterProperties> properties);

    void on_add_input(DeviceTablePtr device_input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking, int port) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl(int port) override;

    void run_kernel(std::shared_ptr<MaximusContext>& ctx,
                    std::vector<CudfTablePtr>& input_tables,
                    std::vector<CudfTablePtr>& output_tables) override;

private:
    int num_partitions_ = 1;
    std::vector<int> port_progress_;

    // Stores one table per partition after partitioning is complete
    std::vector<CudfTablePtr> partition_tables_;

    // Track if partitioning has been done
    bool partitioning_done_ = false;
};

}  // namespace maximus::cudf
