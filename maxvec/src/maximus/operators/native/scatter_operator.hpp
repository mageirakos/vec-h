#pragma once

#include <maximus/operators/abstract_scatter_operator.hpp>

namespace maximus::native {

class ScatterOperator : public AbstractScatterOperator {
public:
    ScatterOperator(std::shared_ptr<MaximusContext>& ctx,
                    std::shared_ptr<Schema> input_schema,
                    std::shared_ptr<ScatterProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;
    void on_no_more_input(int port) override;

    // Multi-port output
    [[nodiscard]] bool has_more_batches_impl(bool blocking, int port) override;
    [[nodiscard]] DeviceTablePtr export_next_batch_impl(int port) override;

private:
    std::vector<TableBatchPtr> input_batches_;

    // Partitioned outputs: one per output port
    std::vector<std::vector<DeviceTablePtr>> partition_outputs_;
    std::vector<int> current_batch_index_;
};

}  // namespace maximus::native
