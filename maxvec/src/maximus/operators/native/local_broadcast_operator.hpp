#pragma once

#include <maximus/operators/abstract_local_broadcast_operator.hpp>
#include <maximus/operators/properties.hpp>

namespace maximus::native {

class LocalBroadcastOperator : public AbstractLocalBroadcastOperator {
public:
    LocalBroadcastOperator(std::shared_ptr<MaximusContext>& ctx,
                           std::shared_ptr<Schema> input_schema,
                           std::shared_ptr<LocalBroadcastProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking, int port) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl(int port) override;

protected:
    // for each output port, determines which batch it is currently processing
    std::vector<std::size_t> current_batch_index;
    // for each batch, determines the set of output ports that have already received it
    // std::vector<std::unordered_set<int>> ports_who_received_batch;
    std::vector<int> num_ports_who_received_batch;
};

}  // namespace maximus::native
