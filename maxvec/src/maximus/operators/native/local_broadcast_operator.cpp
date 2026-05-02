#include <maximus/operators/native/local_broadcast_operator.hpp>

namespace maximus::native {

LocalBroadcastOperator::LocalBroadcastOperator(std::shared_ptr<MaximusContext>& ctx,
                                               std::shared_ptr<Schema> input_schema,
                                               std::shared_ptr<LocalBroadcastProperties> properties)
        : AbstractLocalBroadcastOperator(ctx, std::move(input_schema), std::move(properties)) {
    current_batch_index.resize(this->properties->num_output_ports, 0);
    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::NATIVE);
}

void LocalBroadcastOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);

    const auto& operator_name = name();
    profiler::close_regions({operator_name, "add_input"});
    device_input.convert_to<TableBatchPtr>(ctx_, input_schemas[port]);
    profiler::open_regions({operator_name, "add_input"});

    auto input = device_input.as_table_batch();

    assert(port == 0);
    assert(input);
    outputs_.emplace_back(std::move(input));
}

void LocalBroadcastOperator::on_no_more_input(int port) {
    assert(port == 0);
    // no output port has received any batch yet
    num_ports_who_received_batch.resize(outputs_.size());
}

bool LocalBroadcastOperator::has_more_batches_impl(bool blocking, int port) {
    assert(port < current_batch_index.size());
    assert(current_batch_index[port] <= outputs_.size());
    return current_batch_index[port] < outputs_.size();
}

DeviceTablePtr LocalBroadcastOperator::export_next_batch_impl(int port) {
    assert(has_more_batches(true, port));
    assert(port < current_batch_index.size());
    assert(current_batch_index[port] < outputs_.size());

    const auto batch_index = current_batch_index[port];

    assert(batch_index < outputs_.size());
    assert(outputs_[batch_index]);

    // one more output port has received this batch
    ++num_ports_who_received_batch[batch_index];

    // increase the index of the current batch for this output port
    ++current_batch_index[port];

    // if all output ports have already received this batch
    // then we can move it out of the operator
    // each output port will have its own std::shared_ptr managing the lifetime of the batch
    // but the current operator will not need to keep a reference to it anymore
    if (num_ports_who_received_batch[batch_index] == properties->num_output_ports) {
        return std::move(this->outputs_[batch_index]);
    }

    // if replicate is on, the clone the batch
    if (properties->should_replicate) {
        // if we need to replicate the batch, we need to create a new copy of it
        // so that each output port has its own copy
        outputs_[batch_index].convert_to<TableBatchPtr>(ctx_, input_schemas[port]);
        auto batch = outputs_[batch_index].as_table_batch();
        return DeviceTablePtr(std::move(batch->clone()));
    }

    // otherwise, if the replication is not enabled and
    // not all the output ports have received this batch, and we don't replicate,
    // we still keep a reference to that batch and send it to the output port
    return this->outputs_[batch_index];
}
}  // namespace maximus::native