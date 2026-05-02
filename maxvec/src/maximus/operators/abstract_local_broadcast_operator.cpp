#include <maximus/operators/abstract_local_broadcast_operator.hpp>

namespace maximus {

AbstractLocalBroadcastOperator::AbstractLocalBroadcastOperator(
    std::shared_ptr<MaximusContext> &ctx,
    std::shared_ptr<Schema> input_schema,
    std::shared_ptr<LocalBroadcastProperties> properties)
        : AbstractOperator(PhysicalOperatorType::LOCAL_BROADCAST, ctx, input_schema, input_schema)
        , properties(std::move(properties)) {
    set_blocking_port(0);
}

bool AbstractLocalBroadcastOperator::has_more_batches_impl(bool blocking) {
    throw std::runtime_error("LocalBroadcastOperator::has_more_batches not "
                             "supported! The output port has to be specified.");
}

DeviceTablePtr AbstractLocalBroadcastOperator::export_next_batch_impl() {
    throw std::runtime_error("LocalBroadcastOperator::export_next_batch not "
                             "supported! The output port has to be specified.");
}
}  // namespace maximus