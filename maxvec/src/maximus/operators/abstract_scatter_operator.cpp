#include <maximus/operators/abstract_scatter_operator.hpp>

namespace maximus {

AbstractScatterOperator::AbstractScatterOperator(
    std::shared_ptr<MaximusContext>& ctx,
    std::shared_ptr<Schema> input_schema,
    std::shared_ptr<ScatterProperties> properties)
        : AbstractOperator(PhysicalOperatorType::SCATTER, ctx, input_schema, input_schema)
        , properties(std::move(properties)) {
    set_blocking_port(0);
}

bool AbstractScatterOperator::has_more_batches_impl(bool blocking) {
    throw std::runtime_error("ScatterOperator::has_more_batches not "
                             "supported! The output port has to be specified.");
}

DeviceTablePtr AbstractScatterOperator::export_next_batch_impl() {
    throw std::runtime_error("ScatterOperator::export_next_batch not "
                             "supported! The output port has to be specified.");
}

}  // namespace maximus
