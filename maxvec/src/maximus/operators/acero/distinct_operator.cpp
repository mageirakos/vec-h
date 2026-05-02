#include <maximus/operators/acero/distinct_operator.hpp>

namespace maximus {
namespace acero {

DistinctOperator::DistinctOperator(std::shared_ptr<MaximusContext> &ctx,
                                   std::shared_ptr<Schema> input_schema,
                                   std::shared_ptr<DistinctProperties> properties)
        : AbstractDistinctOperator(ctx, std::move(input_schema), std::move(properties)) {
    // if distinct keys are not provided, we use all the fields in the schema
    auto distinct_keys = this->properties->distinct_keys;
    if (distinct_keys.empty()) {
        for (const auto &field : input_schema->get_schema()->fields()) {
            distinct_keys.push_back(field->name());
        }
    }

    auto options = std::make_shared<arrow::acero::AggregateNodeOptions>(
        std::vector<arrow::compute::Aggregate>(), std::move(distinct_keys));

    // initializing the engine: creates mini source, mini node and mini sink nodes
    // creating the engine
    proxy_operator = std::make_unique<ProxyOperator>(ctx,
                                                     input_schemas,
                                                     std::move(options),
                                                     "aggregate",
                                                     get_id(),
                                                     next_engine_type,
                                                     next_op_type);

    // retrieve the output schema from the engine
    this->output_schema = proxy_operator->output_schema;
    assert(output_schema);

    set_engine_type(EngineType::ACERO);
    set_device_type(DeviceType::CPU);

    proxy_operator->operator_name = name();
}

void DistinctOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);
    assert(proxy_operator && "The proxy operator must be initialized before adding input.");

    // pass the batch through the engine
    proxy_operator->on_add_input(device_input, port);
}

void DistinctOperator::on_no_more_input(int port) {
    assert(proxy_operator && "The proxy operator must be initialized before adding input.");

    // inform the engine that there is no more input
    proxy_operator->on_no_more_input(port);
}

bool DistinctOperator::has_more_batches_impl(bool blocking) {
    return proxy_operator->has_more_batches_impl(blocking);
}

DeviceTablePtr DistinctOperator::export_next_batch_impl() {
    return std::move(proxy_operator->export_next_batch_impl());
}

}  // namespace acero
}  // namespace maximus
