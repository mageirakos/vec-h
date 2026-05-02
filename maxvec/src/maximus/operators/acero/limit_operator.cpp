#include <maximus/operators/acero/limit_operator.hpp>

namespace maximus::acero {

LimitOperator::LimitOperator(std::shared_ptr<MaximusContext> &ctx,
                             std::shared_ptr<Schema> input_schema,
                             std::shared_ptr<LimitProperties> properties)
        : AbstractLimitOperator(ctx, std::move(input_schema), std::move(properties)) {
    auto options = std::make_shared<arrow::acero::FetchNodeOptions>(this->properties->offset,
                                                                    this->properties->limit);

    // initializing the engine: creates mini source, mini node and mini sink nodes
    // creating the engine
    proxy_operator = std::make_unique<ProxyOperator>(
        ctx, input_schemas, std::move(options), "fetch", get_id(), next_engine_type, next_op_type);

    // retrieve the output schema from the proxy operator
    output_schema = proxy_operator->output_schema;
    assert(output_schema);

    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::ACERO);

    proxy_operator->operator_name = name();
}

void LimitOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);
    assert(proxy_operator && "A proxy operator must be initialized before adding input.");
    // pass the batch through the engine
    proxy_operator->on_add_input(device_input, port);
}

void LimitOperator::on_no_more_input(int port) {
    assert(proxy_operator && "A proxy operator must be initialized before adding input.");

    // inform the engine that there is no more input
    proxy_operator->on_no_more_input(port);
}

bool LimitOperator::has_more_batches_impl(bool blocking) {
    return proxy_operator->has_more_batches_impl(blocking);
}

DeviceTablePtr LimitOperator::export_next_batch_impl() {
    return std::move(proxy_operator->export_next_batch_impl());
}

}  // namespace maximus::acero
