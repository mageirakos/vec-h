#include <maximus/operators/acero/group_by_operator.hpp>

namespace maximus::acero {

GroupByOperator::GroupByOperator(std::shared_ptr<MaximusContext> &ctx,
                                 std::shared_ptr<Schema> input_schema,
                                 std::shared_ptr<GroupByProperties> properties)
        : AbstractGroupByOperator(ctx, std::move(input_schema), std::move(properties)) {
    std::vector<arrow::compute::Aggregate> acero_aggregates;

    for (auto &aggr : this->properties->aggregates) {
        acero_aggregates.emplace_back(*aggr->get_aggregate());
    }

    auto options = std::make_shared<arrow::acero::AggregateNodeOptions>(
        std::move(acero_aggregates), this->properties->group_keys);

    // initializing the engine: creates mini source, mini node and mini sink nodes
    // creating the engine
    proxy_operator = std::make_unique<ProxyOperator>(ctx,
                                                     input_schemas,
                                                     std::move(options),
                                                     "aggregate",
                                                     get_id(),
                                                     next_engine_type,
                                                     next_op_type,
                                                     std::vector<int>{0});

    // retrieve the output schema from the proxy operator
    output_schema = proxy_operator->output_schema;
    assert(output_schema);

    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::ACERO);

    proxy_operator->operator_name = name();
}

void GroupByOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);
    assert(proxy_operator && "A proxy operator must be initialized before adding input.");

    // pass the batch through the engine
    proxy_operator->on_add_input(device_input, port);
}

void GroupByOperator::on_no_more_input(int port) {
    assert(proxy_operator && "A proxy operator must be initialized before adding input.");

    // inform the engine that there is no more input
    proxy_operator->on_no_more_input(port);
}

bool GroupByOperator::has_more_batches_impl(bool blocking) {
    return proxy_operator->has_more_batches_impl(blocking);
}

DeviceTablePtr GroupByOperator::export_next_batch_impl() {
    return std::move(proxy_operator->export_next_batch_impl());
}

}  // namespace maximus::acero
