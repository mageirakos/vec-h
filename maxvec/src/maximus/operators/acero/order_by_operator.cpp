#include <maximus/operators/acero/order_by_operator.hpp>

namespace maximus::acero {

OrderByOperator::OrderByOperator(std::shared_ptr<MaximusContext>& ctx,
                                 std::shared_ptr<Schema> input_schema,
                                 std::shared_ptr<OrderByProperties> properties)
        : AbstractOrderByOperator(ctx, std::move(input_schema), std::move(properties)) {
    arrow::compute::NullPlacement null_placement = this->properties->null_order == NullOrder::FIRST
                                                       ? arrow::compute::NullPlacement::AtStart
                                                       : arrow::compute::NullPlacement::AtEnd;

    std::vector<arrow::compute::SortKey> sort_keys;
    sort_keys.reserve(this->properties->sort_keys.size());

    for (auto& sort_key : this->properties->sort_keys) {
        auto& field = sort_key.field;
        auto& order = sort_key.order;

        if (order == SortOrder::ASCENDING) {
            sort_keys.emplace_back(field, arrow::compute::SortOrder::Ascending);
        } else {
            sort_keys.emplace_back(field, arrow::compute::SortOrder::Descending);
        }
    }

    arrow::compute::Ordering ordering(sort_keys, null_placement);

    auto options = std::make_shared<arrow::acero::OrderByNodeOptions>(std::move(ordering));
    assert(options);

    // initializing the engine: creates mini source, mini node and mini sink nodes
    proxy_operator = std::make_unique<ProxyOperator>(ctx,
                                                     input_schemas,
                                                     std::move(options),
                                                     "order_by",
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

void OrderByOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);
    assert(proxy_operator && "A proxy operator must be initialized before adding input.");

    // pass the batch through the engine
    proxy_operator->on_add_input(device_input, port);
}

void OrderByOperator::on_no_more_input(int port) {
    assert(proxy_operator && "A proxy operator must be initialized before adding input.");

    // inform the engine that there is no more input
    proxy_operator->on_no_more_input(port);
}

bool OrderByOperator::has_more_batches_impl(bool blocking) {
    return proxy_operator->has_more_batches_impl(blocking);
}

DeviceTablePtr OrderByOperator::export_next_batch_impl() {
    return std::move(proxy_operator->export_next_batch_impl());
}

}  // namespace maximus::acero
