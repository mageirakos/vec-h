#include <maximus/operators/acero/project_operator.hpp>

namespace maximus::acero {

ProjectOperator::ProjectOperator(std::shared_ptr<MaximusContext>& ctx,
                                 std::shared_ptr<Schema> input_schema,
                                 std::shared_ptr<ProjectProperties> properties)
        : AbstractProjectOperator(ctx, std::move(input_schema), std::move(properties)) {
    auto expressions  = this->properties->project_expressions;
    auto column_names = this->properties->column_names;
    assert(expressions.size() == column_names.size() &&
           "Number of expressions must match number of column names.");

    // the options we need to create the engine's operator
    std::vector<arrow::Expression> exprs;
    exprs.reserve(expressions.size());
    for (auto& e : expressions) {
        auto arrow_expr = e->get_expression();
        exprs.emplace_back(*arrow_expr);
    }

    auto options = std::make_shared<arrow::acero::ProjectNodeOptions>(std::move(exprs),
                                                                      std::move(column_names));

    // initializing the engine: creates mini source, mini node and mini sink nodes
    // creating the engine
    proxy_operator = std::make_unique<ProxyOperator>(ctx,
                                                     input_schemas,
                                                     std::move(options),
                                                     "project",
                                                     get_id(),
                                                     next_engine_type,
                                                     next_op_type);

    // retrieve the output schema from the proxy operator
    output_schema = proxy_operator->output_schema;
    assert(output_schema);

    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::ACERO);

    proxy_operator->operator_name = name();
}

void ProjectOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);
    assert(proxy_operator && "A proxy operator must be initialized before adding input.");
    // pass the batch through the engine
    proxy_operator->on_add_input(device_input, port);
}

void ProjectOperator::on_no_more_input(int port) {
    assert(proxy_operator && "A proxy operator must be initialized before adding input.");

    // inform the engine that there is no more input
    proxy_operator->on_no_more_input(port);
}

bool ProjectOperator::has_more_batches_impl(bool blocking) {
    return proxy_operator->has_more_batches_impl(blocking);
}

DeviceTablePtr ProjectOperator::export_next_batch_impl() {
    return std::move(proxy_operator->export_next_batch_impl());
}

}  // namespace maximus::acero
