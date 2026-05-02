#include <maximus/operators/acero/fused_operator.hpp>
#include <maximus/operators/acero/interop.hpp>
#include <maximus/operators/acero/properties.hpp>

namespace maximus {
namespace acero {

FusedOperator::FusedOperator(std::shared_ptr<MaximusContext>& ctx,
                             std::shared_ptr<Schema> input_schema,
                             std::shared_ptr<FusedProperties> properties)
        : AbstractFusedOperator(ctx, std::move(input_schema), std::move(properties)) {
    auto& node_types      = this->properties->node_types;
    auto& node_properties = this->properties->properties;

    assert(node_types.size() == node_properties.size());
    assert(node_types.size() > 1);

    std::vector<std::string> acero_node_names;
    acero_node_names.reserve(node_types.size());

    std::vector<std::shared_ptr<arrow::acero::ExecNodeOptions>> acero_node_options;
    acero_node_options.reserve(node_types.size());

    for (unsigned i = 0u; i < node_types.size(); ++i) {
        acero_node_names.emplace_back(to_acero_node_name(node_types[i], node_properties[i]));
        acero_node_options.emplace_back(to_acero_options(ctx_, node_properties[i]));
    }

    // initializing the engine: creates mini source, mini node and mini sink nodes
    // creating the engine
    proxy_operator = std::make_unique<ProxyOperator>(ctx,
                                                     input_schemas,
                                                     std::move(acero_node_options),
                                                     std::move(acero_node_names),
                                                     get_id(),
                                                     next_engine_type,
                                                     next_op_type);

    // retrieve the output schema from the engine
    this->output_schema = proxy_operator->output_schema;
    assert(output_schema);

    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::ACERO);

    proxy_operator->operator_name = name();
}

void FusedOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);
    assert(proxy_operator && "The proxy operator must be initialized before adding input.");

    // pass the batch through the engine
    proxy_operator->on_add_input(device_input, port);
}

void FusedOperator::on_no_more_input(int port) {
    // auto start = std::chrono::high_resolution_clock::now();
    assert(proxy_operator && "The proxy operator must be initialized before adding input.");

    // inform the engine that there is no more input
    proxy_operator->on_no_more_input(port);
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // std::cout << "FusedOperator::on_no_more_input: " << duration.count() << " ms" << std::endl;
}

bool FusedOperator::has_more_batches_impl(bool blocking) {
    return proxy_operator->has_more_batches_impl(blocking);
}

DeviceTablePtr FusedOperator::export_next_batch_impl() {
    return std::move(proxy_operator->export_next_batch_impl());
}

}  // namespace acero
}  // namespace maximus
