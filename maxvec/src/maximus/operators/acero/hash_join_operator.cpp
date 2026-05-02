#include <maximus/operators/acero/hash_join_operator.hpp>

namespace maximus::acero {

HashJoinOperator::HashJoinOperator(std::shared_ptr<MaximusContext> &ctx,
                                   std::shared_ptr<Schema> left_schema,
                                   std::shared_ptr<Schema> right_schema,
                                   std::shared_ptr<JoinProperties> properties)
        : maximus::AbstractHashJoinOperator(ctx, left_schema, right_schema, std::move(properties)) {
    // don't take the values by reference
    // since they might be changed internally for acero
    // e.g. LEFT_JOIN might internally be reinterpreted
    // by Acero as RIGHT_JOIN, however, to the outer query plan
    // this should still be LEFT_JOIN
    // for this reason, we don't modify the properties
    // only the local copies of properties
    auto join_type    = this->properties->join_type;
    auto left_keys    = this->properties->left_keys;
    auto right_keys   = this->properties->right_keys;
    auto left_suffix  = this->properties->left_suffix;
    auto right_suffix = this->properties->right_suffix;
    auto &filter      = this->properties->filter;

    bool switch_sides = false;
    switch (join_type) {
        case JoinType::RIGHT_SEMI:
            join_type    = JoinType::LEFT_SEMI;
            switch_sides = true;
            break;
        case JoinType::RIGHT_ANTI:
            join_type    = JoinType::LEFT_ANTI;
            switch_sides = true;
            break;
        case JoinType::RIGHT_OUTER:
            join_type    = JoinType::LEFT_OUTER;
            switch_sides = true;
            break;
        default:
            break;
    }

    if (switch_sides) {
        std::swap(left_schema, right_schema);
        std::swap(left_keys, right_keys);
        std::swap(left_suffix, right_suffix);
    }

    // creating the engine
    auto options = std::make_shared<arrow::acero::HashJoinNodeOptions>(
        static_cast<arrow::acero::JoinType>(join_type),
        std::move(left_keys),
        std::move(right_keys),
        *(filter->get_expression()),
        left_suffix,
        right_suffix);

    assert(options->key_cmp.size() == options->left_keys.size() &&
           "Key comparison functions must be provided for all keys.");
    // std::cout << "LEFT SCHEMA = " << left_schema->to_string() << std::endl;
    // std::cout << "RIGHT SCHEMA = " << right_schema->to_string() << std::endl;

    // creating the engine
    proxy_operator = std::make_unique<ProxyOperator>(ctx,
                                                     input_schemas,
                                                     std::move(options),
                                                     "hashjoin",
                                                     get_id(),
                                                     next_engine_type,
                                                     next_op_type,
                                                     std::vector<int>{get_build_port()});

    // retrieve the output schema from the engine
    output_schema = proxy_operator->output_schema;
    // std::cout << "OUTPUT SCHEMA = " << output_schema->to_string() << std::endl;
    assert(output_schema);

    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::ACERO);

    proxy_operator->operator_name = name();
}

void HashJoinOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);
    assert(proxy_operator && "The proxy operator must be initialized before adding input.");

    // pass the batch through the engine
    proxy_operator->on_add_input(device_input, port);
}

void HashJoinOperator::on_no_more_input(int port) {
    // std::cout << "No more input for hash_join on port " << port << std::endl;
    assert(proxy_operator && "The proxy operator must be initialized before adding input.");

    // inform the engine that there is no more input
    proxy_operator->on_no_more_input(port);
    // std::cout << "Finished no more input for hash_join on port " << port << std::endl;
}

int HashJoinOperator::get_build_port() const {
    return 1;
}

int HashJoinOperator::get_probe_port() const {
    return 0;
}

bool HashJoinOperator::has_more_batches_impl(bool blocking) {
    return proxy_operator->has_more_batches_impl(blocking);
}

DeviceTablePtr HashJoinOperator::export_next_batch_impl() {
    return std::move(proxy_operator->export_next_batch_impl());
}

}  // namespace maximus::acero
