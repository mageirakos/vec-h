#include <maximus/operators/native/hash_join_operator.hpp>

namespace maximus::native {

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

    // the native Cross Join doesn't support additional filters
    assert(!filter);
    // the native Cross Join doesn't support left or right keys
    assert(left_keys.empty());
    assert(right_keys.empty());

    // the current implementation only supports CROSS_JOIN
    assert(join_type == JoinType::CROSS_JOIN);

    // this should already be guaranteed from the parent class, but just in case
    assert(left_schema);
    assert(right_schema);
    auto left_arrow_schema  = left_schema->get_schema();
    auto right_arrow_schema = right_schema->get_schema();

    // TODO: add a suffix to left schema (if non-empty)
    // TODO: add a suffix to right schema (if non-empty)

    // TODO: create the output schema
    // output_schema = std::make_shared<Schema>(left_arrow_schema);
    assert(output_schema);

    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::NATIVE);
}

void HashJoinOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);

    const auto &operator_name = name();
    profiler::close_regions({operator_name, "add_input"});
    device_input.convert_to<TableBatchPtr>(ctx_, input_schemas[port]);
    profiler::open_regions({operator_name, "add_input"});
    auto input = device_input.as_table_batch();

    assert(port < 2 && "The port must be either 0 or 1.");

    // if the input came from the accumulating side
    if (port == get_build_port()) {
        accumulated_batches.push_back(input);
    } else {
        assert(port == get_probe_port());
        assert(no_more_input_[get_build_port()]);
        // if the input came from the streaming side
        // TODO: invoke the concatenate function
        // TODO: push the result to the outputs_ vector
    }
}

void HashJoinOperator::on_no_more_input(int port) {
    return;
}

int HashJoinOperator::get_build_port() const {
    return 1;
}

int HashJoinOperator::get_probe_port() const {
    return 0;
}

}  // namespace maximus::native