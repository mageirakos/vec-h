#include <maximus/operators/abstract_hash_join_operator.hpp>

namespace maximus {

maximus::AbstractHashJoinOperator::AbstractHashJoinOperator(
    std::shared_ptr<MaximusContext> &ctx,
    std::shared_ptr<Schema> left_schema,
    std::shared_ptr<Schema> right_schema,
    std::shared_ptr<JoinProperties> properties)
        : AbstractOperator(PhysicalOperatorType::HASH_JOIN, ctx)
        , properties(std::move(properties)) {
    std::vector<std::shared_ptr<Schema>> inputs;
    inputs.reserve(2);
    inputs.emplace_back(std::move(left_schema));
    inputs.emplace_back(std::move(right_schema));
    assign_input_schemas(inputs);

    if (get_build_port() >= 0) {
        set_blocking_port(get_build_port());
    }
}

int AbstractHashJoinOperator::get_build_port() const {
    return 1;
}

int AbstractHashJoinOperator::get_probe_port() const {
    return 0;
}
}  // namespace maximus