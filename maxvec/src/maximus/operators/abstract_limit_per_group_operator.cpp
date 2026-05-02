#include <maximus/operators/abstract_limit_per_group_operator.hpp>

namespace maximus {

AbstractLimitPerGroupOperator::AbstractLimitPerGroupOperator(
    std::shared_ptr<MaximusContext>& ctx,
    std::shared_ptr<Schema> input_schema,
    std::shared_ptr<LimitPerGroupProperties> properties)
        : AbstractOperator(PhysicalOperatorType::LIMIT_PER_GROUP, ctx, input_schema, input_schema)
        , properties(std::move(properties)) {
}

}  // namespace maximus
