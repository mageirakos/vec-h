#include <maximus/operators/abstract_limit_operator.hpp>

namespace maximus {

AbstractLimitOperator::AbstractLimitOperator(std::shared_ptr<MaximusContext> &ctx,
                                             std::shared_ptr<Schema> input_schema,
                                             std::shared_ptr<LimitProperties> properties)
        : AbstractOperator(PhysicalOperatorType::LIMIT, ctx, input_schema, input_schema)
        , properties(std::move(properties)) {
}

}  // namespace maximus