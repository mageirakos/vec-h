#include <maximus/operators/abstract_distinct_operator.hpp>

namespace maximus {

maximus::AbstractDistinctOperator::AbstractDistinctOperator(
    std::shared_ptr<MaximusContext> &ctx,
    std::shared_ptr<Schema> input_schema,
    std::shared_ptr<DistinctProperties> properties)
        : AbstractOperator(PhysicalOperatorType::DISTINCT, ctx, input_schema, input_schema)
        , properties(std::move(properties)) {
    set_blocking_port(0);
}

}  // namespace maximus