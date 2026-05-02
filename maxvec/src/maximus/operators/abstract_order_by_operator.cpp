#include <maximus/operators/abstract_order_by_operator.hpp>

namespace maximus {

AbstractOrderByOperator::AbstractOrderByOperator(std::shared_ptr<MaximusContext> &ctx,
                                                 std::shared_ptr<Schema> input_schema,
                                                 std::shared_ptr<OrderByProperties> properties)
        : AbstractOperator(PhysicalOperatorType::ORDER_BY, ctx, input_schema, input_schema)
        , properties(std::move(properties)) {
    set_blocking_port(0);
}

}  // namespace maximus