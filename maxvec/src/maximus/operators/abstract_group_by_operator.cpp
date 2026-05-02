#include <maximus/operators/abstract_group_by_operator.hpp>

namespace maximus {

AbstractGroupByOperator::AbstractGroupByOperator(std::shared_ptr<MaximusContext> &ctx,
                                                 std::shared_ptr<Schema> input_schema,
                                                 std::shared_ptr<GroupByProperties> properties)
        : AbstractOperator(PhysicalOperatorType::GROUP_BY, ctx, std::move(input_schema))
        , properties(std::move(properties)) {
    set_blocking_port(0);
}

}  // namespace maximus