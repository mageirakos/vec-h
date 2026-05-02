#include <maximus/operators/abstract_project_operator.hpp>

namespace maximus {

AbstractProjectOperator::AbstractProjectOperator(std::shared_ptr<MaximusContext> &ctx,
                                                 std::shared_ptr<Schema> input_schema,
                                                 std::shared_ptr<ProjectProperties> properties)
        : AbstractOperator(PhysicalOperatorType::PROJECT, ctx, std::move(input_schema))
        , properties(std::move(properties)) {
}
}  // namespace maximus