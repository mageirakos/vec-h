#include <maximus/operators/abstract_filter_operator.hpp>

namespace maximus {

maximus::AbstractFilterOperator::AbstractFilterOperator(
    std::shared_ptr<MaximusContext> &ctx,
    std::shared_ptr<Schema> input_schema,
    std::shared_ptr<FilterProperties> properties)
        : AbstractOperator(PhysicalOperatorType::FILTER, ctx, std::move(input_schema))
        , properties(std::move(properties)) {
    assign_output_schema(this->input_schemas[0]);
}

}  // namespace maximus