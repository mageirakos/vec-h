#include <maximus/operators/abstract_table_source_filter_project_operator.hpp>

namespace maximus {

AbstractTableSourceFilterProjectOperator::AbstractTableSourceFilterProjectOperator(
    std::shared_ptr<MaximusContext> &ctx,
    std::shared_ptr<TableSourceFilterProjectProperties> properties)
        : AbstractOperator(PhysicalOperatorType::TABLE_SOURCE_FILTER_PROJECT, ctx)
        , properties(std::move(properties)) {
}

}  // namespace maximus
