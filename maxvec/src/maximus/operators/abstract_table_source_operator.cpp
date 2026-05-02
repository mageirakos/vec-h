#include <maximus/operators/abstract_table_source_operator.hpp>

namespace maximus {

AbstractTableSourceOperator::AbstractTableSourceOperator(
    std::shared_ptr<MaximusContext> &ctx, std::shared_ptr<TableSourceProperties> properties)
        : AbstractOperator(PhysicalOperatorType::TABLE_SOURCE, ctx)
        , properties(std::move(properties)) {
}
}  // namespace maximus