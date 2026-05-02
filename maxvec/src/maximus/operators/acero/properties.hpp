#pragma once

#include <arrow/acero/options.h>

#include <maximus/dag/query_node.hpp>
#include <maximus/dag/query_plan.hpp>
#include <maximus/operators/properties.hpp>
#include <maximus/types/expression.hpp>

namespace maximus {

arrow::acero::JoinType to_acero_join_type(JoinType join_type);

JoinType from_acero_join_type(arrow::acero::JoinType join_type);

arrow::compute::NullPlacement to_acero_null_ordering(const NullOrder& null_order);

NullOrder from_acero_null_ordering(arrow::compute::NullPlacement null_order);

arrow::compute::SortOrder to_acero_sort_order(const SortOrder& sort_order);

SortOrder from_acero_sort_order(arrow::compute::SortOrder sort_order);

arrow::compute::SortKey to_acero_sort_key(const SortKey& sort_key);

SortKey from_acero_sort_key(const arrow::compute::SortKey& sort_key);

std::shared_ptr<arrow::acero::ExecNodeOptions> to_acero_options(
    std::shared_ptr<MaximusContext>& ctx, const std::shared_ptr<NodeProperties>& properties);

std::shared_ptr<NodeProperties> from_acero_options(
    std::shared_ptr<MaximusContext>& ctx,
    const std::shared_ptr<arrow::acero::ExecNodeOptions>& options);
}  // namespace maximus
