#include <maximus/operators/acero/properties.hpp>
#include <maximus/operators/native/table_source_operator.hpp>
#include <maximus/utils/utils.hpp>

namespace maximus {

arrow::acero::JoinType to_acero_join_type(JoinType join_type) {
    switch (join_type) {
        case JoinType::LEFT_SEMI:
            return arrow::acero::JoinType::LEFT_SEMI;
        case JoinType::RIGHT_SEMI:
            return arrow::acero::JoinType::RIGHT_SEMI;
        case JoinType::LEFT_ANTI:
            return arrow::acero::JoinType::LEFT_ANTI;
        case JoinType::RIGHT_ANTI:
            return arrow::acero::JoinType::RIGHT_ANTI;
        case JoinType::INNER:
            return arrow::acero::JoinType::INNER;
        case JoinType::LEFT_OUTER:
            return arrow::acero::JoinType::LEFT_OUTER;
        case JoinType::RIGHT_OUTER:
            return arrow::acero::JoinType::RIGHT_OUTER;
        case JoinType::FULL_OUTER:
            return arrow::acero::JoinType::FULL_OUTER;
        case JoinType::CROSS_JOIN:
            throw std::runtime_error("CROSS_JOIN is not supported in Acero");
        default:
            throw std::runtime_error("Unsupported JoinType");
    }
}
JoinType from_acero_join_type(arrow::acero::JoinType join_type) {
    switch (join_type) {
        case arrow::acero::JoinType::LEFT_SEMI:
            return JoinType::LEFT_SEMI;
        case arrow::acero::JoinType::RIGHT_SEMI:
            return JoinType::RIGHT_SEMI;
        case arrow::acero::JoinType::LEFT_ANTI:
            return JoinType::LEFT_ANTI;
        case arrow::acero::JoinType::RIGHT_ANTI:
            return JoinType::RIGHT_ANTI;
        case arrow::acero::JoinType::INNER:
            return JoinType::INNER;
        case arrow::acero::JoinType::LEFT_OUTER:
            return JoinType::LEFT_OUTER;
        case arrow::acero::JoinType::RIGHT_OUTER:
            return JoinType::RIGHT_OUTER;
        case arrow::acero::JoinType::FULL_OUTER:
            return JoinType::FULL_OUTER;
        default:
            throw std::runtime_error("Unsupported JoinType");
    }
}
arrow::compute::NullPlacement to_acero_null_ordering(const NullOrder &null_order) {
    switch (null_order) {
        case NullOrder::FIRST:
            return arrow::compute::NullPlacement::AtStart;
        case NullOrder::LAST:
            return arrow::compute::NullPlacement::AtEnd;
        default:
            throw std::runtime_error("Unsupported NullOrder");
    }
}
NullOrder from_acero_null_ordering(arrow::compute::NullPlacement null_order) {
    switch (null_order) {
        case arrow::compute::NullPlacement::AtStart:
            return NullOrder::FIRST;
        case arrow::compute::NullPlacement::AtEnd:
            return NullOrder::LAST;
        default:
            throw std::runtime_error("Unsupported NullOrder");
    }
}
arrow::compute::SortOrder to_acero_sort_order(const SortOrder &sort_order) {
    switch (sort_order) {
        case SortOrder::ASCENDING:
            return arrow::compute::SortOrder::Ascending;
        case SortOrder::DESCENDING:
            return arrow::compute::SortOrder::Descending;
        default:
            throw std::runtime_error("Unsupported SortOrder");
    }
}
SortOrder from_acero_sort_order(arrow::compute::SortOrder sort_order) {
    switch (sort_order) {
        case arrow::compute::SortOrder::Ascending:
            return SortOrder::ASCENDING;
        case arrow::compute::SortOrder::Descending:
            return SortOrder::DESCENDING;
        default:
            throw std::runtime_error("Unsupported SortOrder");
    }
}
arrow::compute::SortKey to_acero_sort_key(const SortKey &sort_key) {
    return arrow::compute::SortKey(sort_key.field, to_acero_sort_order(sort_key.order));
}
SortKey from_acero_sort_key(const arrow::compute::SortKey &sort_key) {
    return SortKey(sort_key.target, from_acero_sort_order(sort_key.order));
}
std::shared_ptr<arrow::acero::ExecNodeOptions> to_acero_options(
    std::shared_ptr<MaximusContext> &ctx, const std::shared_ptr<NodeProperties> &properties) {
    // SOURCE OPERATOR
    if (auto source_properties = std::dynamic_pointer_cast<TableSourceProperties>(properties)) {
        if (source_properties->table) {
            auto device_table = source_properties->table;
            device_table.convert_to<ArrowTablePtr>(ctx, source_properties->schema);
            assert(device_table.is_arrow_table());
            return std::make_shared<arrow::acero::TableSourceNodeOptions>(
                device_table.as_arrow_table());
        } else {
            // read the table from the path
            auto device_table = read_table(ctx,
                                           source_properties->path,
                                           source_properties->schema,
                                           source_properties->include_columns,
                                           DeviceType::CPU);
            assert(device_table.is_table());
            auto table = device_table.as_table();
            assert(table);
            auto table_reader = std::make_shared<arrow::TableBatchReader>(table->get_table());
            return std::make_shared<arrow::acero::RecordBatchReaderSourceNodeOptions>(table_reader);
        }
    }

    // FILTER OPERATOR
    if (auto filter_properties = std::dynamic_pointer_cast<FilterProperties>(properties)) {
        auto arrow_expr = filter_properties->filter_expression->get_expression();
        assert(arrow_expr);
        std::vector<arrow::compute::Expression> exprs;
        exprs.push_back(*arrow_expr);
        return std::make_shared<arrow::acero::FilterNodeOptions>(std::move(exprs[0]));
    }

    // DISTINCT OPERATOR
    if (auto distinct_properties = std::dynamic_pointer_cast<DistinctProperties>(properties)) {
        auto &keys = distinct_properties->distinct_keys;
        return std::make_shared<arrow::acero::AggregateNodeOptions>(
            std::vector<arrow::compute::Aggregate>(), keys);
    }

    // PROJECT OPERATOR
    if (auto project_properties = std::dynamic_pointer_cast<ProjectProperties>(properties)) {
        std::vector<arrow::compute::Expression> arrow_exprs;
        arrow_exprs.reserve(project_properties->project_expressions.size());
        for (const auto &expr : project_properties->project_expressions) {
            auto arrow_expr = expr->get_expression();
            arrow_exprs.push_back(*arrow_expr);
        }
        auto column_names = project_properties->column_names;
        return std::make_shared<arrow::acero::ProjectNodeOptions>(std::move(arrow_exprs),
                                                                  std::move(column_names));
    }

    // HASH-JOIN OPERATOR
    if (auto join_properties = std::dynamic_pointer_cast<JoinProperties>(properties)) {
        return std::make_shared<arrow::acero::HashJoinNodeOptions>(
            to_acero_join_type(join_properties->join_type),
            join_properties->left_keys,
            join_properties->right_keys,
            *join_properties->filter->get_expression(),
            join_properties->left_suffix,
            join_properties->right_suffix);
    }

    // GROUP BY OPERATOR
    if (auto group_properties = std::dynamic_pointer_cast<GroupByProperties>(properties)) {
        std::vector<arrow::compute::Aggregate> arrow_aggs;
        arrow_aggs.reserve(group_properties->aggregates.size());
        for (const auto &agg : group_properties->aggregates) {
            arrow_aggs.emplace_back(*agg->get_aggregate());
        }

        return std::make_shared<arrow::acero::AggregateNodeOptions>(std::move(arrow_aggs),
                                                                    group_properties->group_keys);
    }

    // ORDER BY OPERATOR
    if (auto order_by_properties = std::dynamic_pointer_cast<OrderByProperties>(properties)) {
        std::vector<arrow::compute::SortKey> arrow_sort_keys;
        arrow_sort_keys.reserve(order_by_properties->sort_keys.size());
        for (const auto &sort_key : order_by_properties->sort_keys) {
            arrow_sort_keys.emplace_back(to_acero_sort_key(sort_key));
        }

        arrow::compute::Ordering ordering = arrow::compute::Ordering(
            std::move(arrow_sort_keys), to_acero_null_ordering(order_by_properties->null_order));

        return std::make_shared<arrow::acero::OrderByNodeOptions>(std::move(ordering));
    }

    // LIMIT OPERATOR
    if (auto limit_properties = std::dynamic_pointer_cast<LimitProperties>(properties)) {
        return std::make_shared<arrow::acero::FetchNodeOptions>(0, limit_properties->limit);
    }

    // Handle other cases or throw an exception for unknown types.
    throw std::runtime_error("This Maximus Operator is not supported in Acero.");
}

std::shared_ptr<NodeProperties> from_acero_options(
    std::shared_ptr<MaximusContext> &ctx,
    const std::shared_ptr<arrow::acero::ExecNodeOptions> &options) {
    // SOURCE OPERATOR
    if (auto table_source_options =
            std::dynamic_pointer_cast<arrow::acero::TableSourceNodeOptions>(options)) {
        auto table = std::make_shared<Table>(ctx, table_source_options->table);
        // auto schema = table->get_schema();
        // auto device_table = DeviceTablePtr(std::move(table));
        return std::make_shared<TableSourceProperties>(DeviceTablePtr(std::move(table)));
    }

    // FILTER OPERATOR
    if (auto filter_options = std::dynamic_pointer_cast<arrow::acero::FilterNodeOptions>(options)) {
        auto expr = std::make_shared<Expression>(
            std::make_shared<arrow::compute::Expression>(filter_options->filter_expression));
        // std::cout << "filter expression = " << filter_options->filter_expression.ToString() << std::endl;
        return std::make_shared<FilterProperties>(std::move(expr));
    }

    // PROJECT OPERATOR
    if (auto project_options =
            std::dynamic_pointer_cast<arrow::acero::ProjectNodeOptions>(options)) {
        std::vector<std::shared_ptr<Expression>> expressions;
        expressions.reserve(project_options->expressions.size());
        for (const auto &expr : project_options->expressions) {
            expressions.push_back(
                std::make_shared<Expression>(std::make_shared<arrow::compute::Expression>(expr)));
        }
        return std::make_shared<ProjectProperties>(expressions, project_options->names);
    }

    // HASH-JOIN OPERATOR
    if (auto join_options = std::dynamic_pointer_cast<arrow::acero::HashJoinNodeOptions>(options)) {
        auto expr = std::make_shared<Expression>(
            std::make_shared<arrow::compute::Expression>(join_options->filter));
        return std::make_shared<JoinProperties>(from_acero_join_type(join_options->join_type),
                                                join_options->left_keys,
                                                join_options->right_keys,
                                                expr,
                                                join_options->output_suffix_for_left,
                                                join_options->output_suffix_for_right);
    }

    // GROUP BY OPERATOR
    if (auto group_options =
            std::dynamic_pointer_cast<arrow::acero::AggregateNodeOptions>(options)) {
        std::vector<std::shared_ptr<Aggregate>> aggregates;
        aggregates.reserve(group_options->aggregates.size());
        for (const auto &agg : group_options->aggregates) {
            aggregates.push_back(
                std::make_shared<Aggregate>(std::make_shared<arrow::compute::Aggregate>(agg)));
        }

        return std::make_shared<GroupByProperties>(group_options->keys, std::move(aggregates));
    }

    // ORDER BY OPERATOR
    if (auto order_by_options =
            std::dynamic_pointer_cast<arrow::acero::OrderByNodeOptions>(options)) {
        std::vector<SortKey> sort_keys;
        auto acero_sort_keys  = order_by_options->ordering.sort_keys();
        auto acero_null_order = order_by_options->ordering.null_placement();
        sort_keys.reserve(acero_sort_keys.size());
        for (const auto &sort_key : acero_sort_keys) {
            sort_keys.push_back(from_acero_sort_key(sort_key));
        }

        return std::make_shared<OrderByProperties>(std::move(sort_keys),
                                                   from_acero_null_ordering(acero_null_order));
    }

    // LIMIT OPERATOR
    if (auto limit_options = std::dynamic_pointer_cast<arrow::acero::FetchNodeOptions>(options)) {
        return std::make_shared<LimitProperties>(limit_options->count, limit_options->offset);
    }

    // Handle other cases or throw an exception for unknown types.
    throw std::runtime_error("This Acero Operator is not supported in Maximus.");
}
}  // namespace maximus
