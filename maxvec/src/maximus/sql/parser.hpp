#pragma once
#include <SQLParser.h>

#include <iostream>
#include <maximus/context.hpp>
#include <maximus/dag/query_node.hpp>
#include <maximus/dag/query_plan.hpp>
#include <maximus/database.hpp>
#include <maximus/database_catalogue.hpp>
#include <maximus/operators/properties.hpp>
#include <memory>
#include <string>

namespace maximus {

class ParserContext {
public:
    explicit ParserContext(std::unordered_map<std::string, std::shared_ptr<QueryNode>> tables_map,
                           std::shared_ptr<ParserContext> upper_context = nullptr,
                           const hsql::SelectStatement* select          = nullptr)
            : tables_map(std::move(tables_map))
            , upper_context(std::move(upper_context))
            , select(std::move(select)) {}

    std::unordered_map<std::string, std::shared_ptr<QueryNode>> tables_map;
    std::shared_ptr<ParserContext> upper_context;
    const hsql::SelectStatement* select;
};

class Parser {
public:
    explicit Parser(std::unordered_map<std::string, std::shared_ptr<Schema>> schemas,
                    std::shared_ptr<DatabaseCatalogue> db_catalogue,
                    std::shared_ptr<MaximusContext> ctx)
            : schemas_(std::move(schemas))
            , db_catalogue_(std::move(db_catalogue))
            , ctx_(std::move(ctx)) {}

    std::unordered_map<std::string, std::shared_ptr<Schema>> schemas_;
    std::shared_ptr<DatabaseCatalogue> db_catalogue_;
    std::shared_ptr<MaximusContext> ctx_;

    Status query_plan_from_sql(const std::string& sql_query,
                               std::shared_ptr<QueryPlan>& query_plan);

    Status qp_from_select(const hsql::SelectStatement* select,
                          std::shared_ptr<QueryNode>& query_plan,
                          std::shared_ptr<ParserContext> parser_context);

    void handle_inner_selects(
        std::vector<std::tuple<hsql::SelectStatement*, hsql::Expr*, hsql::OperatorType>>
            inner_selects,
        std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map,
        std::shared_ptr<ParserContext> parser_context);

    void handle_inner_selects(
        std::vector<std::tuple<hsql::SelectStatement*, hsql::Expr*, hsql::OperatorType>>
            inner_selects,
        std::shared_ptr<ParserContext> parser_context,
        const hsql::SelectStatement* select,
        std::vector<std::shared_ptr<Aggregate>> aggregates,
        std::shared_ptr<QueryNode>& result_plan,
        std::unordered_map<std::string, std::string> target_2_name_map);

    Status qp_from_table_ref(
        const hsql::TableRef* table,
        std::shared_ptr<QueryNode>& query_plan,
        std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map);

    Status qp_where(const hsql::Expr* where,
                    std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map,
                    std::shared_ptr<ParserContext> parser_context);

    std::string get_column_fullname(const hsql::Expr* expr);

    void rename_columns(const hsql::SelectStatement* select,
                        std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map,
                        const std::shared_ptr<ParserContext> upper_layer_context);

    Status extract_join_keys(std::vector<std::vector<arrow::FieldRef>>& column_names,
                             std::unordered_map<std::string, int>& table_name_to_idx,
                             hsql::Expr* condition,
                             std::shared_ptr<Expression>& filter);

    Status join_config_from_expr(std::string left_table_name,
                                 std::string right_table_name,
                                 hsql::JoinDefinition* join,
                                 std::shared_ptr<JoinProperties>& join_config);

    std::string get_join_key_column_from_where_clause(const hsql::SelectStatement* select);

    std::string get_join_key_from_select_list(const hsql::SelectStatement* select);

    std::shared_ptr<QueryNode> handle_inner_select(hsql::OperatorType op,
                                                   std::shared_ptr<QueryNode> qp_select,
                                                   std::shared_ptr<QueryNode> qp_other_table,
                                                   hsql::SelectStatement* select,
                                                   std::shared_ptr<ParserContext> parser_context,
                                                   hsql::Expr* join_column_expr);

    std::shared_ptr<QueryNode> handle_inner_select(
        hsql::OperatorType op,
        std::shared_ptr<QueryNode> query_plan_inner_select,
        std::shared_ptr<QueryNode> qp_other_table,
        const hsql::SelectStatement* select,
        hsql::SelectStatement* inner_select,
        std::vector<std::shared_ptr<Aggregate>> aggregates,
        std::shared_ptr<ParserContext> parser_context,
        std::string join_column_expr);

    void handle_join_conditions(
        std::vector<std::pair<hsql::Expr*, hsql::Expr*>> join_conds,
        std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map);

    std::vector<std::string> get_groupby_keys(const hsql::GroupByDescription* group_by);

    Status qp_group_by(const hsql::GroupByDescription* group_by,
                       std::vector<std::shared_ptr<Aggregate>>& aggregates,
                       std::shared_ptr<QueryNode>& query_plan);

    Status qp_from_join(const hsql::TableRef* table,
                        std::unordered_map<std::string, std::shared_ptr<QueryNode>>& query_plan);

    bool extract_final_columns(const hsql::SelectStatement* select,
                               std::vector<std::string>& columns,
                               std::vector<std::shared_ptr<Expression>>& expressions);

    void extract_columns_select_list(
        const hsql::SelectStatement* select,
        std::vector<std::shared_ptr<Expression>>& ref_columns_expression,
        std::vector<std::string>& ref_columns);

    void extract_columns_select_list_and_where(const hsql::SelectStatement* select,
                                               std::vector<std::string>& ref_columns);

    arrow::compute::Expression parse_expression(const hsql::Expr* expr);

    std::string get_aggregate_inner_expression(const hsql::Expr* expr);

    std::string get_aggregate_name_without_alias(const hsql::Expr* expr);

    std::string get_aggregate_name(const hsql::Expr* expr);

    void extract_aggregate(hsql::Expr* expr,
                           const hsql::SelectStatement* select,
                           std::vector<std::string>& column_refs,
                           std::vector<std::shared_ptr<Aggregate>>& aggregates,
                           std::unordered_map<std::string, std::string>& target_2_name_map);

    void extract_pre_aggregate(
        const hsql::Expr* expr,
        std::vector<std::string>& agg_targets,
        std::vector<std::shared_ptr<maximus::Expression>>& pre_agg_expressions);

    bool extract_post_aggregate(const hsql::SelectStatement* select,
                                std::vector<std::string>& columns,
                                std::vector<std::shared_ptr<Expression>>& expressions);

    void parse_where(
        const hsql::Expr* where_expr,
        std::vector<std::pair<hsql::Expr*, hsql::Expr*>>& join_conds,
        std::vector<std::pair<arrow::compute::Expression, std::string>>& filters,
        std::vector<std::tuple<hsql::SelectStatement*, hsql::Expr*, hsql::OperatorType>>&
            inner_select,
        std::vector<arrow::compute::Expression>& post_join_filters);

    void parse_having(
        const hsql::Expr* having,
        std::vector<std::tuple<hsql::SelectStatement*, hsql::Expr*, hsql::OperatorType>>&
            inner_selects,
        std::vector<arrow::compute::Expression>& filters);

    /********************/
    /*  static methods  */
    /********************/

    static Status qp_order_by(std::vector<hsql::OrderDescription*>* order_by,
                              std::shared_ptr<MaximusContext>& ctx,
                              std::shared_ptr<QueryNode>& query_plan);

    static void extract_reference_columns(hsql::Expr* expr, std::vector<hsql::Expr*>& columns);

    static void set_missing_tables_from_upper_scope_context(
        const std::shared_ptr<ParserContext> upper_layer_context,
        std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map,
        hsql::Expr* where_expr);

    static bool table_exist_in_select(const hsql::TableRef* table, const std::string table_name);

    static void extract_join_conditions(
        const hsql::Expr* expr, std::vector<std::pair<hsql::Expr*, hsql::Expr*>>& join_conds);

    static std::unordered_set<std::string> get_tables(const hsql::SelectStatement* select);

    static void handle_filters(
        std::vector<std::pair<arrow::compute::Expression, std::string>> filters,
        std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map);

    static void handle_filters(std::vector<arrow::compute::Expression> filters,
                               std::shared_ptr<QueryNode>& result_plan);

    static void handle_post_join_filters(
        std::vector<arrow::compute::Expression> post_join_filters,
        std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map,
        std::shared_ptr<QueryNode> post_join_plan);

    static Status schema_from_sql(
        std::string sql_string, std::unordered_map<std::string, std::shared_ptr<Schema>>& schemas);
};
}  // namespace maximus
