#include <util/sqlhelper.h>

#include <maximus/frontend/expressions.hpp>
#include <maximus/frontend/query_plan_api.hpp>
#include <maximus/sql/parser.hpp>

namespace maximus {

std::string extract_operator(hsql::OperatorType operator_type) {
    switch (operator_type) {
        case hsql::OperatorType::kOpPlus:
            return "+";
        case hsql::OperatorType::kOpMinus:
            return "-";
        case hsql::OperatorType::kOpAsterisk:
            return "*";
        case hsql::OperatorType::kOpSlash:
            return "/";
        case hsql::OperatorType::kOpGreater:
            return ">";
        case hsql::OperatorType::kOpLess:
            return "<";
        case hsql::OperatorType::kOpLessEq:
            return "<=";
        case hsql::OperatorType::kOpGreaterEq:
            return ">=";
        default:
            std::cout << "operator type: " << operator_type << "\n";
            throw std::runtime_error("[extract operator]: operator not supported yet");
    }
}

std::string extract_hash_function(const hsql::Expr* expr) {
    std::string function = expr->name;
    std::string result   = function;
    if (function == "avg")
        result = "mean";
    else if (function == "quantile_cont")
        result = "approximate_median";

    if (expr->distinct) result = result + "_distinct";

    return result;
}

std::string extract_hash_function(const hsql::Expr* expr, const hsql::SelectStatement* select) {
    std::string result = extract_hash_function(expr);
    result             = bool(select->groupBy) ? "hash_" + result : result;
    return result;
}

bool is_aggregate_function(std::string function) {
    if (function == "substring") return false;

    return true;
}

std::string Parser::get_aggregate_inner_expression(const hsql::Expr* expr) {
    assert(expr->type == hsql::kExprFunctionRef);
    return parse_expression((*expr->exprList).front()).ToString();
}

std::string Parser::get_aggregate_name_without_alias(const hsql::Expr* expr) {
    assert(expr->type == hsql::kExprFunctionRef);
    std::string function = extract_hash_function(expr);
    if (expr->exprList->size()) {
        auto x = (*expr->exprList).front();
        if (x->type == hsql::kExprStar) {
            return "*_" + function;
        }
        return parse_expression(x).ToString() + "_" + function;
    }
    return "_" + function;
}

std::string Parser::get_aggregate_name(const hsql::Expr* expr) {
    assert(expr->type == hsql::kExprFunctionRef);
    if (expr->hasAlias())
        return expr->alias;
    else
        return get_aggregate_name_without_alias(expr);
}

Status Parser::extract_join_keys(std::vector<std::vector<arrow::FieldRef>>& column_names,
                                 std::unordered_map<std::string, int>& table_name_to_idx,
                                 hsql::Expr* condition,
                                 std::shared_ptr<Expression>& filter) {
    switch (condition->opType) {
        case hsql::OperatorType::kOpAnd: {
            hsql::Expr* left  = condition->expr;
            hsql::Expr* right = condition->expr2;
            assert(left);
            assert(right);
            CHECK_STATUS(
                extract_join_keys(column_names, table_name_to_idx, condition->expr, filter));
            CHECK_STATUS(
                extract_join_keys(column_names, table_name_to_idx, condition->expr2, filter));
            break;
        }
        case hsql::OperatorType::kOpEquals: {
            assert(condition->expr && condition->expr2);
            hsql::Expr* left  = condition->expr;
            hsql::Expr* right = condition->expr2;
            assert(left);
            assert(right);
            assert(left->type == hsql::kExprColumnRef);
            assert(right->type == hsql::kExprColumnRef);

            std::string table1_name  = left->table;
            std::string column1_name = get_column_fullname(left);
            std::string table2_name  = right->table;
            std::string column2_name = get_column_fullname(right);
            column_names[table_name_to_idx[table1_name]].push_back(column1_name);
            column_names[table_name_to_idx[table2_name]].push_back(column2_name);
            break;
        }
        case hsql::OperatorType::kOpNotLike: {
            filter = maximus::expr(Parser::parse_expression(condition));
            break;
        }
        default: {
            return Status(ErrorCode::MaximusError,
                          "[extract_join_keys]: Unsupported join condition.");
        }
    }
    return Status::OK();
}

std::vector<std::string> handle_duplicates(std::vector<std::string> column_names) {
    /*
    The handle_duplicates function processes a vector of column names and ensures that all names in the resulting vector are unique.
    If duplicate names are found, the function appends a numerical suffix to each duplicate to make it unique.
    */
    std::unordered_map<std::string, int> name_count;
    std::vector<std::string> result;

    for (auto& name : column_names) {
        if (name_count.find(name) == name_count.end()) {
            // If the name is not in the map, it's unique so far
            name_count[name] = 0;
            result.push_back(name);
        } else {
            // If the name already exists, increment its count and make a unique name
            name_count[name]++;
            std::string new_name = name + "_" + std::to_string(name_count[name]);
            result.push_back(new_name);
            // Add the new name to the map as well to track further duplicates
            name_count[new_name] = 0;
        }
    }

    return result;
}

std::vector<std::string> remove_duplicates(std::vector<std::string> column_names) {
    /*
    The remove_duplicates function processes a vector of column names and ensures that all names in the resulting vector are unique.
    If duplicate names are found, the function removes from the final output.
    */
    std::unordered_set<std::string> name_count;
    std::vector<std::string> result;
    for (auto& name : column_names) {
        if (name_count.find(name) == name_count.end()) {
            // If the name is not in the map, it's unique so far
            name_count.insert(name);
            result.push_back(name);
        }
    }

    return result;
}

Status Parser::join_config_from_expr(std::string left_table_name,
                                     std::string right_table_name,
                                     hsql::JoinDefinition* join,
                                     std::shared_ptr<JoinProperties>& join_config) {
    assert(join);
    std::unordered_map<hsql::JoinType, maximus::JoinType> to_join_type = {
        {hsql::JoinType::kJoinInner, JoinType::INNER},
        {hsql::JoinType::kJoinLeft, JoinType::LEFT_OUTER},
        {hsql::JoinType::kJoinRight, JoinType::RIGHT_OUTER},
        {hsql::JoinType::kJoinNatural, JoinType::INNER},
        {hsql::JoinType::kJoinFull, JoinType::FULL_OUTER},
        {hsql::JoinType::kJoinCross, JoinType::CROSS_JOIN}};

    // assert(join->type);
    if (to_join_type.find(join->type) == to_join_type.end()) {
        return Status(ErrorCode::MaximusError, "Unsupported join type.");
    }

    JoinType join_type = to_join_type[join->type];
    assert(join->condition);

    hsql::Expr* condition = join->condition;
    assert(condition->opType);

    std::vector<std::vector<arrow::FieldRef>> column_names(2, std::vector<arrow::FieldRef>());
    std::unordered_map<std::string, int> table_name_to_idx = {{left_table_name, 0},
                                                              {right_table_name, 1}};

    auto filter = std::make_shared<Expression>(
        std::make_shared<arrow::compute::Expression>(arrow::compute::literal(true)));
    CHECK_STATUS(extract_join_keys(column_names, table_name_to_idx, condition, filter));
    join_config =
        std::make_shared<JoinProperties>(join_type, column_names[0], column_names[1], filter);

    std::cout << "join properties = " << join_config->to_string() << std::endl;
    return Status::OK();
}

Status Parser::qp_order_by(std::vector<hsql::OrderDescription*>* order_by,
                           std::shared_ptr<MaximusContext>& ctx,
                           std::shared_ptr<QueryNode>& query_plan) {
    assert(ctx);
    assert(order_by);
    std::vector<SortKey> sort_keys;
    for (auto& o : *order_by) {
        auto is_ascending =
            o->type == hsql::kOrderAsc ? SortOrder::ASCENDING : SortOrder::DESCENDING;
        auto sk = SortKey(arrow::FieldRef(o->expr->name), is_ascending);
        sort_keys.push_back(sk);
    }
    auto properties    = std::make_shared<OrderByProperties>(sort_keys);
    auto order_by_plan = std::make_shared<QueryNode>(
        EngineType::ACERO, DeviceType::CPU, NodeType::ORDER_BY, std::move(properties), ctx);
    assert(order_by_plan);
    order_by_plan->add_input(query_plan);
    query_plan = order_by_plan;
    return Status::OK();
}

std::string Parser::get_join_key_column_from_where_clause(const hsql::SelectStatement* select) {
    // it is assumed that the select statement only joins itself with ONE other column from the
    // upper layer/scope
    std::queue<hsql::Expr*> expr_stack;
    expr_stack.push(select->whereClause);
    std::unordered_set<std::string> table_names = get_tables(select);
    while (!expr_stack.empty()) {
        auto head = expr_stack.front();
        expr_stack.pop();
        switch (head->type) {
            case hsql::kExprOperator:
                switch (head->opType) {
                    case hsql::kOpAnd:
                        expr_stack.push(head->expr);
                        expr_stack.push(head->expr2);
                        break;
                    case hsql::kOpEquals:
                        if (head->expr->type == hsql::kExprColumnRef &&
                            head->expr->type == hsql::kExprColumnRef &&
                            head->expr->table != head->expr2->table) {
                            if (table_names.find(head->expr->table) != table_names.end()) {
                                return get_column_fullname(head->expr);
                            } else if (table_names.find(head->expr2->table) != table_names.end()) {
                                return get_column_fullname(head->expr2);
                            }
                        }
                        break;
                    default:
                        break;
                }
                break;
            default:
                break;
        }
    }
    throw std::runtime_error("[get_join_key_column_from_where_clause]: Unsupported case for "
                             "extracting the column name of inner select statement for join");
}

std::string Parser::get_join_key_from_select_list(const hsql::SelectStatement* select) {
    // note that when you want to extract column name in this function, the select
    // clause is already parsed. Thus, the names of the final columns have been extracted
    // and renamed.
    assert(select->selectList->size() == 1);
    auto expr = select->selectList->front();
    if (expr->alias) return expr->alias;
    switch (expr->type) {
        case hsql::kExprStar:
            // this is for handling EXIST/NOT EXIST cases
            return get_join_key_column_from_where_clause(select);
        case hsql::kExprColumnRef:
            return expr->name;
        case hsql::kExprFunctionRef:
            return get_aggregate_name(expr);
        case hsql::kExprOperator:
            return parse_expression(expr).ToString();
    }
    std::cout << expr->type << " :type\n";
    throw std::runtime_error("[get_join_key_from_select_list]: Unsupported case for extracting the "
                             "column name of inner select statement for join");
}

void update_tables_map(std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map,
                       const std::shared_ptr<QueryNode> old_plan,
                       const std::shared_ptr<QueryNode> new_plan) {
    for (auto& k : tables_map) {
        if (k.second == old_plan) {
            k.second = new_plan;
        }
    }
}

Status check_tables_map_are_same(
    const std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map) {
    auto first_entry = tables_map.begin();
    for (auto& [table_name, node] : tables_map) {
        if (node != first_entry->second)
            return Status(ErrorCode::MaximusError,
                          "tables are not joined completely after processing 'while' section");
    }
    return Status::OK();
}

std::shared_ptr<QueryNode> get_query_plan_after_joins(
    std::unordered_map<std::string, std::shared_ptr<QueryNode>> tables_map,
    const hsql::Expr* where_expr) {
    assert(where_expr);
    if (where_expr->type == hsql::kExprOperator) {
        return get_query_plan_after_joins(tables_map, where_expr->expr);
    }
    if (where_expr->type == hsql::kExprFunctionRef) {
        return tables_map[(*where_expr->exprList).front()->table];
    }
    if (where_expr->type == hsql::kExprColumnRef) {
        return tables_map[where_expr->table];
    }
    std::cout << where_expr->expr->type << "\n";
    throw std::runtime_error("at least one of the where expression should be of "
                             "hsql::KExpColumnRef or hsql::KExprOperator");
}

std::shared_ptr<QueryNode> Parser::handle_inner_select(
    hsql::OperatorType op,
    std::shared_ptr<QueryNode> query_plan_inner_select,
    std::shared_ptr<QueryNode> qp_other_table,
    const hsql::SelectStatement* select,
    hsql::SelectStatement* inner_select,
    std::vector<std::shared_ptr<Aggregate>> aggregates,
    std::shared_ptr<ParserContext> parser_context,
    std::string join_column) {
    switch (op) {
        case hsql::kOpLess:
        case hsql::kOpLessEq:
        case hsql::kOpGreater:
        case hsql::kOpGreaterEq: {
            std::vector<std::string> groupby_columns = get_groupby_keys(select->groupBy);
            std::vector<std::string> aggregates_name;
            for (auto& agg : aggregates) aggregates_name.push_back(agg->get_aggregate()->name);
            groupby_columns.insert(
                groupby_columns.end(), aggregates_name.begin(), aggregates_name.end());
            // todo: set device type implicitly
            auto cross_join_node = cross_join(qp_other_table,
                                              query_plan_inner_select,
                                              groupby_columns,
                                              {get_join_key_from_select_list(inner_select)},
                                              "",
                                              "");
            auto f               = expr(arrow_expr(cp::field_ref(join_column),
                                     extract_operator(op),
                                     cp::field_ref(get_join_key_from_select_list(inner_select))));
            // todo: set device type implicitly
            return filter(cross_join_node, f);
        }
        default:
            throw std::runtime_error("[after parse having]: Unsupported operator type");
    }
}

std::shared_ptr<QueryNode> Parser::handle_inner_select(
    hsql::OperatorType op,
    std::shared_ptr<QueryNode> qp_select,
    std::shared_ptr<QueryNode> qp_other_table,
    hsql::SelectStatement* select,
    std::shared_ptr<ParserContext> parser_context,
    hsql::Expr* join_column_expr) {
    // todo: set device type implicitly
    switch (op) {
        case hsql::kOpEquals:
            return inner_join(qp_select,
                              qp_other_table,
                              {get_join_key_from_select_list(select)},
                              {get_column_fullname(join_column_expr)});

        case hsql::kOpNotEquals:
            return left_anti_join(qp_other_table,
                                  qp_select,
                                  {get_column_fullname(join_column_expr)},
                                  {get_join_key_from_select_list(select)});
        case hsql::kOpGreater:
        case hsql::kOpGreaterEq:
        case hsql::kOpLess:
        case hsql::kOpLessEq: {
            std::vector<std::string> ref_columns;
            extract_columns_select_list_and_where(parser_context->select, ref_columns);
            std::vector<std::string> combined = {get_column_fullname(join_column_expr)};
            combined.insert(combined.end(), ref_columns.begin(), ref_columns.end());
            combined = remove_duplicates(combined);

            auto cross_join_node = cross_join(qp_other_table,
                                              qp_select,
                                              combined,
                                              {get_join_key_from_select_list(select)},
                                              "",
                                              "");
            auto f         = expr(arrow_expr(cp::field_ref(get_column_fullname(join_column_expr)),
                                     extract_operator(op),
                                     cp::field_ref(get_join_key_from_select_list(select))));
            auto result_qp = filter(cross_join_node, f);
            return result_qp;
        }
        case hsql::kOpExists:
            return left_semi_join(qp_other_table,
                                  qp_select,
                                  {get_column_fullname(join_column_expr)},
                                  {get_join_key_column_from_where_clause(select)},
                                  "",
                                  "");
        default:
            std::cout << "operator type: " << op << "\n";
            throw std::runtime_error("[handle_inner_select]: Unsupported operator");
    }
}

void Parser::handle_join_conditions(
    std::vector<std::pair<hsql::Expr*, hsql::Expr*>> join_conds,
    std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map) {
    std::unordered_map<std::string, std::string> parents_map;
    // set the parent of each table to itself (initialization)
    for (auto& [table_name, v] : tables_map) {
        parents_map[table_name] = table_name;
    }
    std::function<std::string(const std::string&)> find_parent =
        [&](const std::string& node) -> std::string {
        if (parents_map[node] == node) {
            return node;  // The node is its own parents_map (root)
        }
        return parents_map[node] = find_parent(parents_map[node]);
    };
    for (auto& [left_expr, right_expr] : join_conds) {
        auto left_table   = left_expr->table;
        auto right_table  = right_expr->table;
        auto left_column  = get_column_fullname(left_expr);
        auto right_column = get_column_fullname(right_expr);
        std::shared_ptr<maximus::QueryNode> new_plan;
        if (find_parent(left_table) != find_parent(right_table)) {  // tables have not joined before
            auto old_plan_left  = tables_map[left_table];
            auto old_plan_right = tables_map[right_table];
            // todo: set device type implicitly
            new_plan = inner_join(old_plan_left,
                                  old_plan_right,
                                  {left_column},
                                  {right_column},
                                  "",
                                  "",
                                  DeviceType::CPU);
            update_tables_map(tables_map, old_plan_left, new_plan);
            update_tables_map(tables_map, old_plan_right, new_plan);
            // union two nodes (connect them under a single parent)
            std::string left_parent  = find_parent(left_table);
            std::string right_parent = find_parent(right_table);

            if (left_parent != right_parent) {
                parents_map[right_parent] = left_parent;  // Make one root the parent of the other
            }
        } else {
            auto x = arrow_expr(cp::field_ref(left_column), "==", cp::field_ref(right_column));
            auto old_plan =
                tables_map[left_table];  // can be also right_table. Both must be the same
            // todo: set device type implicitly
            new_plan = filter(old_plan, maximus::expr(x), DeviceType::CPU);
            update_tables_map(tables_map, old_plan, new_plan);
        }
        tables_map[left_table]  = new_plan;
        tables_map[right_table] = new_plan;
    }
}

void Parser::handle_inner_selects(
    std::vector<std::tuple<hsql::SelectStatement*, hsql::Expr*, hsql::OperatorType>> inner_selects,
    std::shared_ptr<ParserContext> parser_context,
    const hsql::SelectStatement* select,
    std::vector<std::shared_ptr<Aggregate>> aggregates,
    std::shared_ptr<QueryNode>& result_plan,
    std::unordered_map<std::string, std::string> target_2_name_map) {
    for (auto& [inner_select, join_column_expr, optype] : inner_selects) {
        std::shared_ptr<QueryNode> query_plan_inner_select;
        qp_from_select(inner_select, query_plan_inner_select, parser_context);
        auto join_column_str =
            target_2_name_map[get_aggregate_name_without_alias(join_column_expr)];
        result_plan = handle_inner_select(optype,
                                          query_plan_inner_select,
                                          result_plan,
                                          select,
                                          inner_select,
                                          aggregates,
                                          parser_context,
                                          join_column_str);
    }
}

void Parser::handle_inner_selects(
    std::vector<std::tuple<hsql::SelectStatement*, hsql::Expr*, hsql::OperatorType>> inner_selects,
    std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map,
    std::shared_ptr<ParserContext> parser_context) {
    for (auto& [select, join_column_expr, op] : inner_selects) {
        std::shared_ptr<QueryNode> query_plan_inner_select;
        qp_from_select(select, query_plan_inner_select, parser_context);
        auto old_plan = tables_map[join_column_expr->table];
        auto new_plan = handle_inner_select(
            op, query_plan_inner_select, old_plan, select, parser_context, join_column_expr);
        update_tables_map(tables_map, old_plan, new_plan);
    }
}

void Parser::handle_filters(std::vector<cp::Expression> filters,
                            std::shared_ptr<QueryNode>& result_plan) {
    for (auto& filter_expr : filters) {
        std::cout << "filter: " << filter_expr.ToString() << "\n";
        // todo: set device type implicitly
        result_plan = filter(result_plan, maximus::expr(filter_expr), DeviceType::CPU);
    }
}

void Parser::handle_filters(
    std::vector<std::pair<cp::Expression, std::string>> filters,
    std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map) {
    for (auto& [expr, table_name] : filters) {
        std::cout << "filter: " << expr.ToString() << "\n";
        auto old_plan = tables_map[table_name];
        // todo: set device type implicitly
        auto new_plan          = filter(old_plan, maximus::expr(expr), DeviceType::CPU);
        tables_map[table_name] = new_plan;
    }
}

void Parser::handle_post_join_filters(
    std::vector<arrow::compute::Expression> post_join_filters,
    std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map,
    std::shared_ptr<QueryNode> post_join_plan) {
    auto old_plan = post_join_plan;
    for (auto& post_join_filter : post_join_filters) {
        // todo: set device type implicitly
        auto new_plan = filter(old_plan, maximus::expr(post_join_filter));
        update_tables_map(tables_map, old_plan, new_plan);
        old_plan = new_plan;
    }
}

Status Parser::qp_where(const hsql::Expr* where,
                        std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map,
                        std::shared_ptr<ParserContext> parser_context) {
    assert(where);
    std::vector<std::pair<hsql::Expr*, hsql::Expr*>> join_conds;
    std::vector<std::pair<cp::Expression, std::string>> filters;  //arrow_expression, table_name
    std::vector<std::tuple<hsql::SelectStatement*, hsql::Expr*, hsql::OperatorType>> inner_selects;
    std::vector<cp::Expression> post_join_filters;  //arrow_expression, table_name
    parse_where(where, join_conds, filters, inner_selects, post_join_filters);

    handle_filters(filters, tables_map);
    handle_join_conditions(join_conds, tables_map);
    handle_inner_selects(inner_selects, tables_map, parser_context);
    auto post_join_plan = get_query_plan_after_joins(tables_map, where);
    handle_post_join_filters(post_join_filters, tables_map, post_join_plan);

    return Status::OK();
}

std::vector<std::string> Parser::get_groupby_keys(const hsql::GroupByDescription* group_by) {
    // assert(group_by);
    std::vector<std::string> group_keys;
    if (group_by) {
        std::vector<hsql::Expr*>* columns = group_by->columns;
        for (auto column : *columns) {
            switch (column->type) {
                case hsql::kExprColumnRef:
                    group_keys.push_back(get_column_fullname(column));
                    break;
                default:
                    throw std::runtime_error(
                        "[get groupby keys]: only field names are allowed for group by keys");
            }
        }
    }
    return group_keys;
}

Status Parser::qp_group_by(const hsql::GroupByDescription* group_by,
                           std::vector<std::shared_ptr<Aggregate>>& aggregates,
                           std::shared_ptr<QueryNode>& query_plan) {
    std::vector<std::string> group_keys = get_groupby_keys(group_by);
    auto properties    = std::make_shared<GroupByProperties>(group_keys, aggregates);
    auto group_by_plan = std::make_shared<QueryNode>(
        EngineType::ACERO, DeviceType::CPU, NodeType::GROUP_BY, std::move(properties), ctx_);
    assert(group_by_plan);
    group_by_plan->add_input(query_plan);
    query_plan = group_by_plan;

    return Status::OK();
}

Status Parser::qp_from_join(
    const hsql::TableRef* table,
    std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map) {
    assert(table->join);
    assert(table->join->left);
    assert(table->join->right);
    assert(table->join->condition);
    std::string left_table_name      = table->join->left->name;
    std::string right_table_name     = table->join->right->name;
    std::shared_ptr<QueryNode> left  = tables_map[left_table_name];
    std::shared_ptr<QueryNode> right = tables_map[right_table_name];

    std::shared_ptr<JoinProperties> join_config;
    CHECK_STATUS(
        join_config_from_expr(left_table_name, right_table_name, table->join, join_config));
    std::shared_ptr<QueryNode> query_plan = std::make_shared<QueryNode>(
        EngineType::ACERO, DeviceType::CPU, NodeType::HASH_JOIN, std::move(join_config), ctx_);
    assert(query_plan);
    // todo: should we remove the std::move?(might be bogus)
    query_plan->add_input(std::move(left));
    query_plan->add_input(std::move(right));
    tables_map[left_table_name]  = query_plan;
    tables_map[right_table_name] = query_plan;
    assert(query_plan->in_degree() == 2);
    return Status::OK();
}

Status Parser::qp_from_table_ref(
    const hsql::TableRef* table,
    std::shared_ptr<QueryNode>& query_plan,
    std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map) {
    assert(table);

    switch (table->type) {
        case hsql::TableRefType::kTableName: {
            assert(table->name);
            std::string table_path = db_catalogue_->table_paths(table->name)[0];
            auto schema            = this->schemas_[table->name];
            auto properties        = std::make_shared<TableSourceProperties>(table_path, schema);
            std::cout << properties->to_string() << std::endl;
            query_plan = std::make_shared<QueryNode>(EngineType::NATIVE,
                                                     DeviceType::UNDEFINED,
                                                     NodeType::TABLE_SOURCE,
                                                     std::move(properties),
                                                     ctx_);
            assert(query_plan);
            tables_map[table->getName()] = query_plan;
            std::cout << query_plan->to_string() << std::endl;
            // std::cout << "table scan = " << query_plan->get_operator()->to_string() << std::endl;
            break;
        }

        case hsql::TableRefType::kTableJoin: {
            assert(table->join);
            assert(table->join->left);
            assert(table->join->right);
            assert(table->join->condition);
            std::shared_ptr<QueryNode> left;
            std::shared_ptr<QueryNode> right;
            CHECK_STATUS(qp_from_table_ref(table->join->left, left, tables_map));
            CHECK_STATUS(qp_from_table_ref(table->join->right, right, tables_map));
            break;
        }

        case hsql::TableRefType::kTableSelect: {
            qp_from_select(table->select, query_plan, nullptr);
            if (table->alias && table->alias->columns && table->alias->columns->size()) {
                std::vector<std::string> final_columns;
                std::vector<std::shared_ptr<Expression>> final_columns_expresssions;
                extract_final_columns(table->select, final_columns, final_columns_expresssions);
                std::vector<std::string> new_columns;
                for (auto& column_name : *table->alias->columns) {
                    new_columns.push_back(column_name);
                }
                // todo: set device type manually
                query_plan = rename(query_plan, final_columns, new_columns);
            }
            tables_map[table->getName()] = query_plan;
            break;
        }

        case hsql::TableRefType::kTableCrossProduct: {
            for (auto& t : *table->list) {
                qp_from_table_ref(t, query_plan, tables_map);
            }
        }
    }

    // assert(query_plan);
    return Status::OK();
}

void Parser::extract_aggregate(hsql::Expr* expr,
                               const hsql::SelectStatement* select,
                               std::vector<std::string>& column_refs,
                               std::vector<std::shared_ptr<Aggregate>>& aggregates,
                               std::unordered_map<std::string, std::string>& target_2_name_map) {
    if (expr == nullptr) return;

    switch (expr->type) {
        case hsql::kExprFunctionRef: {
            std::string function = extract_hash_function(expr, select);
            if (!is_aggregate_function(function)) return;

            std::string alias = get_aggregate_name(expr);
            // checks if the aggregate already exists.
            if (target_2_name_map.find(get_aggregate_name_without_alias(expr)) !=
                    target_2_name_map.end() &&
                target_2_name_map[get_aggregate_name_without_alias(expr)] == alias)
                return;

            target_2_name_map[get_aggregate_name_without_alias(expr)] = alias;

            for (auto* x : *expr->exprList) {
                switch (x->type) {
                    case hsql::kExprOperator:
                    case hsql::kExprColumnRef: {
                        auto new_exp           = Parser::parse_expression(x);
                        arrow::FieldRef target = arrow::FieldRef(new_exp.ToString());
                        auto agg               = Aggregate(function, target, alias);
                        aggregates.push_back(std::make_shared<Aggregate>(agg));
                        return;
                    }
                    default:
                        break;
                }
            }
            // handling AGG(*) or AGG() like COUNT(*) or COUNT()
            auto agg = Aggregate(function, column_refs[0], alias);
            aggregates.push_back(std::make_shared<Aggregate>(agg));
            return;
        }
        case hsql::kExprOperator:
            extract_aggregate(expr->expr, select, column_refs, aggregates, target_2_name_map);
            extract_aggregate(expr->expr2, select, column_refs, aggregates, target_2_name_map);
            break;
    }
}

void Parser::extract_pre_aggregate(
    const hsql::Expr* expr,
    std::vector<std::string>& agg_targets,
    std::vector<std::shared_ptr<maximus::Expression>>& pre_agg_expressions) {
    if (expr == nullptr) return;

    switch (expr->type) {
        case hsql::kExprFunctionRef:
            for (auto* x : *expr->exprList) {
                if (x->type == hsql::kExprColumnRef || x->type == hsql::kExprOperator) {
                    auto new_exp = parse_expression(x);
                    agg_targets.push_back(new_exp.ToString());
                    pre_agg_expressions.push_back(maximus::expr(new_exp));
                }
            }
            break;
        case hsql::kExprOperator:
            extract_pre_aggregate(expr->expr, agg_targets, pre_agg_expressions);
            extract_pre_aggregate(expr->expr2, agg_targets, pre_agg_expressions);
            break;
    }
}

bool Parser::extract_post_aggregate(const hsql::SelectStatement* select,
                                    std::vector<std::string>& columns,
                                    std::vector<std::shared_ptr<Expression>>& expressions) {
    bool has_post_agg_projects = false;
    auto processExpression     = [&](const hsql::Expr* expr) {
        auto final_expr = parse_expression(expr);
        expressions.push_back(maximus::expr(final_expr));
        columns.push_back(expr->hasAlias() ? expr->alias : final_expr.ToString());
        has_post_agg_projects = true;
    };

    for (hsql::Expr* expr : *select->selectList) {
        switch (expr->type) {
            case hsql::kExprExtract:
            case hsql::kExprOperator: {
                processExpression(expr);
                break;
            }
            case hsql::kExprFunctionRef: {
                if (extract_hash_function(expr) == "substring") {
                    processExpression(expr);
                }
                break;
            }
        }
    }
    return has_post_agg_projects;
}

bool Parser::extract_final_columns(const hsql::SelectStatement* select,
                                   std::vector<std::string>& columns,
                                   std::vector<std::shared_ptr<Expression>>& expressions) {
    for (hsql::Expr* expr : *select->selectList) {
        std::string final_name;
        switch (expr->type) {
            case hsql::kExprStar:
                return false;
            case hsql::kExprColumnRef: {
                final_name = expr->hasAlias() ? expr->alias : expr->name;
                auto name  = get_column_fullname(expr);
                expressions.push_back(Expression::from_field_ref(name));
                break;
            }
            case hsql::kExprFunctionRef:
            case hsql::kExprOperator:
            case hsql::kExprExtract: {
                final_name = expr->hasAlias() ? expr->alias : parse_expression(expr).ToString();
                expressions.push_back(Expression::from_field_ref(final_name));
                break;
            }
            default:
                throw std::runtime_error("[extract_final_columns]: Unsupported expression type");
        }
        columns.push_back(final_name);
    }
    return true;
}

void Parser::extract_columns_select_list(
    const hsql::SelectStatement* select,
    std::vector<std::shared_ptr<Expression>>& ref_columns_expression,
    std::vector<std::string>& ref_columns) {
    std::vector<hsql::Expr*> ref_columns_hsql;
    for (hsql::Expr* expr : *select->selectList) {
        switch (expr->type) {
            case hsql::kExprColumnRef:
            case hsql::kExprExtract:
                extract_reference_columns(expr, ref_columns_hsql);
                break;
        }
    }
    for (auto& col : ref_columns_hsql) {
        auto name = get_column_fullname(col);
        ref_columns.push_back(name);
        ref_columns_expression.push_back(Expression::from_field_ref(name));
    }
}

void Parser::extract_columns_select_list_and_where(const hsql::SelectStatement* select,
                                                   std::vector<std::string>& ref_columns) {
    auto tables = get_tables(select);
    std::vector<hsql::Expr*> ref_columns_hsql;
    for (hsql::Expr* expr : *select->selectList) {
        extract_reference_columns(expr, ref_columns_hsql);
    }
    extract_reference_columns(select->whereClause, ref_columns_hsql);
    for (auto& col_expr : ref_columns_hsql) {
        if (tables.find(col_expr->table) !=
            tables.end()) {  // only add tables that belong to this SELECT scope
            auto name = get_column_fullname(col_expr);
            ref_columns.push_back(name);
        }
    }
}

Status Parser::qp_from_select(const hsql::SelectStatement* select,
                              std::shared_ptr<QueryNode>& query_plan,
                              std::shared_ptr<ParserContext> upper_scope_parser_context) {
    // assert(query_plan);
    hsql::printSelectStatementInfo(select, 0);

    // todo: reading the whole table at the beginning! can we do it in a more optimized way?
    std::shared_ptr<QueryNode> result_plan;
    std::unordered_map<std::string, std::shared_ptr<QueryNode>> tables_map;
    if (select->fromTable) {
        // reading all of the tables that exist in FROM part of the SELECT
        std::shared_ptr<QueryNode> source_table_plan;
        qp_from_table_ref(select->fromTable, source_table_plan, tables_map);
        result_plan = source_table_plan;
        std::cout << "****\tTables parsed successfully\n";
    }
    set_missing_tables_from_upper_scope_context(
        upper_scope_parser_context, tables_map, select->whereClause);
    rename_columns(select, tables_map, upper_scope_parser_context);

    if (select->fromTable->type == hsql::kTableJoin) {
        qp_from_join(select->fromTable, tables_map);
        std::cout << "****\tQP from JOIN finished!\n";
    }

    if (select->whereClause) {
        auto current_scope_parser_context =
            std::make_shared<ParserContext>(tables_map, upper_scope_parser_context, select);
        qp_where(select->whereClause, tables_map, current_scope_parser_context);
        std::cout << "****\tWhere clause parsed successfully\n";
    }

    if (select->fromTable->type == hsql::kTableName ||
        select->fromTable->type == hsql::kTableSelect) {
        // there is only one table to choose from
        result_plan = tables_map[select->fromTable->getName()];
    } else if (select->fromTable->type == hsql::kTableCrossProduct) {
        // there exist multiple tables, so we expect tables_map is result of multiple joins
        result_plan = get_query_plan_after_joins(tables_map, select->whereClause);
    } else {
        CHECK_STATUS(check_tables_map_are_same(tables_map));
        result_plan = tables_map.begin()->second;
    }

    //handling columns of SELECT list
    std::vector<std::string> ref_columns;
    std::vector<std::shared_ptr<Expression>> ref_columns_expression;
    extract_columns_select_list(select, ref_columns_expression, ref_columns);

    //handling columns required for aggregations
    // e.g sum(a * b), avg(2 * b - 1), etc.
    std::vector<std::string> agg_targets;
    std::vector<std::shared_ptr<Expression>> pre_agg_expressions;
    for (hsql::Expr* expr : *select->selectList) {
        extract_pre_aggregate(expr, agg_targets, pre_agg_expressions);
    }
    if (select->groupBy) {
        extract_pre_aggregate(select->groupBy->having, agg_targets, pre_agg_expressions);
    }

    if (!agg_targets.empty()) {
        std::vector<std::string> combined_columns = ref_columns;
        combined_columns.insert(combined_columns.end(), agg_targets.begin(), agg_targets.end());
        combined_columns = handle_duplicates(combined_columns);
        std::vector<std::shared_ptr<Expression>> combined_expressions = ref_columns_expression;
        combined_expressions.insert(
            combined_expressions.end(), pre_agg_expressions.begin(), pre_agg_expressions.end());
        // todo: set device type implicitly
        result_plan = project(result_plan, combined_expressions, combined_columns);
        std::cout << "****\tPre aggregations finished successfully\n";
    }

    //handling aggregates
    std::vector<std::shared_ptr<Aggregate>> aggregates;
    std::unordered_map<std::string, std::string> target_2_name_map;
    for (hsql::Expr* expr : *select->selectList) {
        extract_aggregate(expr, select, ref_columns, aggregates, target_2_name_map);
    }
    if (select->groupBy) {
        extract_aggregate(
            select->groupBy->having, select, ref_columns, aggregates, target_2_name_map);
    }
    if (aggregates.size()) {
        qp_group_by(select->groupBy, aggregates, result_plan);
        if (select->groupBy && select->groupBy->having) {
            std::vector<std::tuple<hsql::SelectStatement*, hsql::Expr*, hsql::OperatorType>>
                inner_selects;
            std::vector<cp::Expression> filters;
            parse_having(select->groupBy->having, inner_selects, filters);
            handle_filters(filters, result_plan);
            handle_inner_selects(inner_selects,
                                 upper_scope_parser_context,
                                 select,
                                 aggregates,
                                 result_plan,
                                 target_2_name_map);
        }
        std::cout << "****\tAggregates finished!\n";
    }

    // handling post aggregates
    std::vector<std::string> post_agg_columns;
    std::vector<std::shared_ptr<Expression>> post_agg_expresssions;
    if (extract_post_aggregate(select, post_agg_columns, post_agg_expresssions)) {
        for (auto& agg : aggregates) {
            post_agg_columns.push_back(agg->get_aggregate()->name);
            post_agg_expresssions.push_back(Expression::from_field_ref(agg->get_aggregate()->name));
        }
        std::vector<std::shared_ptr<Expression>> combined_expressions = ref_columns_expression;
        combined_expressions.insert(
            combined_expressions.end(), post_agg_expresssions.begin(), post_agg_expresssions.end());
        std::vector<std::string> combined_columns = ref_columns;
        combined_columns.insert(
            combined_columns.end(), post_agg_columns.begin(), post_agg_columns.end());
        combined_columns = handle_duplicates(combined_columns);
        // todo: set device type implicitly
        result_plan = project(result_plan, combined_expressions, combined_columns);
        std::cout << "****\tPost aggregation finished!\n";
    }

    if (select->lockings) {
        return Status(ErrorCode::MaximusError, "LOCK clause is not supported yet.");
    }

    if (select->setOperations) {
        return Status(ErrorCode::MaximusError, "Set operations are not supported yet.");
    }

    // Selecting only requiered columns
    std::vector<std::string> final_columns;
    std::vector<std::shared_ptr<Expression>> final_columns_expresssions;
    if (extract_final_columns(select, final_columns, final_columns_expresssions)) {
        // todo: set device type implicitly
        result_plan = project(result_plan, final_columns_expresssions, final_columns);
        std::cout << "****\tExtract final columns finished!\n";
    }

    if (select->order) {
        qp_order_by(select->order, ctx_, result_plan);
        std::cout << "****\tOrder By finished!\n";
    }

    if (select->limit && select->limit->limit) {
        // todo: set device type implicitly
        result_plan = limit(result_plan, select->limit->limit->ival, 0);
    }

    if (select->limit && select->limit->offset) {
        return Status(ErrorCode::MaximusError, "OFFSET clause is not supported yet.");
    }

    query_plan = result_plan;
    std::cout << "****\tQP from select finished!\n\n";
    return Status::OK();
}

Status Parser::query_plan_from_sql(const std::string& sql_query,
                                   std::shared_ptr<QueryPlan>& query_plan) {
    assert(ctx_);
    assert(db_catalogue_);
    assert(!query_plan);

    hsql::SQLParserResult result;
    hsql::SQLParser::parse(sql_query, &result);

    if (!result.isValid() || result.size() == 0) {
        return Status(ErrorCode::MaximusError, "Invalid SQL query");
    }
    assert(result.size() == 1);

    const hsql::SQLStatement* statement = result.getStatement(0);

    switch (statement->type()) {
        case hsql::kStmtSelect: {
            const auto* select = static_cast<const hsql::SelectStatement*>(statement);
            // hsql::printSelectStatementInfo(select, 0);
            // add the sink on top of the inner query plan
            auto sink_properties = std::make_shared<TableSinkProperties>();
            auto sink            = std::make_shared<QueryNode>(EngineType::NATIVE,
                                                    DeviceType::UNDEFINED,
                                                    NodeType::TABLE_SINK,
                                                    sink_properties,
                                                    ctx_);
            std::shared_ptr<QueryNode> query_plan_select;
            std::shared_ptr<ParserContext> parser_context;
            CHECK_STATUS(qp_from_select(select, query_plan_select, parser_context));
            sink->add_input(query_plan_select);

            // add the query plan root on top of the sink
            query_plan = std::make_shared<QueryPlan>(ctx_);
            assert(query_plan);
            query_plan->add_input(sink);

            assert(query_plan);
            assert(query_plan->in_degree() == 1);

            std::cout << "Full query plan = \n" << query_plan->to_string() << std::endl;
            break;
        }
        default: {
            return Status(ErrorCode::MaximusError, "Only SELECT statements are supported");
        }
    }

    return Status::OK();
}

void Parser::parse_where(
    const hsql::Expr* where_expr,
    std::vector<std::pair<hsql::Expr*, hsql::Expr*>>& join_conds,
    std::vector<std::pair<cp::Expression, std::string>>& filters,
    std::vector<std::tuple<hsql::SelectStatement*, hsql::Expr*, hsql::OperatorType>>& inner_selects,
    std::vector<arrow::compute::Expression>& post_join_filters) {
    switch (where_expr->type) {
        case hsql::kExprOperator:
            switch (where_expr->opType) {
                case hsql::kOpAnd:
                    parse_where(
                        where_expr->expr, join_conds, filters, inner_selects, post_join_filters);
                    parse_where(
                        where_expr->expr2, join_conds, filters, inner_selects, post_join_filters);
                    break;
                case hsql::kOpEquals:
                    if (where_expr->expr->type == hsql::kExprColumnRef &&
                        where_expr->expr2->type == hsql::kExprColumnRef) {
                        //todo: there might be a case where two columns belong to the same table; in this case instead of `join_cond`, `filter` should be used.
                        join_conds.push_back({where_expr->expr, where_expr->expr2});
                    } else if (where_expr->expr->type == hsql::kExprSelect ||
                               where_expr->expr2->type == hsql::kExprSelect) {
                        if (where_expr->expr->type == hsql::kExprSelect) {
                            inner_selects.push_back(
                                {where_expr->expr->select, where_expr->expr2, where_expr->opType});
                        } else {
                            inner_selects.push_back(
                                {where_expr->expr2->select, where_expr->expr, where_expr->opType});
                        }
                    } else {
                        std::string table_name = where_expr->expr->type == hsql::kExprColumnRef
                                                     ? where_expr->expr->table
                                                     : where_expr->expr2->table;
                        auto expr              = parse_expression(where_expr);
                        filters.push_back({expr, table_name});
                    }
                    break;
                case hsql::kOpNotEquals:
                    if (where_expr->expr->type == hsql::kExprColumnRef &&
                        where_expr->expr2->type == hsql::kExprColumnRef) {
                        auto expr = parse_expression(where_expr);
                        post_join_filters.push_back(expr);
                    } else if (where_expr->expr->type == hsql::kExprSelect ||
                               where_expr->expr2->type == hsql::kExprSelect) {
                        throw std::runtime_error("[parse_where]: case not supported");
                    } else {
                        std::string table_name = where_expr->expr->type == hsql::kExprColumnRef
                                                     ? where_expr->expr->table
                                                     : where_expr->expr2->table;
                        auto expr              = parse_expression(where_expr);
                        filters.push_back({expr, table_name});
                    }
                    break;
                case hsql::kOpLike:
                case hsql::kOpNotLike:
                case hsql::kOpLess:
                case hsql::kOpLessEq:
                case hsql::kOpGreater:
                case hsql::kOpGreaterEq:
                case hsql::kOpBetween:
                case hsql::kOpIn: {
                    if (where_expr->expr->type == hsql::kExprSelect ||
                        (where_expr->expr2 && where_expr->expr2->type == hsql::kExprSelect)) {
                        if (where_expr->expr->type == hsql::kExprSelect) {
                            inner_selects.push_back(
                                {where_expr->expr->select, where_expr->expr2, where_expr->opType});
                        } else {
                            inner_selects.push_back(
                                {where_expr->expr2->select, where_expr->expr, where_expr->opType});
                        }
                    } else if (where_expr->select) {
                        inner_selects.push_back(
                            {where_expr->select, where_expr->expr, hsql::kOpEquals});
                    } else {
                        std::string table_name;
                        if (where_expr->expr->type == hsql::kExprFunctionRef) {
                            for (auto& x : *where_expr->expr->exprList) {
                                if (x->type == hsql::kExprColumnRef) {
                                    table_name = x->table;
                                    break;
                                }
                            }
                        } else {
                            table_name = where_expr->expr->type == hsql::kExprColumnRef
                                             ? where_expr->expr->table
                                             : where_expr->expr2->table;
                        }
                        auto expr = parse_expression(where_expr);
                        filters.push_back({expr, table_name});
                    }
                    break;
                }
                case hsql::kOpOr: {
                    // todo: this is just for handling q7, q19 tpch, not a general solution
                    extract_join_conditions(where_expr, join_conds);
                    auto expr = parse_expression(where_expr);
                    post_join_filters.push_back(expr);
                    break;
                }
                case hsql::kOpNot: {
                    if (where_expr->expr->opType == hsql::kOpIn && where_expr->expr->select) {
                        inner_selects.push_back(
                            {where_expr->expr->select, where_expr->expr->expr, hsql::kOpNotEquals});
                    } else if (where_expr->expr->opType == hsql::kOpExists) {
                        std::unordered_set<std::string> table_names =
                            get_tables(where_expr->expr->select);
                        hsql::Expr* foreign_key_expr;
                        std::vector<hsql::Expr*> columns_inner_select;
                        extract_reference_columns(where_expr->expr->select->whereClause,
                                                  columns_inner_select);
                        for (auto& expr : columns_inner_select) {
                            if (table_names.find(expr->table) ==
                                table_names
                                    .end()) {  // table does not belong to tables of current scope SELECT
                                foreign_key_expr = expr;
                                break;
                            }
                        }
                        inner_selects.push_back(
                            {where_expr->expr->select, foreign_key_expr, hsql::kOpNotEquals});
                    } else {
                        throw std::runtime_error(
                            "[parse_where]: Unsupported case for NOT operation");
                    }
                    break;
                }
                case hsql::kOpExists: {
                    std::unordered_set<std::string> table_names = get_tables(where_expr->select);
                    hsql::Expr* foreign_key_expr;
                    std::vector<hsql::Expr*> columns_inner_select;
                    extract_reference_columns(where_expr->select->whereClause,
                                              columns_inner_select);
                    for (auto& expr : columns_inner_select) {
                        if (table_names.find(expr->table) ==
                            table_names
                                .end()) {  // table does not belong to tables of current scope SELECT
                            foreign_key_expr = expr;
                            break;
                        }
                    }
                    inner_selects.push_back(
                        {where_expr->select, foreign_key_expr, hsql::kOpExists});
                    break;
                }
                default:
                    std::cout << "operator type: " << where_expr->opType << "\n";
                    throw std::runtime_error("[parse_where]: Unsupported operator type");
            }
            break;
        default:
            break;
    }
}

arrow::compute::Expression Parser::parse_expression(const hsql::Expr* expr) {
    using arrow::compute::and_;
    using arrow::compute::call;
    using arrow::compute::equal;
    using arrow::compute::field_ref;
    using arrow::compute::greater;
    using arrow::compute::greater_equal;
    using arrow::compute::is_null;
    using arrow::compute::is_valid;
    using arrow::compute::less;
    using arrow::compute::less_equal;
    using arrow::compute::literal;
    using arrow::compute::not_;
    using arrow::compute::not_equal;
    using arrow::compute::or_;

    arrow::compute::Expression expression;

    // Handle different expression types
    switch (expr->type) {
        case hsql::kExprLiteralFloat:
            expression = literal(static_cast<double>(expr->fval));
            break;
        case hsql::kExprLiteralInt:
            expression = literal(static_cast<int64_t>(expr->ival));
            break;
        case hsql::kExprLiteralString:
            expression = literal(std::string(expr->name));
            break;
        case hsql::kExprLiteralDate:
            expression = date_literal(expr->name);
            break;
        case hsql::kExprColumnRef:
            expression = field_ref(get_column_fullname(expr));
            break;
        case hsql::kExprOperator:
            switch (expr->opType) {
                case hsql::kOpEquals:
                    expression = equal(parse_expression(expr->expr), parse_expression(expr->expr2));
                    break;
                case hsql::kOpNotEquals:
                    expression =
                        not_equal(parse_expression(expr->expr), parse_expression(expr->expr2));
                    break;
                case hsql::kOpLess:
                    expression = less(parse_expression(expr->expr), parse_expression(expr->expr2));
                    break;
                case hsql::kOpLessEq:
                    expression =
                        less_equal(parse_expression(expr->expr), parse_expression(expr->expr2));
                    break;
                case hsql::kOpGreater:
                    expression =
                        greater(parse_expression(expr->expr), parse_expression(expr->expr2));
                    break;
                case hsql::kOpGreaterEq:
                    expression =
                        greater_equal(parse_expression(expr->expr), parse_expression(expr->expr2));
                    break;
                case hsql::kOpAnd:
                    expression = and_(parse_expression(expr->expr), parse_expression(expr->expr2));
                    break;
                case hsql::kOpOr:
                    expression = or_(parse_expression(expr->expr), parse_expression(expr->expr2));
                    break;
                case hsql::kOpNot:
                    expression = not_(parse_expression(expr->expr));
                    break;
                case hsql::kOpIsNull:
                    expression = is_null(parse_expression(expr->expr));
                    break;
                case hsql::kOpExists:
                    expression = is_valid(parse_expression(expr->expr));
                    break;
                case hsql::kOpPlus:
                    expression =
                        call("add", {parse_expression(expr->expr), parse_expression(expr->expr2)});
                    break;
                case hsql::kOpMinus:
                    expression = call(
                        "subtract", {parse_expression(expr->expr), parse_expression(expr->expr2)});
                    break;
                case hsql::kOpAsterisk:
                    expression = call(
                        "multiply", {parse_expression(expr->expr), parse_expression(expr->expr2)});
                    break;
                case hsql::kOpSlash:
                    expression = call(
                        "divide", {parse_expression(expr->expr), parse_expression(expr->expr2)});
                    break;
                case hsql::kOpLike:
                    expression = arrow_like(parse_expression(expr->expr), expr->expr2->name);
                    break;
                case hsql::kOpNotLike:
                    expression =
                        arrow_not(arrow_like(parse_expression(expr->expr), expr->expr2->name));
                    break;
                case hsql::kOpBetween:
                    expression = arrow_between(parse_expression(expr->expr),
                                               parse_expression((*expr->exprList)[0]),
                                               parse_expression((*expr->exprList)[1]));
                    break;
                case hsql::kOpCase: {
                    auto _cond  = parse_expression(expr->exprList->front()->expr);
                    auto _true  = parse_expression(expr->exprList->front()->expr2);
                    auto _false = parse_expression(expr->expr2);
                    expression  = arrow_if_else(_cond, _true, _false);
                    break;
                }
                case hsql::kOpIn: {
                    std::vector<arrow::compute::Expression> range;
                    for (auto& x : *expr->exprList) {
                        range.push_back(parse_expression(x));
                    }
                    expression = arrow_in(parse_expression(expr->expr), range);
                    break;
                }
                // Handle other operators similarly
                // ...
                default:
                    std::cout << "operator type: " << expr->opType << "\n";
                    throw std::runtime_error("[parse_expression]: Unsupported operator type");
            }
            break;
        case hsql::kExprCast:
            switch (expr->columnType.data_type) {
                case hsql::DataType::DATE:
                    expression = date_literal(expr->expr->name);
                    break;
            }
            break;
        case hsql::kExprExtract:
            switch (expr->datetimeField) {
                case hsql::kDatetimeYear:
                    expression = year(parse_expression(expr->expr));
                    break;
                default:
                    throw std::runtime_error("Unsupported datetime field");
            }
            break;
        case hsql::kExprFunctionRef:
            if (extract_hash_function(expr) == "substring") {
                auto x_list = *expr->exprList;
                expression  = arrow_substring(
                    parse_expression(x_list[0]), x_list[1]->ival - 1, x_list[2]->ival);
            } else {
                expression = field_ref(get_aggregate_name(expr));
            }
            break;
        default:
            std::cout << "expression type: " << expr->type << "\n";
            throw std::runtime_error("[parse_expression]: Unsupported expression type");
    }
    return expression;
}

void Parser::extract_reference_columns(hsql::Expr* expr, std::vector<hsql::Expr*>& columns) {
    switch (expr->type) {
        case hsql::kExprStar:
            break;
        case hsql::kExprColumnRef:
            columns.push_back(expr);
            break;
        case hsql::kExprOperator:
            switch (expr->opType) {
                case hsql::kOpLike:
                case hsql::kOpNotLike:
                case hsql::kOpEquals:
                case hsql::kOpNotEquals:
                case hsql::kOpLess:
                case hsql::kOpLessEq:
                case hsql::kOpGreater:
                case hsql::kOpGreaterEq:
                case hsql::kOpAnd:
                case hsql::kOpOr:
                case hsql::kOpIsNull:
                case hsql::kOpPlus:
                case hsql::kOpMinus:
                case hsql::kOpAsterisk:
                case hsql::kOpSlash:
                case hsql::kOpCaseListElement:
                    extract_reference_columns(expr->expr, columns);
                    extract_reference_columns(expr->expr2, columns);
                    break;
                case hsql::kOpNot:
                case hsql::kOpBetween:
                case hsql::kOpIn:
                    extract_reference_columns(expr->expr, columns);
                    break;
                case hsql::kOpCase:
                    extract_reference_columns(expr->expr2, columns);
                    extract_reference_columns(expr->exprList->front(), columns);
                    break;
                case hsql::kOpExists:
                    extract_reference_columns(expr->select->whereClause, columns);
                    break;
                default:
                    std::cout << "operator type: " << expr->opType << "\n";
                    throw std::runtime_error(
                        "[extract_reference_columns]: Unsupported operator type");
            }
            break;
        case hsql::kExprFunctionRef:
            for (auto* x : *expr->exprList) {
                extract_reference_columns(x, columns);
            }
            break;
        case hsql::kExprExtract:
            extract_reference_columns(expr->expr, columns);
            break;
        default:
            break;
    }
}

void Parser::rename_columns(const hsql::SelectStatement* select,
                            std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map,
                            const std::shared_ptr<ParserContext> upper_layer_context) {
    // This function first extract all of the used column references in the query
    // Then it raname them with their relative table name as TABLE.COLUMN_NAME
    // In this way it makes a unique name for each column reference and prevent future conflicts

    std::vector<hsql::Expr*> columns_expr;
    for (hsql::Expr* expr : *select->selectList) {
        extract_reference_columns(expr, columns_expr);
    }
    if (select->whereClause) extract_reference_columns(select->whereClause, columns_expr);
    if (select->fromTable->join)
        extract_reference_columns(select->fromTable->join->condition, columns_expr);
    if (select->groupBy && select->groupBy->having)
        extract_reference_columns(select->groupBy->having, columns_expr);

    std::unordered_map<std::string, std::vector<std::string>> old_columns;
    std::unordered_map<std::string, std::vector<std::string>> new_columns;
    for (auto& expr : columns_expr) {
        old_columns[expr->table].push_back(expr->name);
        new_columns[expr->table].push_back(get_column_fullname(expr));
    }
    for (auto& [table_name, old_names] : old_columns) {
        if (!table_exist_in_select(select->fromTable,
                                   table_name))  // to avoid renaming already renamed columns
            continue;

        auto new_names = new_columns[table_name];
        new_names      = handle_duplicates(new_names);
        // todo: to the DeviceType in an automatic way
        auto rename_plan = rename(tables_map[table_name], old_names, new_names, DeviceType::CPU);
        tables_map[table_name] = rename_plan;
    }
    std::cout << "****\tColumns renamed successfully\n";
}

std::string Parser::get_column_fullname(const hsql::Expr* expr) {
    if (expr->hasTable()) return std::string(expr->table) + "." + expr->name;

    return expr->name;
}

void Parser::parse_having(
    const hsql::Expr* expr,
    std::vector<std::tuple<hsql::SelectStatement*, hsql::Expr*, hsql::OperatorType>>& inner_selects,
    std::vector<cp::Expression>& filters) {
    switch (expr->type) {
        case hsql::kExprOperator:
            switch (expr->opType) {
                case hsql::kOpLess:
                case hsql::kOpLessEq:
                case hsql::kOpGreater:
                case hsql::kOpGreaterEq:
                    if (expr->expr->type == hsql::kExprSelect ||
                        expr->expr2->type == hsql::kExprSelect) {
                        if (expr->expr->type == hsql::kExprSelect) {
                            inner_selects.push_back(
                                {expr->expr->select, expr->expr2, expr->opType});
                        } else {
                            inner_selects.push_back(
                                {expr->expr2->select, expr->expr, expr->opType});
                        }
                    } else if (expr->expr->type == hsql::kExprFunctionRef ||
                               expr->expr2->type == hsql::kExprFunctionRef) {
                        auto f = parse_expression(expr);
                        filters.push_back(f);
                    } else {
                        throw std::runtime_error("[Parse Having]: unsupported case!");
                    }
                    break;
                default:
                    throw std::runtime_error("[Parse Having]: Unsupported operator type.");
            }
            break;

        default:
            throw std::runtime_error("[Parse Having]: unsupported expression type.");
    }
}

bool Parser::table_exist_in_select(const hsql::TableRef* table, const std::string table_name) {
    switch (table->type) {
        case hsql::TableRefType::kTableName: {
            if (table_name == table->getName()) return true;
            break;
        }

        case hsql::TableRefType::kTableJoin: {
            if (table_name == table->join->left->getName()) return true;
            if (table_name == table->join->right->getName()) return true;
            break;
        }

        case hsql::TableRefType::kTableSelect: {
            if (table_name == table->getName()) return true;
            break;
        }

        case hsql::TableRefType::kTableCrossProduct: {
            for (auto& t : *table->list) {
                if (table_exist_in_select(t, table_name)) return true;
            }
            break;
        }
    }
    return false;
}

void Parser::set_missing_tables_from_upper_scope_context(
    const std::shared_ptr<ParserContext> upper_layer_context,
    std::unordered_map<std::string, std::shared_ptr<QueryNode>>& tables_map,
    hsql::Expr* where_expr) {
    if (where_expr == nullptr) return;

    std::vector<hsql::Expr*> columns_expr;
    extract_reference_columns(where_expr, columns_expr);
    for (auto& expr : columns_expr) {
        std::string table_name = expr->table;
        if (tables_map.find(table_name) ==
            tables_map
                .end()) {  // table does not exist in current tables_map --> it should be gotten from upper scope
            auto ctx = upper_layer_context;
            while (
                ctx != nullptr &&
                ctx->tables_map.find(table_name) ==
                    ctx->tables_map.end()) {  // still table cannot be found --> check upper scope
                ctx = ctx->upper_context;
            }
            if (ctx == nullptr) {
                std::cout << "WARNING [set_missing_tables_from_upper_scope_context]: The table "
                             "does not exist in the upper scopes!\n";
                std::cout << "WARNING [set_missing_tables_from_upper_scope_context]: Table name: "
                          << table_name << "\n";
            } else {
                tables_map[table_name] = ctx->tables_map[table_name];
            }
        }
    }
    std::cout << "****\tSet missing tables from upper scope finished!\n";
}

void Parser::extract_join_conditions(const hsql::Expr* expr,
                                     std::vector<std::pair<hsql::Expr*, hsql::Expr*>>& join_conds) {
    switch (expr->type) {
        case hsql::kExprOperator:
            switch (expr->opType) {
                case hsql::kOpEquals:
                    if (expr->expr->type == hsql::kExprColumnRef &&
                        expr->expr2->type == hsql::kExprColumnRef) {
                        if (expr->expr->table != expr->expr2->table) {  // otherwise it is a filter
                            join_conds.push_back({expr->expr, expr->expr2});
                        }
                    }
                    break;
                case hsql::kOpAnd:
                case hsql::kOpOr:
                    extract_join_conditions(expr->expr, join_conds);
                    extract_join_conditions(expr->expr2, join_conds);
                    break;
                case hsql::kOpGreater:
                case hsql::kOpGreaterEq:
                case hsql::kOpLess:
                case hsql::kOpLessEq:
                case hsql::kOpIn:
                case hsql::kOpBetween:
                    break;
                default:
                    std::cout << "expr operator: " << expr->opType << "\n";
                    throw std::runtime_error(
                        "[extract_join_conditions]: unsupported operator type.");
            }
            break;
        default:
            break;
    }
}

std::unordered_set<std::string> Parser::get_tables(const hsql::SelectStatement* select) {
    std::unordered_set<std::string> res;
    switch (select->fromTable->type) {
        case hsql::kTableName:
            res.insert(select->fromTable->getName());
            break;
        case hsql::kTableCrossProduct:
            for (auto& t : *select->fromTable->list) {
                res.insert(t->getName());
            }
            break;
        default:
            break;
    }
    return res;
}

std::shared_ptr<arrow::DataType> hsql_datatype_2_arrow_datatype(
    const hsql::ColumnType& columnType) {
    switch (columnType.data_type) {
        case hsql::DataType::BIGINT:
        case hsql::DataType::LONG:
            return arrow::int64();
        case hsql::DataType::BOOLEAN:
            return arrow::boolean();
        case hsql::DataType::CHAR:
        case hsql::DataType::VARCHAR:
            return arrow::utf8();
        case hsql::DataType::TEXT:
            return arrow::utf8();
        case hsql::DataType::DATE:
            return arrow::date32();
        case hsql::DataType::DATETIME:
            return arrow::timestamp(arrow::TimeUnit::MICRO);
        case hsql::DataType::DECIMAL:
            return arrow::decimal128(columnType.precision, columnType.scale);
        case hsql::DataType::DOUBLE:
        case hsql::DataType::REAL:
            return arrow::float64();
        case hsql::DataType::FLOAT:
            return arrow::float32();
        case hsql::DataType::INT:
            return arrow::int32();
        case hsql::DataType::SMALLINT:
            return arrow::int16();
        case hsql::DataType::TIME:
            return arrow::time64(arrow::TimeUnit::MICRO);
        default:
            return arrow::null();
    }
}

Status Parser::schema_from_sql(std::string sql_string,
                               std::unordered_map<std::string, std::shared_ptr<Schema>>& schemas) {
    hsql::SQLParserResult result;
    hsql::SQLParser::parse(sql_string, &result);

    if (!result.isValid() || result.size() == 0) {
        return Status(ErrorCode::MaximusError, "Invalid SQL query");
    }

    const hsql::SQLStatement* statement = result.getStatement(0);

    switch (statement->type()) {
        case hsql::kStmtCreate: {
            // hsql::printCreateStatementInfo(create, 0);
            const auto* create = static_cast<const hsql::CreateStatement*>(statement);
            std::vector<std::shared_ptr<arrow::Field>> fields;
            for (auto& col : *create->columns) {
                fields.push_back(arrow::field(
                    col->name, hsql_datatype_2_arrow_datatype(col->type), col->nullable));
            }
            schemas[create->tableName] = std::make_shared<Schema>(fields);
            break;
        }
        default: {
            std::cout << "statement type: " << statement->type() << "\n";
            return Status(ErrorCode::MaximusError,
                          "Only CREATE statements are supported for parsing schema");
        }
    }

    return Status::OK();
}

}  // namespace maximus
