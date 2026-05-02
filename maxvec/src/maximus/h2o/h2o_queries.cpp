#include <maximus/frontend/expressions.hpp>
#include <maximus/frontend/query_plan_api.hpp>
#include <maximus/h2o/h2o_queries.hpp>
#include <maximus/types/expression.hpp>

namespace maximus::h2o {
std::shared_ptr<arrow::DataType> max_size_string(int size) {
    // return arrow::fixed_size_binary(size);
    return arrow::utf8();
}

std::shared_ptr<arrow::DataType> fixed_size_string(int size) {
    // return arrow::fixed_size_binary(size);
    return arrow::utf8();
}

std::shared_ptr<Schema> schema(const std::string& table_name) {
    if (table_name == "groupby" || table_name == "groupby_nan") {
        // todo: maybe changing id1, id2, id3 to CATEGORICAL type in arrow, I tried once by using arrow::dictionary but
        // I got the following error: ../src/arrow/array/array_dict.cc:84:  Check failed: (data->dictionary) != (nullptr)
        auto fields = {arrow::field("id1", max_size_string(5), false),
                       arrow::field("id2", max_size_string(5), false),
                       arrow::field("id3", max_size_string(12), false),
                       arrow::field("id4", arrow::int32(), false),
                       arrow::field("id5", arrow::int32(), false),
                       arrow::field("id6", arrow::int32(), false),
                       arrow::field("v1", arrow::int32(), false),
                       arrow::field("v2", arrow::int32(), false),
                       arrow::field("v3", arrow::float64(), false)};
        return std::make_shared<Schema>(fields);
    }
    throw std::runtime_error("The schema for given table not known.");
}

std::vector<std::string> table_names() {
    return {"groupby"};
}

std::vector<std::shared_ptr<Schema>> schemas() {
    auto tables = table_names();
    std::vector<std::shared_ptr<maximus::Schema>> table_schemas;
    table_schemas.reserve(tables.size());
    for (const auto& table : tables) {
        table_schemas.push_back(maximus::h2o::schema(table));
    }
    return table_schemas;
}

std::shared_ptr<QueryPlan> q1(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "groupby";
    auto source_node = table_source(db, table_name, schema(table_name), {"id1", "v1"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    auto sum_opts                                = sum_defaults();
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_sum", sum_opts, "v1", "v1_sum")};
    auto group_by_node = group_by(source_node, {"id1"}, aggs, device);
    /* =================================
     * CREATING A ORDER BY NODE
     * =================================
     */
    std::vector<SortKey> sort_keys = {{"v1_sum", SortOrder::ASCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);

    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q2(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "groupby";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"id1", "id2", "v1"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    auto sum_opts                                = sum_defaults();
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_sum", sum_opts, "v1", "v1_sum")};
    auto group_by_node = group_by(source_node, {"id1", "id2"}, aggs, device);
    /* =================================
     * CREATING A ORDER BY NODE
     * =================================
     */
    std::vector<SortKey> sort_keys = {{"v1_sum", SortOrder::ASCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q3(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "groupby";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"id3", "v1", "v3"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    auto sum_opts                                = sum_defaults();
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_sum", sum_opts, "v1", "v1_sum"),
        aggregate("hash_mean", sum_opts, "v3", "v3_mean")};
    auto group_by_node = group_by(source_node, {"id3"}, aggs, device);
    /* =================================
     * CREATING A ORDER BY NODE
     * =================================
     */
    std::vector<SortKey> sort_keys = {{"v1_sum", SortOrder::ASCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q4(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "groupby";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"id4", "v1", "v2", "v3"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    auto sum_opts                                = sum_defaults();
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_mean", sum_opts, "v1", "v1_mean"),
        aggregate("hash_mean", sum_opts, "v2", "v2_mean"),
        aggregate("hash_mean", sum_opts, "v3", "v3_mean")};
    auto group_by_node = group_by(source_node, {"id4"}, aggs, device);
    /* =================================
     * CREATING A ORDER BY NODE
     * =================================
     */
    std::vector<SortKey> sort_keys = {{"v1_mean", SortOrder::ASCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q5(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "groupby";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"id6", "v1", "v2", "v3"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    auto sum_opts                                = sum_defaults();
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_sum", sum_opts, "v1", "v1_sum"),
        aggregate("hash_sum", sum_opts, "v2", "v2_sum"),
        aggregate("hash_sum", sum_opts, "v3", "v3_sum")};
    auto group_by_node = group_by(source_node, {"id6"}, aggs, device);
    /* =================================
     * CREATING A ORDER BY NODE
     * =================================
     */
    std::vector<SortKey> sort_keys = {{"v1_sum", SortOrder::ASCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q6(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "groupby";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"id4", "id5", "v3"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    auto med                                     = median();
    auto stdd                                    = stddev();
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_approximate_median", med, "v3", "v3_median"),
        aggregate("hash_stddev", stdd, "v3", "v3_std")};
    auto group_by_node = group_by(source_node, {"id4", "id5"}, aggs, device);
    /* =================================
     * CREATING A ORDER BY NODE
     * =================================
     */
    std::vector<SortKey> sort_keys = {{"v3_median", SortOrder::ASCENDING},
                                      {"v3_std", SortOrder::ASCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q7(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "groupby";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"id3", "v1", "v2"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_max", "v1", "v1_max"),
        aggregate("hash_min", "v2", "v2_min"),
    };
    auto group_by_node = group_by(source_node, {"id3"}, aggs, device);
    /* =================================
     * CREATING A ORDER BY NODE
     * =================================
     */
    auto range_v1_v2_expr = expr(arrow_expr(cp::field_ref("v1_max"), "-", cp::field_ref("v2_min")));
    auto all_projected    = project(group_by_node,
                                    {expr(cp::field_ref("id3")), range_v1_v2_expr},
                                    {"id3", "range_v1_v2"},
                                 device);
    std::vector<SortKey> sort_keys = {{"range_v1_v2", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(all_projected, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q8(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "groupby_nan";
    auto source_node = table_source(db, table_name, schema(table_name), {"id6", "v3"}, device);
    std::cout << "$$$$$$$$$$$$$$"
              << "\n";
    /* =================================
     * CREATING A FILTER NODE
     * =================================
     */
    auto drop_null_expr = expr(arrow_is_not_null(cp::field_ref("v3")));
    auto filter_node    = filter(source_node, drop_null_expr, device);
    /* =================================
     * CREATING A ORDER BY NODE
     * =================================
     */
    std::vector<SortKey> sort_keys = {{"v3", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(filter_node, sort_keys, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    // todo: a new aggregation method should be supported by Arrow
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_list", "v3", "v3"),
    };
    // auto group_by_node = group_by(order_by_node, {"id6"}, aggs, device);
    auto group_by_node = group_by(order_by_node, {"id6"}, aggs, device);
    auto limit_node    = limit(group_by_node, 2, 0, device);

    return query_plan(table_sink(limit_node));
}

std::shared_ptr<QueryPlan> q9(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "groupby";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"id2", "id4", "v1", "v2"}, device);

    /* ==============================
     * CREATING NEW PRODUCT COLUMN
     * ==============================
     */
    auto product_v1_v2 = arrow_product({cp::field_ref("v1"), cp::field_ref("v2")});
    auto projection1   = project(source_node,
                                 {expr(cp::field_ref("id2")),
                                  expr(cp::field_ref("id4")),
                                  expr(cp::field_ref("v1")),
                                  expr(cp::field_ref("v2")),
                                  expr(product_v1_v2)},
                                 {"id2", "id4", "v1", "v2", "v1v2"},
                               device);

    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    auto stdd                                    = stddev();
    auto sum_opts                                = sum_defaults();
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_mean", sum_opts, "v1", "v1_mean"),
        aggregate("hash_mean", sum_opts, "v2", "v2_mean"),
        aggregate("hash_mean", sum_opts, "v1v2", "v1v2_mean"),
        aggregate("hash_stddev", stdd, "v1", "v1_std"),
        aggregate("hash_stddev", stdd, "v2", "v2_std"),
    };
    auto group_by_node = group_by(projection1, {"id2", "id4"}, aggs, device);

    /* ==============================
     * CALCULATING CORRELATION
     * ==============================
     */
    auto v1_std_v2_std   = arrow_product({cp::field_ref("v1_std"), cp::field_ref("v2_std")});
    auto v1_mean_v2_mean = arrow_product({cp::field_ref("v1_mean"), cp::field_ref("v2_mean")});
    auto cov             = arrow_expr(cp::field_ref("v1v2_mean"), "-", v1_mean_v2_mean);
    auto corr            = arrow_expr(cov, "/", v1_std_v2_std);
    auto corr_square     = arrow_product({corr, corr});

    auto projection2 =
        project(group_by_node,
                {expr(cp::field_ref("id2")), expr(cp::field_ref("id4")), expr(corr_square)},
                {"id2", "id4", "r2"},
                device);

    /* =================================
     * CREATING A ORDER BY NODE
     * =================================
     */
    std::vector<SortKey> sort_keys = {{"r2", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(projection2, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q10(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "groupby";
    auto source_node       = table_source(db,
                                    table_name,
                                    schema(table_name),
                                          {"id1", "id2", "id3", "id4", "id5", "id6", "v3", "v1"},
                                    device);

    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    auto sum_opts   = sum_defaults();
    auto count_opts = count_all();

    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_sum", sum_opts, "v3", "v3"),
        aggregate("hash_count", count_opts, "v1", "count"),
    };
    auto group_by_node =
        group_by(source_node, {"id1", "id2", "id3", "id4", "id5", "id6"}, aggs, device);
    /* =================================
     * CREATING A ORDER BY NODE
     * =================================
     */
    std::vector<SortKey> sort_keys = {{"v3", SortOrder::DESCENDING},
                                      {"count", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> query_plan(const std::string& q,
                                      std::shared_ptr<Database>& db,
                                      DeviceType device) {
    if (q == "q1") {
        return q1(db, device);
    }
    if (q == "q2") {
        return q2(db, device);
    }
    if (q == "q3") {
        return q3(db, device);
    }
    if (q == "q4") {
        return q4(db, device);
    }
    if (q == "q5") {
        return q5(db, device);
    }
    if (q == "q6") {
        return q6(db, device);
    }
    if (q == "q7") {
        return q7(db, device);
    }
    if (q == "q8") {
        throw std::runtime_error("Not implemented yet.");
    }
    if (q == "q9") {
        return q9(db, device);
    }
    if (q == "q10") {
        return q10(db, device);
    }
    throw std::runtime_error("Non-existing h2o query.");
}
}  // namespace maximus::h2o
