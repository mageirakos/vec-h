#include <maximus/frontend/expressions.hpp>
#include <maximus/frontend/query_plan_api.hpp>
#include <maximus/tpch/tpch_queries.hpp>
#include <maximus/types/expression.hpp>

namespace maximus::tpch {
// currently, fixed_size_binary in arrow requires that all the strings
// are padded to have exactly the given length. In most of the generated CSV files for TPCH
// the strings are not padded. So, we use utf8 instead of fixed_size_binary.
std::shared_ptr<arrow::DataType> fixed_size_string(int size) {
    // return arrow::fixed_size_binary(size);
    return arrow::utf8();
}

std::shared_ptr<arrow::DataType> max_size_string(int size) {
    // return arrow::fixed_size_binary(size);
    return arrow::utf8();
}

std::shared_ptr<Schema> schema(const std::string& table_name) {
    if (table_name == "nation") {
        auto fields = {arrow::field("n_nationkey", arrow::int32(), false),
                       arrow::field("n_name", max_size_string(25), false),
                       arrow::field("n_regionkey", arrow::int32(), false),
                       arrow::field("n_comment", max_size_string(152), true)};
        return std::make_shared<Schema>(fields);
    }

    if (table_name == "region") {
        auto fields = {arrow::field("r_regionkey", arrow::int32(), false),
                       arrow::field("r_name", max_size_string(25), false),
                       arrow::field("r_comment", max_size_string(152), true)};
        return std::make_shared<Schema>(fields);
    }

    if (table_name == "part") {
        auto fields = {arrow::field("p_partkey", arrow::int64(), false),
                       arrow::field("p_name", max_size_string(55), false),
                       arrow::field("p_mfgr", max_size_string(25), false),
                       arrow::field("p_brand", max_size_string(10), false),
                       arrow::field("p_type", max_size_string(25), false),
                       arrow::field("p_size", arrow::int32(), false),
                       arrow::field("p_container", max_size_string(10), false),
                       arrow::field("p_retailprice", arrow::float64(), false),
                       arrow::field("p_comment", max_size_string(23), false)};
        return std::make_shared<Schema>(fields);
    }

    if (table_name == "supplier") {
        auto fields = {arrow::field("s_suppkey", arrow::int64(), false),
                       arrow::field("s_name", max_size_string(25), false),
                       arrow::field("s_address", max_size_string(40), false),
                       arrow::field("s_nationkey", arrow::int32(), false),
                       arrow::field("s_phone", fixed_size_string(15), false),
                       arrow::field("s_acctbal", arrow::float64(), false),
                       arrow::field("s_comment", max_size_string(101), false)};
        return std::make_shared<Schema>(fields);
    }

    if (table_name == "partsupp") {
        auto fields = {arrow::field("ps_partkey", arrow::int64(), false),
                       arrow::field("ps_suppkey", arrow::int64(), false),
                       arrow::field("ps_availqty",
                                    arrow::float64(),
                                    false),  // CHANGED: Originally, this should be int64.
                                             // However, in q20, this is compared to the threshold
                       // which is 0.5 * l_quantity, where l_quantity is of type float64
                       // Since cudf, cannot do the comparison when the types are different
                       // here we set the ps_availqty to float64, to be the same type
                       // as l_quantity.
                       arrow::field("ps_supplycost", arrow::float64(), false),
                       arrow::field("ps_comment", max_size_string(199), false)};
        return std::make_shared<Schema>(fields);
    }

    if (table_name == "customer") {
        auto fields = {arrow::field("c_custkey", arrow::int64(), false),
                       arrow::field("c_name", max_size_string(25), false),
                       arrow::field("c_address", max_size_string(40), false),
                       arrow::field("c_nationkey", arrow::int32(), false),
                       arrow::field("c_phone", fixed_size_string(15), false),
                       arrow::field("c_acctbal", arrow::float64(), false),
                       arrow::field("c_mktsegment", max_size_string(10), false),
                       arrow::field("c_comment", max_size_string(117), false)};
        return std::make_shared<Schema>(fields);
    }

    if (table_name == "orders") {
        auto fields = {arrow::field("o_orderkey", arrow::int64(), false),
                       arrow::field("o_custkey", arrow::int64(), false),
                       arrow::field("o_orderstatus", fixed_size_string(1), false),
                       arrow::field("o_totalprice", arrow::float64(), false),
                       arrow::field("o_orderdate", arrow::date32(), false),
                       arrow::field("o_orderpriority", max_size_string(15), false),
                       arrow::field("o_clerk", fixed_size_string(15), false),
                       arrow::field("o_shippriority", arrow::int32(), false),
                       arrow::field("o_comment", max_size_string(79), false)};
        return std::make_shared<Schema>(fields);
    }

    if (table_name == "lineitem") {
        auto fields = {arrow::field("l_orderkey", arrow::int64(), false),
                       arrow::field("l_partkey", arrow::int64(), false),
                       arrow::field("l_suppkey", arrow::int64(), false),
                       arrow::field("l_linenumber", arrow::int64(), false),
                       arrow::field("l_quantity", arrow::float64(), false),
                       arrow::field("l_extendedprice", arrow::float64(), false),
                       arrow::field("l_discount", arrow::float64(), false),
                       arrow::field("l_tax", arrow::float64(), false),
                       arrow::field("l_returnflag", fixed_size_string(1), false),
                       arrow::field("l_linestatus", fixed_size_string(1), false),
                       arrow::field("l_shipdate", arrow::date32(), false),
                       arrow::field("l_commitdate", arrow::date32(), false),
                       arrow::field("l_receiptdate", arrow::date32(), false),
                       arrow::field("l_shipinstruct", max_size_string(25), false),
                       arrow::field("l_shipmode", max_size_string(10), false),
                       arrow::field("l_comment", max_size_string(44), false)};
        return std::make_shared<Schema>(fields);
    }

    throw std::runtime_error("The schema for given table not known.");
}

std::vector<std::string> table_names() {
    return {"lineitem", "orders", "customer", "part", "partsupp", "supplier", "nation", "region"};
}

std::vector<std::shared_ptr<Schema>> schemas() {
    auto tables = table_names();
    std::vector<std::shared_ptr<maximus::Schema>> table_schemas;
    table_schemas.reserve(tables.size());
    for (const auto& table : tables) {
        table_schemas.push_back(maximus::tpch::schema(table));
    }
    return table_schemas;
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
        return q8(db, device);
    }
    if (q == "q9") {
        return q9(db, device);
    }
    if (q == "q10") {
        return q10(db, device);
    }
    if (q == "q11") {
        return q11(db, device);
    }
    if (q == "q12") {
        return q12(db, device);
    }
    if (q == "q13") {
        return q13(db, device);
    }
    if (q == "q14") {
        return q14(db, device);
    }
    if (q == "q15") {
        return q15(db, device);
    }
    if (q == "q16") {
        return q16(db, device);
    }
    if (q == "q17") {
        return q17(db, device);
    }
    if (q == "q18") {
        return q18(db, device);
    }
    if (q == "q19") {
        return q19(db, device);
    }
    if (q == "q20") {
        return q20(db, device);
    }
    if (q == "q21") {
        return q21_optimized(db, device);
        // return q21(db, device);
    }
    if (q == "q22") {
        return q22(db, device);
    }

    auto tables = table_names();
    // added for additional benchmarking
    if (std::find(tables.begin(), tables.end(), q) != tables.end()) {
        return q_empty(db, device, q);
    }

    throw std::runtime_error("Non-existing TPCH query.");
}

std::shared_ptr<QueryPlan> q1(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();

    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */

    auto source_node = table_source(db,
                                    "lineitem",
                                    schema("lineitem"),
                                    {"l_shipdate",
                                     "l_returnflag",
                                     "l_linestatus",
                                     "l_quantity",
                                     "l_extendedprice",
                                     "l_discount",
                                     "l_tax"},
                                    device);

    /* ==============================
     * CREATING A FILTER NODE
     * ==============================
     */
    auto sept_2_1998 = date_literal("1998-09-02");
    auto filter_expr = expr(arrow_expr(cp::field_ref("l_shipdate"), "<=", sept_2_1998));
    auto filter_node = filter(source_node, filter_expr, device);

    /* ==============================
     * CREATING A PROJECT NODE
     * ==============================
     */
    // these are arrow::compute::Expression objects
    auto _one                = float64_literal(1.00);
    auto _one_minus_discount = arrow_expr(_one, "-", cp::field_ref("l_discount"));
    auto _one_plus_tax       = arrow_expr(_one, "+", cp::field_ref("l_tax"));
    auto _disc_price = arrow_expr(cp::field_ref("l_extendedprice"), "*", _one_minus_discount);
    auto _charge     = arrow_product({_disc_price, _one_plus_tax});

    // these are std::shared_ptr<maximus::Expression> objects
    // observe that arrow expressions have an underscore prefix whereas maximus expressions do not
    auto l_returnflag = Expression::from_field_ref("l_returnflag");
    auto l_linestatus = Expression::from_field_ref("l_linestatus");
    auto quantity     = Expression::from_field_ref("l_quantity");
    auto base_price   = Expression::from_field_ref("l_extendedprice");
    auto discount     = Expression::from_field_ref("l_discount");
    auto charge       = expr(_charge);
    auto disc_price   = expr(_disc_price);

    std::vector<std::shared_ptr<Expression>> projection_list = {l_returnflag,
                                                                l_linestatus,
                                                                quantity,
                                                                base_price,
                                                                disc_price,
                                                                charge,
                                                                quantity,
                                                                base_price,
                                                                discount};

    std::vector<std::string> project_names = {"l_returnflag",
                                              "l_linestatus",
                                              "sum_qty",
                                              "sum_base_price",
                                              "sum_disc_price",
                                              "sum_charge",
                                              "avg_qty",
                                              "avg_price",
                                              "avg_disc"};

    auto project_node = project(filter_node, projection_list, project_names, device);

    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    auto sum_opts                                = sum_defaults();
    auto count_opts                              = count_all();
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_sum", sum_opts, "sum_qty", "sum_qty"),
        aggregate("hash_sum", sum_opts, "sum_base_price", "sum_base_price"),
        aggregate("hash_sum", sum_opts, "sum_disc_price", "sum_disc_price"),
        aggregate("hash_sum", sum_opts, "sum_charge", "sum_charge"),
        aggregate("hash_mean", sum_opts, "avg_qty", "avg_qty"),
        aggregate("hash_mean", sum_opts, "avg_price", "avg_price"),
        aggregate("hash_mean", sum_opts, "avg_disc", "avg_disc"),
        aggregate("hash_count", count_opts, "l_returnflag", "count_order")};
    auto group_by_node = group_by(project_node, {"l_returnflag", "l_linestatus"}, aggs, device);

    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"l_returnflag"}, {"l_linestatus"}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);

    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q2(std::shared_ptr<Database>& db, DeviceType device) {
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    auto part = table_source(
        db, "part", schema("part"), {"p_partkey", "p_type", "p_size", "p_mfgr"}, device);

    auto partsupp = table_source(
        db, "partsupp", schema("partsupp"), {"ps_partkey", "ps_suppkey", "ps_supplycost"}, device);

    auto supplier = table_source(
        db,
        "supplier",
        schema("supplier"),
        {"s_suppkey", "s_nationkey", "s_acctbal", "s_name", "s_address", "s_phone", "s_comment"},
        device);

    auto nation = table_source(db, "nation", schema("nation"), {}, device);

    auto region = table_source(db, "region", schema("region"), {}, device);

    /* ==============================
     * CREATING A FILTER FOR PART
     * ==============================
     */
    auto part_filter =
        expr(arrow_expr(arrow_expr(cp::field_ref("p_size"),
                                   "==",
                                   int32_literal(15)),  // CHANGED: originally, this should be 15
                        "&&",
                        arrow_field_ends_with("p_type", "BRASS")));

    auto filter_part_node = filter(part, part_filter, device);

    /* ==============================
     * CREATING A PROJECT FOR PART
     * ==============================
     */
    auto part_project_node = project(filter_part_node, {"p_partkey", "p_mfgr"}, device);

    /* ==============================
     * CREATING PSP JOIN
     * ==============================
     */
    auto psp_join_node =
        inner_join(part_project_node, partsupp, {"p_partkey"}, {"ps_partkey"}, "", "", device);

    /* ==============================
     * CREATING A PSPS JOIN
     * ==============================
     */
    auto psps_join_node =
        inner_join(psp_join_node, supplier, {"ps_suppkey"}, {"s_suppkey"}, "", "", device);

    /* =================================
     * CREATING A PROJECT FOR PSPS JOIN
     * =================================
     */
    auto psps_project_node = project(psps_join_node,
                                     {"p_partkey",
                                      "ps_supplycost",
                                      "p_mfgr",
                                      "s_nationkey",
                                      "s_acctbal",
                                      "s_name",
                                      "s_address",
                                      "s_phone",
                                      "s_comment"},
                                     device);

    /* =================================
     * CREATING A REGION FILTER
     * =================================
     */
    auto region_filter = expr(arrow_expr(cp::field_ref("r_name"), "==", string_literal("EUROPE")));
    auto region_filter_node = filter(region, region_filter, device);

    /* =================================
     * CREATING A NR JOIN
     * =================================
     */
    auto nr_join_node =
        inner_join(nation, region_filter_node, {"n_regionkey"}, {"r_regionkey"}, "", "", device);

    /* =================================
     * CREATING A NR PROJECT
     * =================================
     */
    auto nr_project_node = project(nr_join_node, {"n_nationkey", "n_name"}, device);

    /* =================================
     * CREATING A PSPS-NR JOIN
     * =================================
     */
    auto psps_nr_join_node = inner_join(
        psps_project_node, nr_project_node, {"s_nationkey"}, {"n_nationkey"}, "", "", device);

    /* ====================================
     * CREATING A PROJECT FOR PSPS-NR JOIN
     * ====================================
     */
    auto psps_nr_project_node = project(psps_nr_join_node,
                                        {"p_partkey",
                                         "ps_supplycost",
                                         "p_mfgr",
                                         "n_name",
                                         "s_acctbal",
                                         "s_name",
                                         "s_address",
                                         "s_phone",
                                         "s_comment"},
                                        device);

    /* =================================
     * CREATING A GROUP BY NODE
     * =================================
     */
    auto aggs          = {aggregate("hash_min", "ps_supplycost", "min_supplycost")};
    auto group_by_node = group_by(psps_nr_project_node, {"p_partkey"}, aggs, device);

    /* =================================
     * CREATING A PSPSNR-AGGR JOIN
     * =================================
     */
    // here, psps_nr_project_node has two outputs: it's connected to group_by_node (previous) and pspsnr_aggr_join_node here
    auto pspsnr_aggr_join_node = inner_join(
        psps_nr_project_node,
        group_by_node,
        {"p_partkey", "ps_supplycost"},
        {"p_partkey", "min_supplycost"},
        "_l",
        "_r",
        device);  // here we add a suffix to the right keys, since arrow doesn't coalesce keys with the same name

    /* =================================
     * CREATING A PROJECT FOR PSPSNR-AGGR JOIN
     * =================================
     */
    auto pspsnr_aggr_project_node = project(pspsnr_aggr_join_node,
                                            exprs({"s_acctbal",
                                                   "s_name",
                                                   "n_name",
                                                   "p_partkey_l",
                                                   "p_mfgr",
                                                   "s_address",
                                                   "s_phone",
                                                   "s_comment"}),
                                            {"s_acctbal",
                                             "s_name",
                                             "n_name",
                                             "p_partkey",
                                             "p_mfgr",
                                             "s_address",
                                             "s_phone",
                                             "s_comment"},
                                            device);

    /* =================================
     * CREATING A ORDER BY NODE
     * =================================
     */
    std::vector<SortKey> sort_keys = {
        {"s_acctbal", SortOrder::DESCENDING}, {"n_name"}, {"s_name"}, {"p_partkey"}};
    auto order_by_node = order_by(pspsnr_aggr_project_node, sort_keys, device);

    /* =================================
     * CREATING A LIMIT NODE
     * =================================
     */
    auto limit_node = limit(order_by_node, 100, 0, device);

    return query_plan(table_sink(limit_node));
}

std::shared_ptr<QueryPlan> q3(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx = db->get_context();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    auto orders = table_source(db,
                               "orders",
                               schema("orders"),
                               {"o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"},
                               device);
    auto customer =
        table_source(db, "customer", schema("customer"), {"c_custkey", "c_mktsegment"}, device);
    auto lineitem = table_source(db,
                                 "lineitem",
                                 schema("lineitem"),
                                 {"l_orderkey", "l_shipdate", "l_extendedprice", "l_discount"},
                                 device);

    /* ==============================
     * CREATING A FILTER FOR ORDERS
     * ==============================
     */
    auto date_filter =
        expr(arrow_expr(cp::field_ref("o_orderdate"), "<", date_literal("1995-03-15")));
    auto filter_orders_node = filter(orders, date_filter, device);

    /* ================================
     * CREATING A FILTER FOR CUSTOMERS
     * ================================
     */
    auto segment_filter =
        expr(arrow_expr(cp::field_ref("c_mktsegment"), "==", string_literal("BUILDING")));
    auto filter_customers_node = filter(customer, segment_filter, device);

    /* ==========================================
     * CREATING A JOIN NODE (ORDERS, CUSTOMERS)
     * ==========================================
     */
    auto oc_join_node = inner_join(
        filter_orders_node, filter_customers_node, {"o_custkey"}, {"c_custkey"}, "", "", device);

    /* ==========================================================
     * CREATING A PROJECTION AFTER THE JOIN NODE (ORDERS, CUSTOMERS)
     * ==========================================================
     */
    auto oc_project_node =
        project(oc_join_node, {"o_orderkey", "o_orderdate", "o_shippriority"}, device);

    /* ================================
     * CREATING A FILTER FOR LINEITEM
     * ================================
     */
    auto lineitem_filter =
        expr(arrow_expr(cp::field_ref("l_shipdate"), ">", date_literal("1995-03-15")));
    auto filter_lineitem_node = filter(lineitem, lineitem_filter, device);

    /* ================================
     * CREATING A PROJECT FOR LINEITEM
     * ================================
     */
    auto lineitem_project_node =
        project(filter_lineitem_node, {"l_orderkey", "l_extendedprice", "l_discount"}, device);

    /* ===============================================================
     * CREATING A JOIN NODE (LINEITEM_PROJECT_NODE, OC_PROJECT_NODE)
     * ===============================================================
     */
    auto loc_join_node = inner_join(
        lineitem_project_node, oc_project_node, {"l_orderkey"}, {"o_orderkey"}, "", "", device);

    /* ===============================================================
     * LOC (LINEITEM x OC_JOIN) PROJECTION
     * ===============================================================
     */
    auto loc_project_exprs = exprs({"l_orderkey", "o_orderdate", "o_shippriority"});
    loc_project_exprs.push_back(
        expr(arrow_expr(cp::field_ref("l_extendedprice"),
                        "*",
                        arrow_expr(float64_literal(1.00), "-", cp::field_ref("l_discount")))));
    auto loc_project_node = project(loc_join_node,
                                    loc_project_exprs,
                                    {"l_orderkey", "o_orderdate", "o_shippriority", "volume"},
                                    device);

    /* ===============================================================
     * CREATING A GROUP BY NODE
     * ===============================================================
     */
    auto aggregates                     = {aggregate("hash_sum", "volume", "revenue")};
    std::vector<std::string> group_keys = {"l_orderkey", "o_orderdate", "o_shippriority"};
    auto group_by_node = group_by(loc_project_node, group_keys, aggregates, device);

    /* ===============================================================
     * CREATING AN ORDER BY NODE
     * ===============================================================
     */
    std::vector<SortKey> sort_keys = {{"revenue", SortOrder::DESCENDING}, {"o_orderdate"}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);

    /* ===============================================================
     * CREATING A LIMIT NODE
     * ===============================================================
     */
    auto limit_node = limit(order_by_node, 10, 0, device);

    return query_plan(table_sink(limit_node));
}

std::shared_ptr<QueryPlan> q4(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();

    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    auto lineitem = table_source(db,
                                 "lineitem",
                                 schema("lineitem"),
                                 {"l_orderkey", "l_commitdate", "l_receiptdate"},
                                 device);

    auto orders = table_source(
        db, "orders", schema("orders"), {"o_orderkey", "o_orderdate", "o_orderpriority"}, device);

    /* ==============================
     * CREATING A FILTER FOR LINEITEM
     * ==============================
     */
    auto lineitem_filter =
        expr(arrow_expr(cp::field_ref("l_commitdate"), "<", cp::field_ref("l_receiptdate")));
    auto filter_lineitem_node = filter(lineitem, lineitem_filter, device);

    /* ================================
     * CREATING A PROJECT FOR LINEITEM
     * ================================
     */
    auto lineitem_project_node = project(filter_lineitem_node, {"l_orderkey"}, device);

    /* ==============================
     * CREATING A FILTER FOR ORDERS
     * ==============================
     */
    auto order_date_filter  = expr(arrow_in_range(
        cp::field_ref("o_orderdate"), date_literal("1993-07-01"), date_literal("1993-10-01")));
    auto filter_orders_node = filter(orders, order_date_filter, device);

    /* ================================
     * CREATING A PROJECT FOR ORDERS
     * ================================
     */
    auto order_project_node =
        project(filter_orders_node, {"o_orderkey", "o_orderpriority"}, device);

    /* ==========================================
     * CREATING A JOIN NODE (LINEITEM, ORDERS)
     * ==========================================
     */
    auto lo_join = inner_join(
        lineitem_project_node, order_project_node, {"l_orderkey"}, {"o_orderkey"}, "", "", device);

    /* ==========================================
     * CREATING A LO-JOIN DISTINCT
     * ==========================================
     */
    auto lo_distinct = distinct(lo_join, {"l_orderkey", "o_orderkey", "o_orderpriority"}, device);

    /* ==========================================
     * CREATING A LO-DISTINCT PROJECT
     * ==========================================
     */
    auto lo_distinct_project = project(lo_distinct, {"o_orderpriority"}, device);

    /* ==========================================
     * CREATING A GROUP BY NODE
     * ==========================================
     */
    auto count_opts                                    = count_all();
    std::vector<std::shared_ptr<Aggregate>> aggregates = {
        aggregate("hash_count", count_opts, "o_orderpriority", "order_count")};
    auto group_by_node = group_by(lo_distinct_project, {"o_orderpriority"}, aggregates, device);

    /* ==========================================
     * CREATING AN ORDER BY NODE
     * ==========================================
     */
    auto order_by_node = order_by(group_by_node, {{"o_orderpriority"}}, device);

    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q5(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();

    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    auto nation = table_source(
        db, "nation", schema("nation"), {"n_nationkey", "n_regionkey", "n_name"}, device);
    auto region = table_source(db, "region", schema("region"), {"r_regionkey", "r_name"}, device);
    auto supplier =
        table_source(db, "supplier", schema("supplier"), {"s_suppkey", "s_nationkey"}, device);
    auto customer =
        table_source(db, "customer", schema("customer"), {"c_custkey", "c_nationkey"}, device);
    auto orders = table_source(
        db, "orders", schema("orders"), {"o_orderdate", "o_orderkey", "o_custkey"}, device);
    auto lineitem = table_source(db,
                                 "lineitem",
                                 schema("lineitem"),
                                 {"l_suppkey", "l_orderkey", "l_extendedprice", "l_discount"},
                                 device);

    /* ==============================
     * CREATING A FILTER FOR REGION
     * ==============================
     */
    auto region_filter = expr(arrow_expr(cp::field_ref("r_name"), "==", string_literal("ASIA")));
    auto filter_region_node = filter(region, region_filter, device);

    /* ==============================
     * CREATING A JOIN NODE (NATION, REGION)
     * ==============================
     */
    auto nr_join_node =
        inner_join(nation, filter_region_node, {"n_regionkey"}, {"r_regionkey"}, "", "", device);
    auto nr_project_node = project(nr_join_node, {"n_nationkey", "n_name"}, device);

    /* ==============================
     * CREATING A JOIN NODE (SUPPLIER, NR_JOIN)
     * ==============================
     */
    auto snr_join_node =
        inner_join(supplier, nr_project_node, {"s_nationkey"}, {"n_nationkey"}, "", "", device);
    auto snr_project_node = project(snr_join_node, {"s_suppkey", "s_nationkey", "n_name"}, device);

    /* ==============================
     * CREATING A JOIN NODE LSNR (LINEITEM, SNR)
     * ==============================
     */
    auto lsnr_join_node =
        inner_join(lineitem, snr_project_node, {"l_suppkey"}, {"s_suppkey"}, "", "", device);

    /* ==============================
     * CREATING A FILTER FOR ORDERS
     * ==============================
     */
    auto order_date_filter   = expr(arrow_in_range(
        cp::field_ref("o_orderdate"), date_literal("1994-01-01"), date_literal("1995-01-01")));
    auto filter_orders_node  = filter(orders, order_date_filter, device);
    auto project_orders_node = project(filter_orders_node, {"o_orderkey", "o_custkey"}, device);

    /* ==============================
     * CREATING A JOIN NODE OC (ORDERS, CUSTOMER)
     * ==============================
     */
    auto oc_join_node =
        inner_join(project_orders_node, customer, {"o_custkey"}, {"c_custkey"}, "", "", device);

    auto oc_project_node = project(oc_join_node, {"o_orderkey", "c_nationkey"}, device);

    /* ==============================
     * CREATING A JOIN NODE LSNROC (LSNR, OC)
     * ==============================
     */
    auto lsnroc_join_node = inner_join(lsnr_join_node,
                                       oc_project_node,
                                       {"l_orderkey", "s_nationkey"},
                                       {"o_orderkey", "c_nationkey"},
                                       "",
                                       "",
                                       device);

    /* ==============================
     * CREATING A PROJECT NODE
     * ==============================
     */
    std::vector<std::string> lsnroc_project_columns = {"l_extendedprice", "l_discount", "n_name"};
    auto lsnroc_project_expr                        = exprs(lsnroc_project_columns);
    lsnroc_project_expr.push_back(
        expr(arrow_expr(cp::field_ref("l_extendedprice"),
                        "*",
                        arrow_expr(float64_literal(1.00), "-", cp::field_ref("l_discount")))));
    lsnroc_project_columns.emplace_back("volume");

    auto lsnroc_project_node = project(lsnroc_join_node,
                                       std::move(lsnroc_project_expr),
                                       std::move(lsnroc_project_columns),
                                       device);

    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    auto group_by_node = group_by(
        lsnroc_project_node, {"n_name"}, {aggregate("hash_sum", "volume", "revenue")}, device);

    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    auto order_by_node = order_by(group_by_node, {{"revenue", SortOrder::DESCENDING}}, device);

    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q6(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();

    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    auto source_node = table_source(db,
                                    "lineitem",
                                    schema("lineitem"),
                                    {"l_shipdate", "l_extendedprice", "l_discount", "l_quantity"},
                                    device);

    // std::cout << source_node->get_input_schemas().size() << std::endl;
    // std::cout << source_node->get_input_schemas()[0]->get_schema()->ToString() << std::endl;

    /* ==============================
     * CREATING A FILTER NODE
     * ==============================
     */
    auto filter_expr = expr(
        arrow_all({arrow_in_range(cp::field_ref("l_shipdate"),
                                  date_literal("1994-01-01"),
                                  date_literal("1995-01-01")),  // left-inclusive, right-exclusive
                   arrow_between(cp::field_ref("l_discount"),
                                 float64_literal(0.05),
                                 float64_literal(0.07)),  // inclusive
                   arrow_expr(cp::field_ref("l_quantity"), "<", float64_literal(24.00))}));
    auto filter_node = filter(source_node, filter_expr, device);

    /* ==============================
     * CREATING A PROJECT NODE
     * ==============================
     */
    auto project_expr =
        expr(arrow_expr(cp::field_ref("l_extendedprice"), "*", cp::field_ref("l_discount")));
    auto project_node = project(filter_node, {project_expr}, {"product"}, device);

    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    auto sum_opts      = sum_ignore_nulls();
    auto agg           = {aggregate("sum", std::move(sum_opts), "product", "revenue")};
    auto group_by_node = group_by(project_node, {} /* empty group keys */, agg, device);

    return query_plan(table_sink(group_by_node));
}

std::shared_ptr<QueryPlan> q7(std::shared_ptr<Database>& db, DeviceType device) {
    auto supplier =
        table_source(db, "supplier", schema("supplier"), {"s_nationkey", "s_suppkey"}, device);

    auto nation = table_source(db, "nation", schema("nation"), {"n_nationkey", "n_name"}, device);

    auto customer =
        table_source(db, "customer", schema("customer"), {"c_custkey", "c_nationkey"}, device);

    auto orders = table_source(db, "orders", schema("orders"), {"o_custkey", "o_orderkey"}, device);

    auto lineitem =
        table_source(db,
                     "lineitem",
                     schema("lineitem"),
                     {"l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"},
                     device);

    auto n1_nation = rename(nation, {"n_nationkey", "n_name"}, {"n1_nationkey", "n1_name"}, device);

    std::vector<cp::Expression> target_nations = {string_literal("FRANCE"),
                                                  string_literal("GERMANY")};
    auto n1_nation_filter   = expr(arrow_in(cp::field_ref("n1_name"), target_nations));
    auto n1_nation_filtered = filter(n1_nation, n1_nation_filter, device);

    auto sn_join =
        inner_join(supplier, n1_nation_filtered, {"s_nationkey"}, {"n1_nationkey"}, "", "", device);

    auto sn = project(sn_join, {"s_suppkey", "n1_name"}, device);

    auto n2_nation_filtered = rename(
        n1_nation_filtered, {"n1_nationkey", "n1_name"}, {"n2_nationkey", "n2_name"}, device);

    auto cn_joined =
        inner_join(customer, n2_nation_filtered, {"c_nationkey"}, {"n2_nationkey"}, "", "", device);

    auto cn = project(cn_joined, {"c_custkey", "n2_name"}, device);

    auto cno_joined = inner_join(orders, cn, {"o_custkey"}, {"c_custkey"}, "", "", device);

    auto cno = project(cno_joined, {"o_orderkey", "n2_name"}, device);

    auto lineitem_filtered = filter(
        lineitem,
        expr(arrow_between(
            cp::field_ref("l_shipdate"), date_literal("1995-01-01"), date_literal("1996-12-31"))),
        device);

    auto cnol_joined =
        inner_join(lineitem_filtered, cno, {"l_orderkey"}, {"o_orderkey"}, "", "", device);

    auto cnol = project(cnol_joined,
                        {"l_suppkey", "l_shipdate", "l_extendedprice", "l_discount", "n2_name"},
                        device);

    auto all = inner_join(cnol, sn, {"l_suppkey"}, {"s_suppkey"}, "", "", device);

    auto name_filter = expr(arrow_expr(
        arrow_expr(arrow_expr(cp::field_ref("n1_name"), "==", string_literal("FRANCE")),
                   "&&",
                   arrow_expr(cp::field_ref("n2_name"), "==", string_literal("GERMANY"))),
        "||",
        arrow_expr(arrow_expr(cp::field_ref("n1_name"), "==", string_literal("GERMANY")),
                   "&&",
                   arrow_expr(cp::field_ref("n2_name"), "==", string_literal("FRANCE")))));

    auto all_filtered = filter(all, name_filter, device);

    auto volume_expr =
        expr(arrow_expr(cp::field_ref("l_extendedprice"),
                        "*",
                        arrow_expr(float64_literal(1.0), "-", cp::field_ref("l_discount"))));

    std::vector<std::shared_ptr<Expression>> all_exprs = {expr(cp::field_ref("n1_name")),
                                                          expr(cp::field_ref("n2_name")),
                                                          expr(year(cp::field_ref("l_shipdate"))),
                                                          volume_expr};
    std::vector<std::string> all_exprs_names = {"supp_nation", "cust_nation", "l_year", "volume"};

    auto all_project = project(all_filtered, all_exprs, all_exprs_names, device);

    auto all_grouped = group_by(all_project,
                                {"supp_nation", "cust_nation", "l_year"},
                                {aggregate("hash_sum", "volume", "revenue")},
                                device);

    auto all_ordered =
        order_by(all_grouped, {{"supp_nation"}, {"cust_nation"}, {"l_year"}}, device);

    return query_plan(table_sink(all_ordered));
}

std::shared_ptr<QueryPlan> q8(std::shared_ptr<Database>& db, DeviceType device) {
    auto region = table_source(db, "region", schema("region"), {"r_regionkey", "r_name"}, device);
    auto nation = table_source(
        db, "nation", schema("nation"), {"n_nationkey", "n_regionkey", "n_name"}, device);
    auto customer =
        table_source(db, "customer", schema("customer"), {"c_custkey", "c_nationkey"}, device);
    auto orders = table_source(
        db, "orders", schema("orders"), {"o_orderkey", "o_custkey", "o_orderdate"}, device);
    auto lineitem =
        table_source(db,
                     "lineitem",
                     schema("lineitem"),
                     {"l_orderkey", "l_partkey", "l_suppkey", "l_extendedprice", "l_discount"},
                     device);
    auto part = table_source(db, "part", schema("part"), {"p_partkey", "p_type"}, device);
    auto supplier =
        table_source(db, "supplier", schema("supplier"), {"s_suppkey", "s_nationkey"}, device);

    auto region_filtered = filter(
        region, expr(arrow_expr(cp::field_ref("r_name"), "==", string_literal("AFRICA"))), device);
    auto region_project = project(region_filtered, {"r_regionkey"}, device);

    auto nation_select = project(
        nation, exprs({"n_nationkey", "n_regionkey"}), {"n1_nationkey", "n1_regionkey"}, device);

    auto nr_joined = inner_join(
        region_project, nation_select, {"r_regionkey"}, {"n1_regionkey"}, "", "", device);

    auto nr = project(nr_joined, {"n1_nationkey"}, device);

    auto cnr_joined = inner_join(customer, nr, {"c_nationkey"}, {"n1_nationkey"}, "", "", device);

    auto cnr = project(cnr_joined, {"c_custkey"}, device);

    auto orders_filter = expr(arrow_between(
        cp::field_ref("o_orderdate"), date_literal("1995-01-01"), date_literal("1996-12-31")));

    auto orders_filtered = filter(orders, orders_filter, device);

    auto ocnr_joined =
        inner_join(orders_filtered, cnr, {"o_custkey"}, {"c_custkey"}, "", "", device);

    auto ocnr = project(ocnr_joined, {"o_orderkey", "o_orderdate"}, device);

    auto locnr_joined = inner_join(lineitem, ocnr, {"l_orderkey"}, {"o_orderkey"}, "", "", device);

    auto locnr = project(locnr_joined,
                         {"l_partkey", "l_suppkey", "l_extendedprice", "l_discount", "o_orderdate"},
                         device);

    auto part_filter =
        expr(arrow_expr(cp::field_ref("p_type"), "==", string_literal("ECONOMY ANODIZED STEEL")));
    auto part_filtered  = filter(part, part_filter, device);
    auto part_projected = project(part_filtered, {"p_partkey"}, device);

    auto locnrp_joined =
        inner_join(locnr, part_projected, {"l_partkey"}, {"p_partkey"}, "", "", device);
    auto locnrp = project(
        locnrp_joined, {"l_suppkey", "l_extendedprice", "l_discount", "o_orderdate"}, device);

    auto locnrps_joined =
        inner_join(locnrp, supplier, {"l_suppkey"}, {"s_suppkey"}, "", "", device);
    auto locnrps = project(
        locnrps_joined, {"l_extendedprice", "l_discount", "o_orderdate", "s_nationkey"}, device);

    auto n2_nation =
        project(nation, exprs({"n_nationkey", "n_name"}), {"n2_nationkey", "n2_name"}, device);

    auto all_joined =
        inner_join(locnrps, n2_nation, {"s_nationkey"}, {"n2_nationkey"}, "", "", device);

    /*
    auto all =
        project(all_joined, {"l_extendedprice", "l_discount", "o_orderdate", "n2_name"}, device);
    */

    auto volume_expr =
        arrow_expr(cp::field_ref("l_extendedprice"),
                   "*",
                   arrow_expr(float64_literal(1.0), "-", cp::field_ref("l_discount")));

    auto if_else_expr =
        arrow_if_else(arrow_expr(cp::field_ref("n2_name"), "==", string_literal("BRAZIL")),
                      volume_expr,
                      float64_literal(0.0));

    auto all_projected = project(all_joined,
                                 {expr(year(cp::field_ref("o_orderdate"))),
                                  expr(volume_expr),
                                  expr(cp::field_ref("n2_name")),
                                  expr(if_else_expr)},
                                 {"o_year", "volume", "nation", "if_else"},
                                 device);

    auto all_grouped = group_by(all_projected,
                                {"o_year"},
                                {aggregate("hash_sum", "if_else", "sum_if_else"),
                                 aggregate("hash_sum", "volume", "sum_volume")},
                                device);

    auto all_grouped_projected =
        project(all_grouped,
                {expr(cp::field_ref("o_year")),
                 expr(arrow_expr(cp::field_ref("sum_if_else"), "/", cp::field_ref("sum_volume")))},
                {"o_year", "mkt_share"},
                device);

    auto all_ordered = order_by(all_grouped_projected, {{"o_year"}}, device);

    return query_plan(table_sink(all_ordered));
}

std::shared_ptr<QueryPlan> q9(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx      = db->get_context();
    auto part     = table_source(db, "part", schema("part"), {"p_name", "p_partkey"}, device);
    auto partsupp = table_source(
        db, "partsupp", schema("partsupp"), {"ps_suppkey", "ps_partkey", "ps_supplycost"}, device);
    auto supplier =
        table_source(db, "supplier", schema("supplier"), {"s_suppkey", "s_nationkey"}, device);
    auto nation   = table_source(db, "nation", schema("nation"), {"n_nationkey", "n_name"}, device);
    auto lineitem = table_source(
        db,
        "lineitem",
        schema("lineitem"),
        {"l_suppkey", "l_partkey", "l_orderkey", "l_extendedprice", "l_discount", "l_quantity"},
        device);
    auto orders =
        table_source(db, "orders", schema("orders"), {"o_orderkey", "o_orderdate"}, device);

    // auto part_filtered = filter(part, expr(arrow_field_like("p_name", ".*green.*")), device);
    auto part_filtered = filter(part, expr(arrow_field_like("p_name", "%green%")), device);

    auto p = project(part_filtered, {"p_partkey"}, device);

    auto psp = inner_join(partsupp, p, {"ps_partkey"}, {"p_partkey"}, "", "", device);

    auto sn_joined = inner_join(supplier, nation, {"s_nationkey"}, {"n_nationkey"}, "", "", device);

    auto sn = project(sn_joined, {"s_suppkey", "n_name"}, device);

    auto pspsn = inner_join(psp, sn, {"ps_suppkey"}, {"s_suppkey"}, "", "", device);

    auto lpspsn_joined = inner_join(
        lineitem, pspsn, {"l_suppkey", "l_partkey"}, {"ps_suppkey", "ps_partkey"}, "", "", device);

    auto lpspsn = project(
        lpspsn_joined,
        {"l_orderkey", "l_extendedprice", "l_discount", "l_quantity", "ps_supplycost", "n_name"},
        device);

    auto all_joined = inner_join(orders, lpspsn, {"o_orderkey"}, {"l_orderkey"}, "", "", device);

    auto all = project(
        all_joined,
        {"l_extendedprice", "l_discount", "l_quantity", "ps_supplycost", "n_name", "o_orderdate"},
        device);

    // amount = l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity)
    auto amount_expr = expr(
        arrow_expr(arrow_expr(cp::field_ref("l_extendedprice"),
                              "*",
                              arrow_expr(float64_literal(1.0), "-", cp::field_ref("l_discount"))),
                   "-",
                   arrow_expr(cp::field_ref("ps_supplycost"), "*", cp::field_ref("l_quantity"))));

    std::vector<std::shared_ptr<Expression>> all_expressions = {
        expr(cp::field_ref("n_name")), expr(year(cp::field_ref("o_orderdate"))), amount_expr};

    std::vector<std::string> all_exprs_names = {"nation", "o_year", "amount"};

    auto all_projected = project(all, all_expressions, all_exprs_names, device);

    auto all_grouped = group_by(all_projected,
                                {"nation", "o_year"},
                                {aggregate("hash_sum", "amount", "sum_profit")},
                                device);

    auto all_ordered =
        order_by(all_grouped, {{"nation"}, {"o_year", SortOrder::DESCENDING}}, device);

    return query_plan(table_sink(all_ordered));
}

std::shared_ptr<QueryPlan> q10(std::shared_ptr<Database>& db, DeviceType device) {
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    auto lineitem = table_source(db,
                                 "lineitem",
                                 schema("lineitem"),
                                 {"l_orderkey", "l_returnflag", "l_extendedprice", "l_discount"},
                                 device);

    auto orders = table_source(
        db, "orders", schema("orders"), {"o_orderkey", "o_custkey", "o_orderdate"}, device);

    auto customer = table_source(
        db,
        "customer",
        schema("customer"),
        {"c_custkey", "c_nationkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment"},
        device);

    auto nation = table_source(db, "nation", schema("nation"), {"n_nationkey", "n_name"}, device);

    /* ==============================
     * CREATING A LINEITEM FILTER
     * ==============================
     */
    auto lineitem_filter_expr =
        expr(arrow_expr(cp::field_ref("l_returnflag"), "==", string_literal("R")));
    auto filter_lineitem_node = filter(lineitem, lineitem_filter_expr, device);

    /* ==============================
     * CREATING A LINEITEM PROJECT
     * ==============================
     */
    auto project_lineitem_node =
        project(filter_lineitem_node, {"l_orderkey", "l_extendedprice", "l_discount"}, device);

    /* ==============================
     * CREATING AN ORDERS FILTER
     * ==============================
     */
    auto order_date_filter  = expr(arrow_in_range(
        cp::field_ref("o_orderdate"), date_literal("1993-10-01"), date_literal("1994-01-01")));
    auto filter_orders_node = filter(orders, order_date_filter, device);

    /* ==============================
     * CREATING AN ORDERS PROJECT
     * ==============================
     */
    auto project_orders_node = project(filter_orders_node, {"o_orderkey", "o_custkey"}, device);

    /* =========================================
     * CREATING A JOIN NODE (LINEITEM, ORDERS)
     * =========================================
     */
    auto lo_join_node = inner_join(
        project_lineitem_node, project_orders_node, {"l_orderkey"}, {"o_orderkey"}, "", "", device);

    /* =========================================
     * CREATING A LO-JOIN PROJECTION
     * =========================================
     */
    std::vector<std::string> lo_project_columns = {"l_extendedprice", "l_discount", "o_custkey"};
    auto lo_project_expr                        = exprs(lo_project_columns);
    // add a new column "volume" to the expressions and the column names
    lo_project_expr.push_back(
        expr(arrow_expr(cp::field_ref("l_extendedprice"),
                        "*",
                        arrow_expr(float64_literal(1.00), "-", cp::field_ref("l_discount")))));
    lo_project_columns.emplace_back("volume");
    auto project_lo_node = project(lo_join_node, lo_project_expr, lo_project_columns, device);

    /* =========================================
     * CREATING A GROUP-BY FOR LO-JOIN PROJECT
     * =========================================
     */
    std::vector<std::shared_ptr<Aggregate>> aggregates = {
        aggregate("hash_sum", "volume", "revenue")};
    auto lo_group_by_node = group_by(project_lo_node, {"o_custkey"}, aggregates, device);

    /* =========================================
     * CREATING A LOC-JOIN NODE
     * =========================================
     */
    auto loc_join_node =
        inner_join(customer, lo_group_by_node, {"c_custkey"}, {"o_custkey"}, "", "", device);

    /* =========================================
     * CREATING A LOCN-JOIN
     * =========================================
     */
    auto locn_join_node =
        inner_join(loc_join_node, nation, {"c_nationkey"}, {"n_nationkey"}, "", "", device);

    /* =========================================
     * CREATING A LOCN-PROJECT
     * =========================================
     */
    auto project_locn_node = project(locn_join_node,
                                     {"c_custkey",
                                      "c_name",
                                      "revenue",
                                      "c_acctbal",
                                      "n_name",
                                      "c_address",
                                      "c_phone",
                                      "c_comment"},
                                     device);

    /* =========================================
     * CREATING AN ORDER-BY NODE
     * =========================================
     */
    auto order_by_node = order_by(project_locn_node, {{"revenue", SortOrder::DESCENDING}}, device);

    /* =========================================
     * CREATING A LIMIT NODE
     * =========================================
     */
    auto limit_node = limit(order_by_node, 20, 0, device);

    return query_plan(table_sink(limit_node));
}

std::shared_ptr<QueryPlan> q11(std::shared_ptr<Database>& db, DeviceType device) {
    /* ==============================
     * CREATING TABLE SOURCE NODES
     * ==============================
     */
    auto nation = table_source(db, "nation", schema("nation"), {}, device);

    auto partsupp = table_source(db, "partsupp", schema("partsupp"), {}, device);

    auto supplier = table_source(db, "supplier", schema("supplier"), {}, device);

    /* ==============================
     * CREATING A FILTER FOR NATION
     * ==============================
     */
    auto nation_filter = expr(arrow_expr(cp::field_ref("n_name"), "==", string_literal("GERMANY")));
    auto filter_nation_node = filter(nation, nation_filter, device);

    /* ===========================================
     * CREATING A JOIN NODE (PARTSUPP, SUPPLIER)
     * ===========================================
     */
    auto supplier_join_node =
        inner_join(partsupp, supplier, {"ps_suppkey"}, {"s_suppkey"}, "", "", device);

    /* ===========================================
     * CREATING A JOIN NODE (SUPPLIER_JOIN, NATION)
     * ===========================================
     */
    auto joined_filtered = inner_join(
        supplier_join_node, filter_nation_node, {"s_nationkey"}, {"n_nationkey"}, "", "", device);

    /* ===========================================
     * COMPUTING SUPPLY_COST * AVAIL_QTY
     * ===========================================
    */
    // in order to compute sum(ps_supplycost * ps_availqty) * 0.0001000000,
    // we first compute value = ps_supplycost * ps_availqty:
    auto cost_times_qty =
        expr(arrow_product({cp::field_ref("ps_supplycost"), cp::field_ref("ps_availqty")}));
    auto project_node = project(joined_filtered,
                                {expr(cp::field_ref("ps_partkey")), std::move(cost_times_qty)},
                                {"ps_partkey", "value"},
                                device);

    /* ===========================================
     * ADDING A GLOBAL AGGR NODE
     * ===========================================
    */
    // now we compute global_value = sum(value) = sum(ps_supplycost * ps_availqty)
    std::vector<std::shared_ptr<Aggregate>> aggregates = {
        aggregate("sum", "value", "global_value")};
    auto global_aggr_group_by = group_by(project_node, {}, aggregates, device);

    // add another projection to remove all columns except the aggregated value
    auto global_value_project_expr =
        expr(arrow_product({cp::field_ref("global_value"), float64_literal(0.0001)}));
    auto global_aggr =
        project(global_aggr_group_by, {global_value_project_expr}, {"global_value"}, device);

    /* ===========================================
     * CREATING A PS_PARTKEY GROUP BY NODE
     * ===========================================
     */
    auto partsupp_aggregate = aggregate("hash_sum", "value", "value");
    auto partkey_aggr       = group_by(project_node, {"ps_partkey"}, {partsupp_aggregate}, device);

    /* ===========================================
     * CREATING A CROSS JOIN NODE
     * ===========================================
     */
    auto cross_join_node = cross_join(
        partkey_aggr, global_aggr, {"value", "ps_partkey"}, {"global_value"}, "", "", device);

    /* ===========================================
     * CREATING A FILTER FOR CROSS JOIN
     * ===========================================
     */
    auto greater_than_global =
        expr(arrow_expr(cp::field_ref("value"), ">", cp::field_ref("global_value")));
    auto filter_node = filter(cross_join_node, greater_than_global, device);

    /* ===========================================
     * CREATING AN ORDER-BY NODE
     * ===========================================
     */
    auto order_by_node = order_by(filter_node, {{"value", SortOrder::DESCENDING}}, device);

    /* ===========================================
     * CREATING A FINAL PROJECT NODE (SELECT)
     * ===========================================
     */
    auto select_node = project(order_by_node, {"ps_partkey", "value"}, device);

    return query_plan(table_sink(select_node));
}

std::shared_ptr<QueryPlan> q12(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();

    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    auto lineitem = table_source(db, "lineitem", schema("lineitem"), {}, device);
    auto orders   = table_source(db, "orders", schema("orders"), {}, device);

    /* ==============================
     * CREATING A FILTER FOR LINEITEM
     * ==============================
     */
    auto lineiterm_filter     = expr(arrow_all(
        {arrow_in(cp::field_ref("l_shipmode"), {string_literal("MAIL"), string_literal("SHIP")}),
             arrow_in_range(
             cp::field_ref("l_commitdate"), date_literal("1994-01-01"), date_literal("1995-01-01")),
             arrow_expr(cp::field_ref("l_commitdate"), "<", cp::field_ref("l_receiptdate")),
             arrow_expr(cp::field_ref("l_shipdate"), "<", cp::field_ref("l_commitdate"))}));
    auto filter_lineitem_node = filter(lineitem, lineiterm_filter, device);

    /* ==============================
     * CREATING A LO-JOIN
     * ==============================
     */
    auto lo_join_node =
        inner_join(filter_lineitem_node, orders, {"l_orderkey"}, {"o_orderkey"}, "", "", device);

    /* ==============================
     * CREATING A LO-JOIN PROJECT
     * ==============================
     */

    auto if_else_in =
        expr(arrow_if_else(arrow_in(cp::field_ref("o_orderpriority"),
                                    {string_literal("1-URGENT"), string_literal("2-HIGH")}),
                           int64_literal(1),
                           int64_literal(0)));

    auto if_else_not_in =
        expr(arrow_if_else(arrow_not_in(cp::field_ref("o_orderpriority"),
                                        {string_literal("1-URGENT"), string_literal("2-HIGH")}),
                           int64_literal(1),
                           int64_literal(0)));
    std::vector<std::string> lo_project_names = {"l_shipmode", "high_line_count", "low_line_count"};
    auto lo_project_expr                      = exprs({"l_shipmode"});
    lo_project_expr.push_back(std::move(if_else_in));
    lo_project_expr.push_back(std::move(if_else_not_in));
    auto lo_project_node = project(lo_join_node, lo_project_expr, lo_project_names, device);

    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggregates = {
        aggregate("hash_sum", "high_line_count", "high_line_count"),
        aggregate("hash_sum", "low_line_count", "low_line_count")};
    auto lo_group_by_node = group_by(lo_project_node, {"l_shipmode"}, aggregates, device);

    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    auto order_by_node = order_by(lo_group_by_node, {{"l_shipmode"}}, device);

    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q13(std::shared_ptr<Database>& db, DeviceType device) {
    auto customer = table_source(db, "customer", schema("customer"), {}, device);
    auto orders   = table_source(db, "orders", schema("orders"), {}, device);

    // auto special_orders_filter = expr(arrow_field_like("o_comment", ".*special.*requests.*"));
    auto special_orders_filter =
        expr(arrow_not(arrow_field_like("o_comment", "%special%requests%")));

    auto special_orders = filter(orders, special_orders_filter, device);

    // here we use outer join (instead of left_anti_join) since we also want to include the columns from orders
    // e.g. o_orderkey is later used in group_by
    auto co_joined =
        left_outer_join(customer, special_orders, {"c_custkey"}, {"o_custkey"}, "", "", device);

    auto aggregate_opts = count_valid();
    auto c_orders       = group_by(co_joined,
                                   {"c_custkey"},
                                   {aggregate("hash_count", count_valid(), "o_orderkey", "c_count")},
                             device);

    auto c_orders_grouped =
        group_by(c_orders,
                 {"c_count"},
                 {aggregate("hash_count", count_all(), "c_custkey", "custdist")},
                 device);

    auto c_orders_sorted =
        order_by(c_orders_grouped,
                 {{"custdist", SortOrder::DESCENDING}, {"c_count", SortOrder::DESCENDING}},
                 device);

    return query_plan(table_sink(c_orders_sorted));
}

std::shared_ptr<QueryPlan> q14(std::shared_ptr<Database>& db, DeviceType device) {
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    auto lineitem = table_source(db, "lineitem", schema("lineitem"), {}, device);
    auto part     = table_source(db, "part", schema("part"), {}, device);

    /* ==============================
     * CREATING A FILTER FOR LINEITEM
     * ==============================
     */
    auto filter_expr          = expr(arrow_in_range(
        cp::field_ref("l_shipdate"), date_literal("1995-09-01"), date_literal("1995-10-01")));
    auto filter_lineitem_node = filter(lineitem, filter_expr, device);

    /* ==============================
     * CREATING AN INNER JOIN
     * ==============================
     */
    auto join_node =
        inner_join(filter_lineitem_node, part, {"l_partkey"}, {"p_partkey"}, "", "", device);

    /* ==============================
     * CREATING A FILTER FOR PART
     * ==============================
     */
    auto _starts_with_promo = arrow_field_starts_with("p_type", "PROMO");

    // promo revenue calculation
    auto _promo_revenue =
        arrow_expr(cp::field_ref("l_extendedprice"),
                   "*",
                   arrow_expr(float64_literal(1.0), "-", cp::field_ref("l_discount")));
    auto promo_revenue = expr(_promo_revenue);

    auto if_else_expr =
        expr(arrow_if_else(_starts_with_promo, _promo_revenue, float64_literal(0.0)));

    // Create a projection node for promo revenue calculation
    std::vector<std::shared_ptr<Expression>> promo_expressions = {if_else_expr, promo_revenue};
    std::vector<std::string> promo_column_names                = {"if_else", "promo_revenue"};
    auto promo_project_node = project(join_node, promo_expressions, promo_column_names, device);

    /* ==============================
     * ADDING GROUP BY (SUM)
     * ==============================
     */
    auto sum_opts                                      = sum_ignore_nulls();
    std::vector<std::shared_ptr<Aggregate>> aggregates = {
        aggregate("sum", sum_opts, "if_else", "if_else"),
        aggregate("sum", sum_opts, "promo_revenue", "promo_revenue")};
    auto group_by_node = group_by(promo_project_node, {}, aggregates, device);

    /* ==============================
     * ADDING A FINAL PROJECTION
     * ==============================
     */
    // here we want to compute 100*if_else/promo_revenue
    auto final_expr =
        expr(arrow_expr(arrow_expr(cp::field_ref("if_else"), "*", float64_literal(100.0)),
                        "/",
                        cp::field_ref("promo_revenue")));
    auto final_project_node =
        project(group_by_node, {std::move(final_expr)}, {"promo_revenue"}, device);

    return query_plan(table_sink(final_project_node));
}

std::shared_ptr<QueryPlan> q15(std::shared_ptr<Database>& db, DeviceType device) {
    auto lineitem = table_source(db, "lineitem", schema("lineitem"), {}, device);
    auto supplier = table_source(db, "supplier", schema("supplier"), {}, device);

    auto lineitem_filter   = expr(arrow_in_range(
        cp::field_ref("l_shipdate"), date_literal("1996-01-01"), date_literal("1996-04-01")));
    auto filtered_lineitem = filter(lineitem, lineitem_filter, device);

    auto lineitem_exprs = exprs({"l_suppkey"});
    lineitem_exprs.push_back(
        expr(arrow_expr(cp::field_ref("l_extendedprice"),
                        "*",
                        arrow_expr(float64_literal(1.0), "-", cp::field_ref("l_discount")))));
    auto lineitem_project =
        project(filtered_lineitem, lineitem_exprs, {"l_suppkey", "revenue"}, device);

    auto revenue_by_supplier = group_by(lineitem_project,
                                        {"l_suppkey"},
                                        {aggregate("hash_sum", "revenue", "total_revenue")},
                                        device);

    auto global_revenue = group_by(
        revenue_by_supplier, {}, {aggregate("max", "total_revenue", "max_total_revenue")}, device);

    auto joined = cross_join(revenue_by_supplier,
                             global_revenue,
                             {"l_suppkey", "total_revenue"},
                             {"max_total_revenue"},
                             "",
                             "",
                             device);

    auto revenue_filter = expr(
        arrow_expr(arrow_abs(arrow_expr(
                       cp::field_ref("total_revenue"), "-", cp::field_ref("max_total_revenue"))),
                   "<",
                   float64_literal(1e-9)));

    auto high_revenue = filter(joined, revenue_filter, device);

    auto revenue_supplier =
        inner_join(high_revenue, supplier, {"l_suppkey"}, {"s_suppkey"}, "", "", device);

    auto project_expressions =
        exprs({"l_suppkey", "s_name", "s_address", "s_phone", "total_revenue"});
    std::vector<std::string> project_names = {
        "s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"};

    auto project_node = project(revenue_supplier, project_expressions, project_names, device);

    return query_plan(table_sink(project_node));
}

std::shared_ptr<QueryPlan> q16(std::shared_ptr<Database>& db, DeviceType device) {
    auto part     = table_source(db, "part", schema("part"), {}, device);
    auto supplier = table_source(db, "supplier", schema("supplier"), {}, device);
    auto partsupp = table_source(db, "partsupp", schema("partsupp"), {}, device);

    auto part_filter =
        expr(arrow_all({arrow_expr(cp::field_ref("p_brand"), "!=", string_literal("Brand#45")),
                        arrow_not(arrow_field_starts_with("p_type", "MEDIUM POLISHED")),
                        arrow_in(cp::field_ref("p_size"),
                                 {int32_literal(49),
                                  int32_literal(14),
                                  int32_literal(23),
                                  int32_literal(45),
                                  int32_literal(19),
                                  int32_literal(3),
                                  int32_literal(36),
                                  int32_literal(9)})}));

    auto part_filtered = filter(part, part_filter, device);

    // auto supplier_filter =
    // expr(arrow_not(arrow_field_like("s_comment", ".*Customer.*Complaints.*")));
    auto supplier_filter = expr(arrow_not(arrow_field_like("s_comment", "%Customer%Complaints%")));
    auto supplier_filtered = filter(supplier, supplier_filter, device);

    auto partsupp_filtered_full =
        inner_join(partsupp, supplier_filtered, {"ps_suppkey"}, {"s_suppkey"}, "", "", device);

    auto partsupp_filtered = project(partsupp_filtered_full, {"ps_partkey", "ps_suppkey"}, device);

    auto joined =
        inner_join(part_filtered, partsupp_filtered, {"p_partkey"}, {"ps_partkey"}, "", "", device);

    auto grouped = group_by(joined,
                            {"p_brand", "p_type", "p_size"},
                            {aggregate("hash_count_distinct", "ps_suppkey", "supplier_cnt")},
                            device);

    auto selected = project(grouped, {"p_brand", "p_type", "p_size", "supplier_cnt"}, device);

    auto ordered =
        order_by(selected,
                 {{"supplier_cnt", SortOrder::DESCENDING}, {"p_brand"}, {"p_type"}, {"p_size"}},
                 device);

    return query_plan(table_sink(ordered));
}

std::shared_ptr<QueryPlan> q17(std::shared_ptr<Database>& db, DeviceType device) {
    auto part     = table_source(db, "part", schema("part"), {}, device);
    auto lineitem = table_source(db, "lineitem", schema("lineitem"), {}, device);

    auto parts_filter =
        expr(arrow_expr(arrow_expr(cp::field_ref("p_brand"), "==", string_literal("Brand#23")),
                        "&&",
                        arrow_expr(cp::field_ref("p_container"), "==", string_literal("MED BOX"))));

    auto parts_filtered = filter(part, parts_filter, device);

    auto joined =
        inner_join(lineitem, parts_filtered, {"l_partkey"}, {"p_partkey"}, "", "", device);

    auto quantity_by_part_grouped =
        group_by(joined,
                 {"l_partkey"},
                 {aggregate("hash_mean", "l_quantity", "quantity_threshold")},
                 device);

    // multiply the quantity threshold by 0.2
    auto quantity_by_part =
        project(quantity_by_part_grouped,
                {Expression::from_field_ref("l_partkey"),
                 expr(arrow_expr(cp::field_ref("quantity_threshold"), "*", float64_literal(0.2)))},
                {"l_partkey", "quantity_threshold"},
                device);

    auto joined2 =
        inner_join(joined,
                   quantity_by_part,
                   {"l_partkey"},
                   {"l_partkey"},
                   "",
                   "_r",  // here we add suffixes to the columns because the keys have the same name
                          // both cudf and acero can have problems with it
                   device);

    // now we revert the names back to the original ones (without suffixes)
    // the resulting schema will have the same names as below, with the addition of "l_partkey_r", which we omit
    // these column names have no suffixes nor repeated l_partkey column
    /*
    std::vector<std::string> column_names = {
        "l_orderkey", "l_partkey", "l_suppkey", "l_linenumber", "l_quantity", "l_extendedprice",
        "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipdate", "l_commitdate",
        "l_receiptdate", "l_shipinstruct","l_shipmode","l_comment","p_partkey","p_name","p_mfgr","p_brand","p_type",
        "p_size","p_container","p_retailprice","p_comment","quantity_threshold"
    };

    auto joined2_projected = project(joined2, column_names, device);
    */

    auto joined2_filtered = filter(
        joined2,
        expr(arrow_expr(cp::field_ref("l_quantity"), "<", cp::field_ref("quantity_threshold"))),
        device);

    auto final_group_by =
        group_by(joined2_filtered, {}, {aggregate("sum", "l_extendedprice", "avg_yearly")}, device);

    auto final_project =
        project(final_group_by,
                {expr(arrow_expr(cp::field_ref("avg_yearly"), "/", float64_literal(7.0)))},
                {"avg_yearly"},
                device);

    return query_plan(table_sink(final_project));
}

std::shared_ptr<QueryPlan> q18(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx = db->get_context();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    auto lineitem = table_source(db, "lineitem", schema("lineitem"), {}, device);
    auto orders   = table_source(db, "orders", schema("orders"), {}, device);
    auto customer = table_source(db, "customer", schema("customer"), {}, device);

    /* =================================
     * CREATING A GROUP-BY FOR LINEITEM
     * =================================
     */
    std::vector<std::shared_ptr<Aggregate>> aggregates = {
        aggregate("hash_sum", "l_quantity", "sum(l_quantity)"),
    };
    std::vector<arrow::FieldRef> group_keys = {arrow::FieldRef("l_orderkey")};
    auto group_by_node = group_by(lineitem, {"l_orderkey"}, aggregates, device);

    /* ==============================
     * CREATING A FILTER NODE
     * ==============================
     */
    auto filter_expr =
        expr(arrow_expr(cp::field_ref("sum(l_quantity)"), ">", float64_literal(300.0)));
    auto filter_node = filter(group_by_node, filter_expr, device);

    /* ==============================
     * CREATING A JOIN NODE
     * ==============================
     */
    auto ol_join_node =
        inner_join(orders, filter_node, {"o_orderkey"}, {"l_orderkey"}, "", "", device);

    /* ==============================
     * CREATING ANOTHER JOIN NODE
     * ==============================
     */
    auto loc_join_node =
        inner_join(ol_join_node, customer, {"o_custkey"}, {"c_custkey"}, "", "", device);

    /* ==============================
     * CREATING A PROJECT NODE
     * ==============================
     */
    std::vector<std::string> project_columns = {
        "c_name", "o_custkey", "o_orderkey", "o_orderdate", "o_totalprice", "sum(l_quantity)"};
    auto project_expr = exprs(
        {"c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice", "sum(l_quantity)"});
    auto project_node = project(loc_join_node, project_expr, project_columns, device);

    /* ==============================
     * CREATING AN ORDER-BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"o_totalprice", SortOrder::DESCENDING},
                                      {"o_orderdate", SortOrder::ASCENDING}};
    auto order_by_node             = order_by(project_node, sort_keys, device);

    return query_plan(table_sink(limit(order_by_node, 100, 0, device)));
}

std::shared_ptr<QueryPlan> q19(std::shared_ptr<Database>& db, DeviceType device) {
    auto lineitem = table_source(db, "lineitem", schema("lineitem"), {}, device);
    auto part     = table_source(db, "part", schema("part"), {}, device);

    auto joined = inner_join(lineitem, part, {"l_partkey"}, {"p_partkey"}, "", "", device);

    auto filter_expr = expr(arrow_any(
        {arrow_all(
             {arrow_expr(cp::field_ref("p_brand"), "==", string_literal("Brand#12")),
              arrow_in(cp::field_ref("p_container"),
                       {string_literal("SM CASE"),
                        string_literal("SM BOX"),
                        string_literal("SM PACK"),
                        string_literal("SM PKG")}),
              arrow_between(cp::field_ref("l_quantity"), float64_literal(1), float64_literal(11)),
              arrow_between(cp::field_ref("p_size"), int32_literal(1), int32_literal(5)),
              arrow_in(cp::field_ref("l_shipmode"),
                       {string_literal("AIR"), string_literal("AIR REG")}),
              arrow_expr(
                  cp::field_ref("l_shipinstruct"), "==", string_literal("DELIVER IN PERSON"))}),
         arrow_all(
             {arrow_expr(cp::field_ref("p_brand"), "==", string_literal("Brand#23")),
              arrow_in(cp::field_ref("p_container"),
                       {string_literal("MED BAG"),
                        string_literal("MED BOX"),
                        string_literal("MED PKG"),
                        string_literal("MED PACK")}),
              arrow_between(cp::field_ref("l_quantity"), float64_literal(10), float64_literal(20)),
              arrow_between(cp::field_ref("p_size"), int32_literal(1), int32_literal(10)),
              arrow_in(cp::field_ref("l_shipmode"),
                       {string_literal("AIR"), string_literal("AIR REG")}),
              arrow_expr(
                  cp::field_ref("l_shipinstruct"), "==", string_literal("DELIVER IN PERSON"))}),
         arrow_all(
             {arrow_expr(cp::field_ref("p_brand"), "==", string_literal("Brand#34")),
              arrow_in(cp::field_ref("p_container"),
                       {string_literal("LG CASE"),
                        string_literal("LG BOX"),
                        string_literal("LG PACK"),
                        string_literal("LG PKG")}),
              arrow_between(cp::field_ref("l_quantity"), float64_literal(20), float64_literal(30)),
              arrow_between(cp::field_ref("p_size"), int32_literal(1), int32_literal(15)),
              arrow_in(cp::field_ref("l_shipmode"),
                       {string_literal("AIR"), string_literal("AIR REG")}),
              arrow_expr(
                  cp::field_ref("l_shipinstruct"), "==", string_literal("DELIVER IN PERSON"))})}));

    auto filter_node = filter(joined, filter_expr, device);

    auto project_expr =
        expr(arrow_expr(cp::field_ref("l_extendedprice"),
                        "*",
                        arrow_expr(float64_literal(1.0), "-", cp::field_ref("l_discount"))));

    auto project_node = project(filter_node, {std::move(project_expr)}, {"revenue"}, device);

    auto group_by_node =
        group_by(project_node, {}, {aggregate("sum", "revenue", "revenue")}, device);

    return query_plan(table_sink(group_by_node));
}

std::shared_ptr<QueryPlan> q20(std::shared_ptr<Database>& db, DeviceType device) {
    /* ================================
     * CREATING TABLE SOURCE NODES
     * ================================
     */
    auto supplier = table_source(db, "supplier", schema("supplier"), {}, device);
    auto nation   = table_source(db, "nation", schema("nation"), {}, device);
    auto part     = table_source(db, "part", schema("part"), {}, device);
    auto partsupp = table_source(db, "partsupp", schema("partsupp"), {}, device);
    auto lineitem = table_source(db, "lineitem", schema("lineitem"), {}, device);

    /* ================================
     * CREATING A FILTER FOR NATION
     * ================================
     */
    auto nation_filter = expr(arrow_expr(cp::field_ref("n_name"), "==", string_literal("CANADA")));
    auto filter_nation_node = filter(nation, nation_filter, device);

    /* ================================
     * CREATING A JOIN NODE (SUPPLIER, NATION)
     * ================================
     */
    auto sn_join_node =
        inner_join(supplier, filter_nation_node, {"s_nationkey"}, {"n_nationkey"}, "", "", device);
    auto supplier_ca = project(sn_join_node, {"s_suppkey", "s_name", "s_address"}, device);

    /* ================================
     * CREATING A FILTER FOR PART
     * ================================
     */
    auto part_filter = expr(arrow_field_starts_with("p_name", "forest"));
    auto part_forest = filter(part, part_filter, device);

    /* ================================
     * CREATING JOIN NODES (SN, PART)
     * ================================
     */
    auto snp_join_node =
        left_semi_join(partsupp, supplier_ca, {"ps_suppkey"}, {"s_suppkey"}, "", "", device);

    auto partsupp_forest_ca =
        left_semi_join(snp_join_node, part_forest, {"ps_partkey"}, {"p_partkey"}, "", "", device);

    /* ================================
     * CREATING A LINEITEM FILTER
     * ================================
     */
    auto lineitem_filter      = expr(arrow_in_range(
        cp::field_ref("l_shipdate"), date_literal("1994-01-01"), date_literal("1995-01-01")));
    auto filter_lineitem_node = filter(lineitem, lineitem_filter, device);

    /* ================================
     * CREATING A JOIN NODE (PARTSUPP, LINEITEM)
     * ================================
     */
    auto lpartsupp_forest_ca = left_semi_join(filter_lineitem_node,
                                              partsupp_forest_ca,
                                              {"l_partkey", "l_suppkey"},
                                              {"ps_partkey", "ps_suppkey"},
                                              "",
                                              "",
                                              device);

    /* ================================
     * CREATING A GROUP BY NODE
     * ================================
     */
    auto sum_opts      = sum_ignore_nulls();
    auto aggregates    = {aggregate("hash_sum", "l_quantity", "sum_quantity")};
    auto group_by_node = group_by(lpartsupp_forest_ca, {"l_suppkey"}, aggregates, device);

    /* ================================
     * CREATING A PROJECT NODE
     * ================================
     */
    auto half_sum_quantity =
        expr(arrow_expr(cp::field_ref("sum_quantity"), "*", float64_literal(0.5)));
    auto qty_threshold = project(group_by_node,
                                 {expr(cp::field_ref("l_suppkey")), std::move(half_sum_quantity)},
                                 {"l_suppkey", "qty_threshold"},
                                 device);

    /* ================================
     * CREATING AN INNER-JOIN
     * ================================
     */
    auto partsupp_forest_ca_joined = inner_join(
        partsupp_forest_ca, qty_threshold, {"ps_suppkey"}, {"l_suppkey"}, "", "", device);

    /* ================================
     * CREATING A FILTER NODE
     * ================================
     */
    auto thr_expr =
        expr(arrow_expr(cp::field_ref("ps_availqty"), ">", cp::field_ref("qty_threshold")));
    auto partsupp_forest_ca_filtered = filter(partsupp_forest_ca_joined, thr_expr, device);

    /* ================================
     * CREATING A SEMI-JOIN NODE
     * ================================
     */
    auto supplier_ca_partsupp_join = left_semi_join(
        supplier_ca, partsupp_forest_ca_filtered, {"s_suppkey"}, {"ps_suppkey"}, "", "", device);

    /* ================================
     * CREATING A PROJECT NODE
     * ================================
     */
    auto project_supplier_ca_partsupp =
        project(supplier_ca_partsupp_join, {"s_name", "s_address"}, device);

    /* ================================
     * CREATING AN ORDER BY NODE
     * ================================
     */
    auto order_by_node = order_by(project_supplier_ca_partsupp, {{"s_name"}}, device);

    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q21(std::shared_ptr<Database>& db, DeviceType device) {
    auto lineitem = table_source(db, "lineitem", schema("lineitem"), {}, device);
    auto orders   = table_source(db, "orders", schema("orders"), {}, device);
    auto supplier = table_source(db, "supplier", schema("supplier"), {}, device);
    auto nation   = table_source(db, "nation", schema("nation"), {}, device);

    auto num_suppliers = group_by(lineitem,
                                  {"l_orderkey"},
                                  {aggregate("hash_count_distinct", "l_suppkey", "n_supplier")},
                                  device);

    auto orders_with_more_than_one_supplier =
        filter(num_suppliers,
               expr(arrow_expr(cp::field_ref("n_supplier"), ">", int32_literal(1))),
               device);

    auto semi_joined = left_semi_join(lineitem,
                                      orders_with_more_than_one_supplier,
                                      {"l_orderkey"},
                                      {"l_orderkey"},
                                      "",
                                      "",
                                      device);

    auto inner_joined =
        inner_join(semi_joined, orders, {"l_orderkey"}, {"o_orderkey"}, "", "", device);

    auto inner_filtered =
        filter(inner_joined,
               expr(arrow_expr(cp::field_ref("o_orderstatus"), "==", string_literal("F"))),
               device);

    auto if_else_expr = expr(arrow_if_else(
        arrow_expr(cp::field_ref("l_receiptdate"), ">", cp::field_ref("l_commitdate")),
        int64_literal(1),
        int64_literal(0)));

    std::vector<std::string> project_names = {"l_orderkey", "l_suppkey", "if_else"};
    auto project_expr                      = exprs({"l_orderkey", "l_suppkey"});
    project_expr.push_back(std::move(if_else_expr));

    auto inner_projected = project(inner_filtered, project_expr, project_names, device);

    auto inner_grouped = group_by(inner_projected,
                                  {"l_orderkey", "l_suppkey"},
                                  {aggregate("hash_sum", "if_else", "failed_delivery_commit")},
                                  device);

    auto delivery_grouped =
        group_by(inner_grouped,
                 {"l_orderkey"},
                 {aggregate("hash_count", count_all(), "l_suppkey", "n_supplier"),
                  aggregate("hash_sum", "failed_delivery_commit", "num_failed")},
                 device);

    auto lineitems_filter =
        expr(arrow_expr(arrow_expr(cp::field_ref("n_supplier"), ">", int32_literal(1)),
                        "&&",
                        arrow_expr(cp::field_ref("num_failed"), "==", int64_literal(1))));

    auto line_items_needed = filter(delivery_grouped, lineitems_filter, device);

    auto line_items = left_semi_join(
        lineitem, line_items_needed, {"l_orderkey"}, {"l_orderkey"}, "", "_r", device);

    auto supplier_joined =
        inner_join(supplier, line_items, {"s_suppkey"}, {"l_suppkey"}, "", "", device);

    auto supplier_filtered =
        filter(supplier_joined,
               expr(arrow_expr(cp::field_ref("l_receiptdate"), ">", cp::field_ref("l_commitdate"))),
               device);

    auto sn_joined =
        inner_join(supplier_filtered, nation, {"s_nationkey"}, {"n_nationkey"}, "", "", device);

    auto sn_filtered =
        filter(sn_joined,
               expr(arrow_expr(cp::field_ref("n_name"), "==", string_literal("SAUDI ARABIA"))),
               device);

    auto sn_grouped = group_by(sn_filtered,
                               {"s_name"},
                               {aggregate("hash_count", count_all(), "l_orderkey", "numwait")},
                               device);

    auto sn_ordered =
        order_by(sn_grouped, {{"numwait", SortOrder::DESCENDING}, {"s_name"}}, device);

    auto sn_limit = limit(sn_ordered, 100, 0, device);

    return query_plan(table_sink(sn_limit));
}

std::shared_ptr<QueryPlan> q21_optimized(std::shared_ptr<Database>& db, DeviceType device) {
    auto lineitem = table_source(db, "lineitem", schema("lineitem"), {}, device);
    auto orders   = table_source(db, "orders", schema("orders"), {}, device);
    auto supplier = table_source(db, "supplier", schema("supplier"), {}, device);
    auto nation   = table_source(db, "nation", schema("nation"), {}, device);

    auto lineitem_with_failed =
        filter(lineitem,
               expr(arrow_expr(cp::field_ref("l_receiptdate"), ">", cp::field_ref("l_commitdate"))),
               device);

    // Step 1: Group by l_orderkey and count distinct l_suppkey
    auto grouped_lineitem =
        group_by(lineitem,
                 {"l_orderkey"},
                 {aggregate("hash_count_distinct", "l_suppkey", "n_supp_by_order")},
                 device);

    // Step 2: Filter orders with more than one supplier
    auto filtered_orders =
        filter(grouped_lineitem,
               expr(arrow_expr(cp::field_ref("n_supp_by_order"), ">", int32_literal(1))),
               device);

    // Step 3: Join with lineitem on l_orderkey
    auto joined_lineitem = left_semi_join(
        lineitem_with_failed, filtered_orders, {"l_orderkey"}, {"l_orderkey"}, "", "", device);

    // Step 4: Join with supplier and nation
    auto supplier_joined =
        inner_join(supplier, joined_lineitem, {"s_suppkey"}, {"l_suppkey"}, "", "", device);

    auto nation_joined =
        inner_join(supplier_joined, nation, {"s_nationkey"}, {"n_nationkey"}, "", "", device);

    // Step 5: Join with orders on l_orderkey
    auto orders_joined =
        inner_join(nation_joined, orders, {"l_orderkey"}, {"o_orderkey"}, "", "", device);

    // Step 6: Filter results
    auto filtered_results = filter(
        orders_joined,
        expr(arrow_all({arrow_expr(cp::field_ref("n_name"), "==", string_literal("SAUDI ARABIA")),
                        arrow_expr(cp::field_ref("o_orderstatus"), "==", string_literal("F"))})),
        device);

    // Step 7: Group by s_name and count the number of waits
    auto grouped_final = group_by(filtered_results,
                                  {"s_name"},
                                  {aggregate("hash_count", count_all(), "l_orderkey", "numwait")},
                                  device);

    // Step 8: Order by numwait and s_name
    auto ordered_results =
        order_by(grouped_final, {{"numwait", SortOrder::DESCENDING}, {"s_name"}}, device);

    // Step 9: Limit to top 100 results
    auto limited_results = limit(ordered_results, 100, 0, device);

    return query_plan(table_sink(limited_results));
}

std::shared_ptr<QueryPlan> q22(std::shared_ptr<Database>& db, DeviceType device) {
    auto customer = table_source(db, "customer", schema("customer"), {}, device);
    auto orders   = table_source(db, "orders", schema("orders"), {}, device);

    std::vector<cp::Expression> valid_codes = {string_literal("13"),
                                               string_literal("31"),
                                               string_literal("23"),
                                               string_literal("29"),
                                               string_literal("30"),
                                               string_literal("18"),
                                               string_literal("17")};

    auto customer_filter =
        expr(arrow_expr(arrow_in(arrow_substring(cp::field_ref("c_phone"), 0, 2), valid_codes),
                        "&&",
                        arrow_expr(cp::field_ref("c_acctbal"), ">", float64_literal(0.0))));

    auto customers_filtered = filter(customer, customer_filter, device);

    auto customers_aggregated =
        group_by(customers_filtered,
                 {},
                 {aggregate("mean", sum_ignore_nulls(), "c_acctbal", "acctbal_min")},
                 device);

    auto acctbal_mins = project(customers_aggregated,
                                {expr(cp::field_ref("acctbal_min")), expr(int32_literal(1))},
                                {"acctbal_min", "join_id"},
                                device);

    auto customer_with_codes = project(customers_filtered,
                                       {expr(cp::field_ref("c_custkey")),
                                        expr(cp::field_ref("c_acctbal")),
                                        expr(arrow_substring(cp::field_ref("c_phone"), 0, 2)),
                                        expr(int32_literal(1))},
                                       {"c_custkey", "c_acctbal", "cntrycode", "join_id"},
                                       device);

    auto joined = left_outer_join(
        customer_with_codes, acctbal_mins, {"join_id"}, {"join_id"}, "", "_r", device);

    auto join_filter =
        expr(arrow_expr(arrow_in(cp::field_ref("cntrycode"), valid_codes),
                        "&&",
                        arrow_expr(cp::field_ref("c_acctbal"), ">", cp::field_ref("acctbal_min"))));

    auto joined_filtered = filter(joined, join_filter, device);

    auto anti_joined =
        left_anti_join(joined_filtered, orders, {"c_custkey"}, {"o_custkey"}, "", "", device);

    auto anti_join_select = project(anti_joined, {"cntrycode", "c_acctbal"}, device);

    auto anti_join_grouped = group_by(anti_join_select,
                                      {"cntrycode"},
                                      {aggregate("hash_count", count_all(), "c_acctbal", "numcust"),
                                       aggregate("hash_sum", "c_acctbal", "totacctbal")},
                                      device);

    auto anti_join_sorted = order_by(anti_join_grouped, {{"cntrycode"}}, device);

    return query_plan(table_sink(anti_join_sorted));
}

std::shared_ptr<QueryPlan> q_empty(std::shared_ptr<Database>& db,
                                   DeviceType device,
                                   const std::string& table_name) {
    auto table_schema = schema(table_name);
    auto columns      = table_schema->column_names();
    int num_columns   = columns.size();
    std::vector<std::string> chosen_columns(columns.begin(), columns.begin() + num_columns);
    auto source     = table_source(db, table_name, table_schema, chosen_columns, device);
    auto limit_node = limit(source, 1, 0, device);
    return query_plan(table_sink(limit_node));
}

}  // namespace maximus::tpch
