#include <maximus/clickbench/clickbench_queries.hpp>
#include <maximus/frontend/expressions.hpp>
#include <maximus/frontend/query_plan_api.hpp>
#include <maximus/types/expression.hpp>

namespace maximus::clickbench {

std::shared_ptr<arrow::DataType> max_size_string(int size) {
    // return arrow::fixed_size_binary(size);
    return arrow::utf8();
}

std::shared_ptr<arrow::DataType> fixed_size_string(int size) {
    // return arrow::fixed_size_binary(size);
    return arrow::utf8();
}

std::shared_ptr<Schema> schema(const std::string& table_name) {
    if (table_name == "hits" || table_name == "t") {
        auto fields = {
            arrow::field("WatchID", arrow::int64(), false),
            arrow::field("JavaEnable", arrow::int16(), false),
            arrow::field("Title", arrow::utf8(), false),
            arrow::field("GoodEvent", arrow::int16(), false),
            // arrow::field("EventTime", Datetime(time_unit='ns', time_zone=None), false),
            // arrow::field("EventDate", Datetime(time_unit='ns', time_zone=None), false),
            arrow::field("EventTime", arrow::timestamp(arrow::TimeUnit::NANO), false),
            arrow::field("EventDate", arrow::timestamp(arrow::TimeUnit::NANO), false),
            // arrow::field("EventTime", arrow::date32(), false),
            // arrow::field("EventDate", arrow::date32(), false),
            arrow::field("CounterID", arrow::int32(), false),
            arrow::field("ClientIP", arrow::int32(), false),
            arrow::field("RegionID", arrow::int32(), false),
            arrow::field("UserID", arrow::int64(), false),
            arrow::field("CounterClass", arrow::int16(), false),
            arrow::field("OS", arrow::int16(), false),
            arrow::field("UserAgent", arrow::int16(), false),
            arrow::field("URL", arrow::utf8(), false),
            arrow::field("Referer", arrow::utf8(), false),
            arrow::field("IsRefresh", arrow::int16(), false),
            arrow::field("RefererCategoryID", arrow::int16(), false),
            arrow::field("RefererRegionID", arrow::int32(), false),
            arrow::field("URLCategoryID", arrow::int16(), false),
            arrow::field("URLRegionID", arrow::int32(), false),
            arrow::field("ResolutionWidth", arrow::int16(), false),
            arrow::field("ResolutionHeight", arrow::int16(), false),
            arrow::field("ResolutionDepth", arrow::int16(), false),
            arrow::field("FlashMajor", arrow::int16(), false),
            arrow::field("FlashMinor", arrow::int16(), false),
            arrow::field("FlashMinor2", arrow::utf8(), false),
            arrow::field("NetMajor", arrow::int16(), false),
            arrow::field("NetMinor", arrow::int16(), false),
            arrow::field("UserAgentMajor", arrow::int16(), false),
            arrow::field("UserAgentMinor", arrow::utf8(), false),
            arrow::field("CookieEnable", arrow::int16(), false),
            arrow::field("JavascriptEnable", arrow::int16(), false),
            arrow::field("IsMobile", arrow::int16(), false),
            arrow::field("MobilePhone", arrow::int16(), false),
            arrow::field("MobilePhoneModel", arrow::utf8(), false),
            arrow::field("Params", arrow::utf8(), false),
            arrow::field("IPNetworkID", arrow::int32(), false),
            arrow::field("TraficSourceID", arrow::int16(), false),
            arrow::field("SearchEngineID", arrow::int16(), false),
            arrow::field("SearchPhrase", arrow::utf8(), false),
            arrow::field("AdvEngineID", arrow::int16(), false),
            arrow::field("IsArtifical", arrow::int16(), false),
            arrow::field("WindowClientWidth", arrow::int16(), false),
            arrow::field("WindowClientHeight", arrow::int16(), false),
            arrow::field("ClientTimeZone", arrow::int16(), false),
            arrow::field("ClientEventTime", arrow::int64(), false),
            arrow::field("SilverlightVersion1", arrow::int16(), false),
            arrow::field("SilverlightVersion2", arrow::int16(), false),
            arrow::field("SilverlightVersion3", arrow::int32(), false),
            arrow::field("SilverlightVersion4", arrow::int16(), false),
            arrow::field("PageCharset", arrow::utf8(), false),
            arrow::field("CodeVersion", arrow::int32(), false),
            arrow::field("IsLink", arrow::int16(), false),
            arrow::field("IsDownload", arrow::int16(), false),
            arrow::field("IsNotBounce", arrow::int16(), false),
            arrow::field("FUniqID", arrow::int64(), false),
            arrow::field("OriginalURL", arrow::utf8(), false),
            arrow::field("HID", arrow::int32(), false),
            arrow::field("IsOldCounter", arrow::int16(), false),
            arrow::field("IsEvent", arrow::int16(), false),
            arrow::field("IsParameter", arrow::int16(), false),
            arrow::field("DontCountHits", arrow::int16(), false),
            arrow::field("WithHash", arrow::int16(), false),
            arrow::field("HitColor", arrow::utf8(), false),
            arrow::field("LocalEventTime", arrow::int64(), false),
            arrow::field("Age", arrow::int16(), false),
            arrow::field("Sex", arrow::int16(), false),
            arrow::field("Income", arrow::int16(), false),
            arrow::field("Interests", arrow::int16(), false),
            arrow::field("Robotness", arrow::int16(), false),
            arrow::field("RemoteIP", arrow::int32(), false),
            arrow::field("WindowName", arrow::int32(), false),
            arrow::field("OpenerName", arrow::int32(), false),
            arrow::field("HistoryLength", arrow::int16(), false),
            arrow::field("BrowserLanguage", arrow::utf8(), false),
            arrow::field("BrowserCountry", arrow::utf8(), false),
            arrow::field("SocialNetwork", arrow::utf8(), false),
            arrow::field("SocialAction", arrow::utf8(), false),
            arrow::field("HTTPError", arrow::int16(), false),
            arrow::field("SendTiming", arrow::int32(), false),
            arrow::field("DNSTiming", arrow::int32(), false),
            arrow::field("ConnectTiming", arrow::int32(), false),
            arrow::field("ResponseStartTiming", arrow::int32(), false),
            arrow::field("ResponseEndTiming", arrow::int32(), false),
            arrow::field("FetchTiming", arrow::int32(), false),
            arrow::field("SocialSourceNetworkID", arrow::int16(), false),
            arrow::field("SocialSourcePage", arrow::utf8(), false),
            arrow::field("ParamPrice", arrow::int64(), false),
            arrow::field("ParamOrderID", arrow::utf8(), false),
            arrow::field("ParamCurrency", arrow::utf8(), false),
            arrow::field("ParamCurrencyID", arrow::int16(), false),
            arrow::field("OpenstatServiceName", arrow::utf8(), false),
            arrow::field("OpenstatCampaignID", arrow::utf8(), false),
            arrow::field("OpenstatAdID", arrow::utf8(), false),
            arrow::field("OpenstatSourceID", arrow::utf8(), false),
            arrow::field("UTMSource", arrow::utf8(), false),
            arrow::field("UTMMedium", arrow::utf8(), false),
            arrow::field("UTMCampaign", arrow::utf8(), false),
            arrow::field("UTMContent", arrow::utf8(), false),
            arrow::field("UTMTerm", arrow::utf8(), false),
            arrow::field("FromTag", arrow::utf8(), false),
            arrow::field("HasGCLID", arrow::int16(), false),
            arrow::field("RefererHash", arrow::int64(), false),
            arrow::field("URLHash", arrow::int64(), false),
            arrow::field("CLID", arrow::int32(), false),
        };
        return std::make_shared<Schema>(fields);
    }
    throw std::runtime_error("The schema for given table not known.");
}

std::vector<std::string> table_names() {
    return {"t"};
}

std::vector<std::shared_ptr<Schema>> schemas() {
    auto tables = table_names();
    std::vector<std::shared_ptr<maximus::Schema>> table_schemas;
    table_schemas.reserve(tables.size());
    for (const auto& table : tables) {
        table_schemas.push_back(maximus::clickbench::schema(table));
    }
    return table_schemas;
}

std::shared_ptr<QueryPlan> q0(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(db, table_name, schema(table_name), {"UserID"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    auto count_opts                              = count_all();
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("count", count_opts, "UserID", "count")};
    auto group_by_node = group_by(source_node, {}, aggs, device);
    return query_plan(table_sink(group_by_node));
}

std::shared_ptr<QueryPlan> q1(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    // std::string table_name = "hits";
    std::string table_name = "t";
    auto source_node = table_source(db, table_name, schema(table_name), {"AdvEngineID"}, device);
    /* ==============================
     * CREATING A FILTER
     * ==============================
     */
    auto engine_id_filter = arrow_expr(cp::field_ref("AdvEngineID"), "!=", int32_literal(0));
    auto filter_node      = filter(source_node, expr(engine_id_filter), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    auto count_opts                              = count_all();
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("count", count_opts, "AdvEngineID", "count")};
    auto group_by_node = group_by(filter_node, {}, aggs, device);
    return query_plan(table_sink(group_by_node));
}

std::shared_ptr<QueryPlan> q2(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(
        db, table_name, schema(table_name), {"AdvEngineID", "ResolutionWidth"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    auto count_opts                              = count_all();
    auto sum_opts                                = sum_defaults();
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("sum", count_opts, "AdvEngineID", "AdvEngineID_sum"),
        aggregate("count", count_opts, "AdvEngineID", "count"),
        aggregate("mean", sum_opts, "ResolutionWidth", "ResolutionWidth_mean")};
    auto group_by_node = group_by(source_node, {}, aggs, device);
    return query_plan(table_sink(group_by_node));
}

std::shared_ptr<QueryPlan> q3(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(db, table_name, schema(table_name), {"UserID"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    auto sum_opts                                = sum_defaults();
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("mean", sum_opts, "UserID", "UserID_mean")};
    auto group_by_node = group_by(source_node, {}, aggs, device);
    return query_plan(table_sink(group_by_node));
}

std::shared_ptr<QueryPlan> q4(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(db, table_name, schema(table_name), {"UserID"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("count_distinct", "UserID", "UserID_distinct_count")};
    auto group_by_node = group_by(source_node, {}, aggs, device);
    return query_plan(table_sink(group_by_node));
}

std::shared_ptr<QueryPlan> q5(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node = table_source(db, table_name, schema(table_name), {"SearchPhrase"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("count_distinct", "SearchPhrase", "SearchPhrase_distinct_count")};
    auto group_by_node = group_by(source_node, {}, aggs, device);
    return query_plan(table_sink(group_by_node));
}

std::shared_ptr<QueryPlan> q6(std::shared_ptr<Database>& db, DeviceType device) {
    // todo: the result is not correct. Probably due to the type of the column EventDate
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node = table_source(db, table_name, schema(table_name), {"EventDate"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {aggregate("min", "EventDate", "EventDate_min"),
                                                    aggregate("max", "EventDate", "EventDate_max")};
    auto group_by_node                           = group_by(source_node, {}, aggs, device);
    return query_plan(table_sink(group_by_node));
}

std::shared_ptr<QueryPlan> q7(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"AdvEngineID", "UserID"}, device);
    /* ==============================
     * CREATING A FILTER
     * ==============================
     */
    auto engine_id_filter = arrow_expr(cp::field_ref("AdvEngineID"), "!=", int32_literal(0));
    auto filter_node      = filter(source_node, expr(engine_id_filter), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    auto count_opts                              = count_all();
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", count_opts, "UserID", "count"),
    };
    auto group_by_node = group_by(filter_node, {"AdvEngineID"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"count", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q8(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"RegionID", "UserID"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count_distinct", "UserID", "UserID")};
    auto group_by_node = group_by(source_node, {"RegionID"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"UserID", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q9(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(db,
                                    table_name,
                                    schema(table_name),
                                          {"RegionID", "AdvEngineID", "ResolutionWidth", "UserID"},
                                    device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_sum", "AdvEngineID", "AdvEngineID_sum"),
        aggregate("hash_mean", "ResolutionWidth", "ResolutionWidth_mean"),
        aggregate("hash_count_distinct", "UserID", "UserID_nunique")};
    auto group_by_node = group_by(source_node, {"RegionID"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"AdvEngineID_sum", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q10(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"MobilePhoneModel", "UserID"}, device);
    /* ==============================
     * CREATING A FILTER
     * ==============================
     */
    auto engine_id_filter = arrow_expr(cp::field_ref("MobilePhoneModel"), "!=", string_literal(""));
    auto filter_node      = filter(source_node, expr(engine_id_filter), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count_distinct", "UserID", "UserID")};
    auto group_by_node = group_by(filter_node, {"MobilePhoneModel"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"UserID", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q11(std::shared_ptr<Database>& db, DeviceType device) {
    // todo: the MobilePhone column at the end is wrong, though the result of aggregation is correct.
    // todo: this is because of type of MobilePhone which is int16. When it is changed to int32 the problem is fixed.
    // todo: maybe the problem is where maximus print the data
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(
        db, table_name, schema(table_name), {"MobilePhoneModel", "UserID", "MobilePhone"}, device);
    /* ==============================
     * CREATING A FILTER
     * ==============================
     */
    auto engine_id_filter = arrow_expr(cp::field_ref("MobilePhoneModel"), "!=", string_literal(""));
    auto filter_node      = filter(source_node, expr(engine_id_filter), device);

    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count_distinct", "UserID", "count_distinct")};
    auto group_by_node = group_by(filter_node, {"MobilePhone", "MobilePhoneModel"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"count_distinct", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q12(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"SearchPhrase", "UserID"}, device);
    /* ==============================
     * CREATING A FILTER
     * ==============================
     */
    auto search_phrase_filter = arrow_expr(cp::field_ref("SearchPhrase"), "!=", string_literal(""));
    auto filter_node          = filter(source_node, expr(search_phrase_filter), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {aggregate("hash_count", "UserID", "count")};
    auto group_by_node = group_by(filter_node, {"SearchPhrase"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"count", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q13(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"SearchPhrase", "UserID"}, device);
    /* ==============================
     * CREATING A FILTER
     * ==============================
     */
    auto search_phrase_filter = arrow_expr(cp::field_ref("SearchPhrase"), "!=", string_literal(""));
    auto filter_node          = filter(source_node, expr(search_phrase_filter), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count_distinct", "UserID", "UserID")};
    auto group_by_node = group_by(filter_node, {"SearchPhrase"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"UserID", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q14(std::shared_ptr<Database>& db, DeviceType device) {
    // todo: the SearchEngineID column at the end is wrong, though the result of aggregation is correct.
    // todo: this is because of type of MobilePhone which is int16. When it is changed to int32 the problem is fixed.
    // todo: maybe the problem is where maximus print the data
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(
        db, table_name, schema(table_name), {"SearchEngineID", "SearchPhrase", "UserID"}, device);
    /* ==============================
     * CREATING A FILTER
     * ==============================
     */
    auto search_phrase_filter = arrow_expr(cp::field_ref("SearchPhrase"), "!=", string_literal(""));
    auto filter_node          = filter(source_node, expr(search_phrase_filter), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {aggregate("hash_count", "UserID", "count")};
    auto group_by_node = group_by(filter_node, {"SearchEngineID", "SearchPhrase"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"count", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q15(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"SearchEngineID", "UserID"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "SearchEngineID", "count")};
    auto group_by_node = group_by(source_node, {"UserID"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"count", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q16(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(
        db, table_name, schema(table_name), {"SearchEngineID", "SearchPhrase", "UserID"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "SearchEngineID", "count")};
    auto group_by_node = group_by(source_node, {"UserID", "SearchPhrase"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"count", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q17(std::shared_ptr<Database>& db, DeviceType device) {
    return q16(db, device);
}

std::shared_ptr<QueryPlan> q18(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(
        db, table_name, schema(table_name), {"UserID", "EventTime", "SearchPhrase"}, device);

    std::vector<std::shared_ptr<Expression>> all_exprs = {expr(cp::field_ref("UserID")),
                                                          expr(cp::field_ref("EventTime")),
                                                          expr(cp::field_ref("SearchPhrase")),
                                                          expr(minute(cp::field_ref("EventTime")))};
    std::vector<std::string> all_exprs_names = {"UserID", "EventTime", "SearchPhrase", "Minute"};
    auto all_project = project(source_node, all_exprs, all_exprs_names, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "Minute", "count"),
    };
    auto group_by_node = group_by(all_project, {"UserID", "Minute", "SearchPhrase"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"count", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q19(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(db, table_name, schema(table_name), {"UserID"}, device);
    /* ==============================
     * CREATING A FILTER
     * ==============================
     */
    auto user_filter = arrow_expr(cp::field_ref("UserID"), "==", int64_literal(435090932899640449));
    auto filter_node = filter(source_node, expr(user_filter), device);
    return query_plan(table_sink(filter_node));
}

std::shared_ptr<QueryPlan> q20(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node = table_source(db, table_name, schema(table_name), {"UserID", "URL"}, device);
    /* ==============================
     * CREATING A FILTER
     * ==============================
     */
    auto url_filter  = arrow_field_like("URL", "%google%");
    auto filter_node = filter(source_node, expr(url_filter), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {aggregate("count", "URL", "count")};
    auto group_by_node                           = group_by(filter_node, {}, aggs, device);
    return query_plan(table_sink(group_by_node));
}

std::shared_ptr<QueryPlan> q21(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"URL", "SearchPhrase"}, device);
    /* ==============================
     * CREATING A FILTER
     * ==============================
     */
    auto url_filter           = arrow_field_like("URL", "%google%");
    auto filter_node          = filter(source_node, expr(url_filter), device);
    auto search_phrase_filter = arrow_expr(cp::field_ref("SearchPhrase"), "!=", string_literal(""));
    filter_node               = filter(filter_node, expr(search_phrase_filter), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "URL", "count"),
        aggregate("hash_min", "URL", "min"),
    };
    auto group_by_node = group_by(filter_node, {"SearchPhrase"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"count", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q22(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(
        db, table_name, schema(table_name), {"URL", "Title", "UserID", "SearchPhrase"}, device);
    /* ==============================
     * CREATING A FILTER
     * ==============================
     */
    auto title_filter         = arrow_field_like("Title", "%Google%");
    auto filter_node          = filter(source_node, expr(title_filter), device);
    auto url_filter           = arrow_not(arrow_field_like("URL", "%\\.google\\.%"));
    filter_node               = filter(filter_node, expr(url_filter), device);
    auto search_phrase_filter = arrow_expr(cp::field_ref("SearchPhrase"), "!=", string_literal(""));
    filter_node               = filter(filter_node, expr(search_phrase_filter), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "URL", "count"),
        aggregate("hash_min", "URL", "url_min"),
        aggregate("hash_min", "Title", "title_min"),
        aggregate("hash_count_distinct", "UserID", "distinct_users"),
    };
    auto group_by_node = group_by(filter_node, {"SearchPhrase"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"count", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q23(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"URL", "EventTime", "UserID"}, device);
    /* ==============================
     * CREATING A FILTER
     * ==============================
     */
    auto url_filter  = arrow_field_like("URL", "%google%");
    auto filter_node = filter(source_node, expr(url_filter), device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"EventTime"}};
    auto order_by_node             = order_by(filter_node, sort_keys, device);
    auto project_node              = project(order_by_node, {"UserID"}, device);
    return query_plan(table_sink(project_node));
}

std::shared_ptr<QueryPlan> q24(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"SearchPhrase", "EventTime"}, device);
    /* ==============================
     * CREATING A FILTER
     * ==============================
     */
    auto search_phrase_filter = arrow_expr(cp::field_ref("SearchPhrase"), "!=", string_literal(""));
    auto filter_node          = filter(source_node, expr(search_phrase_filter), device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"EventTime"}};
    auto order_by_node             = order_by(filter_node, sort_keys, device);
    auto project_node              = project(order_by_node, {"SearchPhrase"}, device);
    return query_plan(table_sink(project_node));
}

std::shared_ptr<QueryPlan> q25(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node = table_source(db, table_name, schema(table_name), {"SearchPhrase"}, device);
    /* ==============================
     * CREATING A FILTER
     * ==============================
     */
    auto search_phrase_filter = arrow_expr(cp::field_ref("SearchPhrase"), "!=", string_literal(""));
    auto filter_node          = filter(source_node, expr(search_phrase_filter), device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"SearchPhrase"}};
    auto order_by_node             = order_by(filter_node, sort_keys, device);
    auto project_node              = project(order_by_node, {"SearchPhrase"}, device);
    return query_plan(table_sink(project_node));
}

std::shared_ptr<QueryPlan> q26(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"SearchPhrase", "EventTime"}, device);
    /* ==============================
     * CREATING A FILTER
     * ==============================
     */
    auto search_phrase_filter = arrow_expr(cp::field_ref("SearchPhrase"), "!=", string_literal(""));
    auto filter_node          = filter(source_node, expr(search_phrase_filter), device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"EventTime"}, {"SearchPhrase"}};
    auto order_by_node             = order_by(filter_node, sort_keys, device);
    auto project_node              = project(order_by_node, {"SearchPhrase"}, device);
    return query_plan(table_sink(project_node));
}

std::shared_ptr<QueryPlan> q27(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"CounterID", "URL"}, device);
    /* ==============================
     * CREATING A FILTER
     * ==============================
     */
    auto url_filter  = arrow_expr(cp::field_ref("URL"), "!=", string_literal(""));
    auto filter_node = filter(source_node, expr(url_filter), device);

    std::vector<std::shared_ptr<Expression>> all_exprs = {expr(cp::field_ref("CounterID")),
                                                          expr(cp::field_ref("URL")),
                                                          expr(arrow_len(cp::field_ref("URL")))};
    std::vector<std::string> all_exprs_names           = {"CounterID", "URL", "len_url"};
    auto all_project = project(filter_node, all_exprs, all_exprs_names, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "URL", "c"),
        aggregate("hash_mean", "len_url", "l"),
    };
    auto group_by_node = group_by(all_project, {"CounterID"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    auto count_filter = arrow_expr(cp::field_ref("c"), ">", int64_literal(100000));
    filter_node       = filter(group_by_node, expr(count_filter), device);

    std::vector<SortKey> sort_keys = {{"l", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(filter_node, sort_keys, device);
    auto limit_node                = limit(order_by_node, 25, 0, device);
    return query_plan(table_sink(limit_node));
}


std::shared_ptr<QueryPlan> q29(std::shared_ptr<Database>& db, DeviceType device) {
    // todo: the polars implementation of clickbench for this query is not correct
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db, table_name, schema(table_name), {"ResolutionWidth"}, device);
    /* =================================
     * CREATING A LIMIT NODE
     * =================================
     */
    auto limit_node = limit(source_node, 90, 0, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<maximus::Expression>> additions;
    std::vector<std::string> column_names;
    for (int i = 0; i < 90; i++) {
        additions.push_back(
            expr(arrow_expr(cp::field_ref("ResolutionWidth"), "+", int32_literal(i))));
        column_names.push_back("col_" + std::to_string(i));
    }
    auto all_projected = project(limit_node, additions, column_names, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs;
    for (int i = 0; i < 90; i++) {
        auto new_column_name = "sum_" + std::to_string(i);
        aggs.push_back(aggregate("sum", column_names[i], new_column_name));
    }
    auto group_by_node = group_by(all_projected, {}, aggs, device);
    return query_plan(table_sink(group_by_node));
}

std::shared_ptr<QueryPlan> q30(std::shared_ptr<Database>& db, DeviceType device) {
    // todo: the SearchEngineID column at the end is wrong, though the result of aggregation is correct.
    // todo: this is because of type of MobilePhone which is int16. When it is changed to int32 the problem is fixed.
    // todo: maybe the problem is where maximus print the data
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db,
                     table_name,
                     schema(table_name),
                     {"SearchEngineID", "ClientIP", "SearchPhrase", "IsRefresh", "ResolutionWidth"},
                     device);
    /* ==============================
     * CREATING A FILTER NODE
     * ==============================
     */
    auto search_phrase_filter = arrow_expr(cp::field_ref("SearchPhrase"), "!=", string_literal(""));
    auto filter_node          = filter(source_node, expr(search_phrase_filter), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "SearchEngineID", "c"),
        aggregate("hash_sum", "IsRefresh", "IsRefreshSum"),
        aggregate("hash_mean", "ResolutionWidth", "AvgResolutionWidth"),
    };
    auto group_by_node = group_by(filter_node, {"SearchEngineID", "ClientIP"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"c", SortOrder::DESCENDING},
                                      {"ClientIP", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q31(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db,
                     table_name,
                     schema(table_name),
                     {"WatchID", "ClientIP", "SearchPhrase", "IsRefresh", "ResolutionWidth"},
                     device);
    /* ==============================
     * CREATING A FILTER NODE
     * ==============================
     */
    auto search_phrase_filter = arrow_expr(cp::field_ref("SearchPhrase"), "!=", string_literal(""));
    auto filter_node          = filter(source_node, expr(search_phrase_filter), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "ClientIP", "c"),
        aggregate("hash_sum", "IsRefresh", "IsRefreshSum"),
        aggregate("hash_mean", "ResolutionWidth", "AvgResolutionWidth"),
    };
    auto group_by_node = group_by(filter_node, {"WatchID", "ClientIP"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"c", SortOrder::DESCENDING},
                                      {"ClientIP", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q32(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(db,
                                    table_name,
                                    schema(table_name),
                                          {"WatchID", "ClientIP", "IsRefresh", "ResolutionWidth"},
                                    device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "ClientIP", "c"),
        aggregate("hash_sum", "IsRefresh", "IsRefreshSum"),
        aggregate("hash_mean", "ResolutionWidth", "AvgResolutionWidth"),
    };
    auto group_by_node = group_by(source_node, {"WatchID", "ClientIP"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"c", SortOrder::DESCENDING},
                                      {"ClientIP", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q33(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(db, table_name, schema(table_name), {"URL"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "URL", "c"),
    };
    auto group_by_node = group_by(source_node, {"URL"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"c", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q34(std::shared_ptr<Database>& db, DeviceType device) {
    return q33(db, device);
}

std::shared_ptr<QueryPlan> q35(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(db, table_name, schema(table_name), {"ClientIP"}, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "ClientIP", "c"),
    };
    auto group_by_node = group_by(source_node, {"ClientIP"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"c", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q36(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(db,
                                    table_name,
                                    schema(table_name),
                                          {"CounterID", "EventDate", "DontCountHits", "IsRefresh", "URL"},
                                    device);
    /* ==============================
     * CREATING A FILTER NODE
     * ==============================
     */
    auto filter_expression =
        arrow_all({arrow_expr(cp::field_ref("CounterID"), "==", int32_literal(62)),
                   arrow_expr(cp::field_ref("EventDate"), ">=", date_literal("2013-07-01")),
                   arrow_expr(cp::field_ref("EventDate"), "<=", date_literal("2013-07-31")),
                   arrow_expr(cp::field_ref("DontCountHits"), "==", int32_literal(0)),
                   arrow_expr(cp::field_ref("IsRefresh"), "==", int32_literal(0)),
                   arrow_expr(cp::field_ref("URL"), "!=", string_literal(""))});
    auto filter_node = filter(source_node, expr(filter_expression), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "URL", "PageViews"),
    };
    auto group_by_node = group_by(filter_node, {"URL"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"PageViews", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q37(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db,
                     table_name,
                     schema(table_name),
                     {"CounterID", "EventDate", "DontCountHits", "IsRefresh", "Title"},
                     device);
    /* ==============================
     * CREATING A FILTER NODE
     * ==============================
     */
    auto filter_expression =
        arrow_all({arrow_expr(cp::field_ref("CounterID"), "==", int32_literal(62)),
                   arrow_expr(cp::field_ref("EventDate"), ">=", date_literal("2013-07-01")),
                   arrow_expr(cp::field_ref("EventDate"), "<=", date_literal("2013-07-31")),
                   arrow_expr(cp::field_ref("DontCountHits"), "==", int32_literal(0)),
                   arrow_expr(cp::field_ref("IsRefresh"), "==", int32_literal(0)),
                   arrow_expr(cp::field_ref("Title"), "!=", string_literal(""))});
    auto filter_node = filter(source_node, expr(filter_expression), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "Title", "PageViews"),
    };
    auto group_by_node = group_by(filter_node, {"Title"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"PageViews", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    return query_plan(table_sink(order_by_node));
}

std::shared_ptr<QueryPlan> q38(std::shared_ptr<Database>& db, DeviceType device) {
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db,
                     table_name,
                     schema(table_name),
                     {"CounterID", "EventDate", "IsRefresh", "IsLink", "IsDownload", "URL"},
                     device);
    /* ==============================
     * CREATING A FILTER NODE
     * ==============================
     */
    auto filter_expression =
        arrow_all({arrow_expr(cp::field_ref("CounterID"), "==", int32_literal(62)),
                   arrow_expr(cp::field_ref("EventDate"), ">=", date_literal("2013-07-01")),
                   arrow_expr(cp::field_ref("EventDate"), "<=", date_literal("2013-07-31")),
                   arrow_expr(cp::field_ref("IsRefresh"), "==", int32_literal(0)),
                   arrow_expr(cp::field_ref("IsLink"), "!=", int32_literal(0)),
                   arrow_expr(cp::field_ref("IsDownload"), "==", int32_literal(0))});
    auto filter_node = filter(source_node, expr(filter_expression), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "URL", "PageViews"),
    };
    auto group_by_node = group_by(filter_node, {"URL"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"PageViews", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    auto limit_node                = limit(order_by_node, 10, 1000, device);
    return query_plan(table_sink(limit_node));
}

std::shared_ptr<QueryPlan> q39(std::shared_ptr<Database>& db, DeviceType device) {
    // todo: this query is not implemented in polars by clickbench
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(db,
                                    table_name,
                                    schema(table_name),
                                          {"CounterID",
                                           "EventDate",
                                           "IsRefresh",
                                           "TraficSourceID",
                                           "SearchEngineID",
                                           "AdvEngineID",
                                           "Referer",
                                           "URL"},
                                    device);
    /* ==============================
     * CREATING A FILTER NODE
     * ==============================
     */
    auto filter_expression = arrow_all({
        arrow_expr(cp::field_ref("CounterID"), "==", int32_literal(62)),
        arrow_expr(cp::field_ref("EventDate"), ">=", date_literal("2013-07-01")),
        arrow_expr(cp::field_ref("EventDate"), "<=", date_literal("2013-07-31")),
        arrow_expr(cp::field_ref("IsRefresh"), "==", int32_literal(0)),
    });
    auto filter_node       = filter(source_node, expr(filter_expression), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "URL", "PageViews"),
    };
    auto group_by_node =
        group_by(filter_node,
                 {"TraficSourceID", "SearchEngineID", "AdvEngineID", "Referer", "URL"},
                 aggs,
                 device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"PageViews", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    auto limit_node                = limit(order_by_node, 10, 1000, device);
    return query_plan(table_sink(limit_node));
}


std::shared_ptr<QueryPlan> q40(std::shared_ptr<Database>& db, DeviceType device) {
    // todo: this query is not implemented in polars by clickbench
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(
        db,
        table_name,
        schema(table_name),
        {"CounterID", "EventDate", "IsRefresh", "TraficSourceID", "RefererHash", "URLHash"},
        device);
    /* ==============================
     * CREATING A FILTER NODE
     * ==============================
     */
    auto filter_expression = arrow_all(
        {arrow_expr(cp::field_ref("CounterID"), "==", int32_literal(62)),
         arrow_expr(cp::field_ref("EventDate"), ">=", date_literal("2013-07-01")),
         arrow_expr(cp::field_ref("EventDate"), "<=", date_literal("2013-07-31")),
         arrow_expr(cp::field_ref("IsRefresh"), "==", int32_literal(0)),
         arrow_any({
             arrow_expr(cp::field_ref("TraficSourceID"), "==", int32_literal(6)),
             arrow_expr(cp::field_ref("TraficSourceID"), "==", int32_literal(-1)),
         }),
         arrow_expr(cp::field_ref("RefererHash"), "==", int64_literal(3594120000172545465))});
    auto filter_node = filter(source_node, expr(filter_expression), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "URLHash", "PageViews"),
    };
    auto group_by_node = group_by(filter_node, {"URLHash", "EventDate"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"URLHash", SortOrder::DESCENDING},
                                      {"PageViews", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    auto limit_node                = limit(order_by_node, 10, 100, device);
    return query_plan(table_sink(limit_node));
}

std::shared_ptr<QueryPlan> q41(std::shared_ptr<Database>& db, DeviceType device) {
    // todo: the "WindowClientWidth", "WindowClientHeight" column are wrong, though the result of aggregation is correct.
    // todo: this is because of type of MobilePhone which is int16. When it is changed to int32 the problem is fixed.
    // todo: maybe the problem is where maximus print the data

    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node       = table_source(db,
                                    table_name,
                                    schema(table_name),
                                          {"CounterID",
                                           "EventDate",
                                           "IsRefresh",
                                           "DontCountHits",
                                           "URLHash",
                                           "WindowClientWidth",
                                           "WindowClientHeight"},
                                    device);
    /* ==============================
     * CREATING A FILTER NODE
     * ==============================
     */
    auto filter_expression =
        arrow_all({arrow_expr(cp::field_ref("CounterID"), "==", int32_literal(62)),
                   arrow_expr(cp::field_ref("EventDate"), ">=", date_literal("2013-07-01")),
                   arrow_expr(cp::field_ref("EventDate"), "<=", date_literal("2013-07-31")),
                   arrow_expr(cp::field_ref("IsRefresh"), "==", int32_literal(0)),
                   arrow_expr(cp::field_ref("DontCountHits"), "==", int32_literal(0)),
                   arrow_expr(cp::field_ref("URLHash"), "==", int64_literal(2868770270353813622))});
    auto filter_node = filter(source_node, expr(filter_expression), device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "IsRefresh", "PageViews"),
    };
    auto group_by_node =
        group_by(filter_node, {"WindowClientWidth", "WindowClientHeight"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"PageViews", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    auto limit_node                = limit(order_by_node, 10, 10000, device);
    return query_plan(table_sink(limit_node));
}

std::shared_ptr<QueryPlan> q42(std::shared_ptr<Database>& db, DeviceType device) {
    // todo: this query is not implemented in polars by clickbench
    auto ctx          = db->get_context();
    auto db_catalogue = db->get_catalogue();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    std::string table_name = "t";
    auto source_node =
        table_source(db,
                     table_name,
                     schema(table_name),
                     {"CounterID", "EventDate", "EventTime", "IsRefresh", "DontCountHits"},
                     device);
    /* ==============================
     * CREATING A FILTER NODE
     * ==============================
     */
    auto filter_expression = arrow_all({
        arrow_expr(cp::field_ref("CounterID"), "==", int32_literal(62)),
        arrow_expr(cp::field_ref("EventDate"), ">=", date_literal("2013-07-14")),
        arrow_expr(cp::field_ref("EventDate"), "<=", date_literal("2013-07-15")),
        arrow_expr(cp::field_ref("IsRefresh"), "==", int32_literal(0)),
        arrow_expr(cp::field_ref("DontCountHits"), "==", int32_literal(0)),
    });
    auto filter_node       = filter(source_node, expr(filter_expression), device);


    std::vector<std::shared_ptr<Expression>> all_exprs = {expr(cp::field_ref("CounterID")),
                                                          expr(cp::field_ref("EventDate")),
                                                          expr(cp::field_ref("IsRefresh")),
                                                          expr(cp::field_ref("DontCountHits")),
                                                          expr(minute(cp::field_ref("EventTime")))};
    std::vector<std::string> all_exprs_names           = {
        "CounterID", "EventDate", "IsRefresh", "DontCountHits", "Minute"};
    auto all_project = project(filter_node, all_exprs, all_exprs_names, device);
    /* ==============================
     * CREATING A GROUP BY NODE
     * ==============================
     */
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_count", "Minute", "PageViews"),
    };
    auto group_by_node = group_by(all_project, {"Minute"}, aggs, device);
    /* ==============================
     * CREATING AN ORDER BY NODE
     * ==============================
     */
    std::vector<SortKey> sort_keys = {{"PageViews", SortOrder::DESCENDING}};
    auto order_by_node             = order_by(group_by_node, sort_keys, device);
    auto limit_node                = limit(order_by_node, 10, 1000, device);
    return query_plan(table_sink(limit_node));
}

std::shared_ptr<QueryPlan> query_plan(const std::string& q,
                                      std::shared_ptr<Database>& db,
                                      DeviceType device) {
    if (q == "q0") {
        return q0(db, device);
    }
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
        return q21(db, device);
    }
    if (q == "q22") {
        return q22(db, device);
    }
    if (q == "q23") {
        return q23(db, device);
    }
    if (q == "q24") {
        return q24(db, device);
    }
    if (q == "q25") {
        return q25(db, device);
    }
    if (q == "q26") {
        return q26(db, device);
    }
    if (q == "q27") {
        return q27(db, device);
    }
    if (q == "q28") {
        throw std::runtime_error("Not implemented yet");
    }
    if (q == "q29") {
        return q29(db, device);
    }
    if (q == "q30") {
        return q30(db, device);
    }
    if (q == "q31") {
        return q31(db, device);
    }
    if (q == "q32") {
        return q32(db, device);
    }
    if (q == "q33") {
        return q33(db, device);
    }
    if (q == "q34") {
        return q34(db, device);
    }
    if (q == "q35") {
        return q35(db, device);
    }
    if (q == "q36") {
        return q36(db, device);
    }
    if (q == "q37") {
        return q37(db, device);
    }
    if (q == "q38") {
        return q38(db, device);
    }
    if (q == "q39") {
        return q39(db, device);
    }
    if (q == "q40") {
        return q40(db, device);
    }
    if (q == "q41") {
        return q41(db, device);
    }
    if (q == "q42") {
        return q42(db, device);
    }
    throw std::runtime_error("Non-existing clickbench query.");
}
}  // namespace maximus::clickbench