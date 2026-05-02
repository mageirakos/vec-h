#include <arrow/json/from_string.h>

#include <maximus/error_handling.hpp>
#include <maximus/utils/json_helpers.hpp>

namespace maximus {
std::shared_ptr<arrow::RecordBatch> rb_from_json(const std::shared_ptr<arrow::Schema> &schema,
                                                 std::string_view json) {
    // first create an arrow::StructType from the schema
    auto struct_type = arrow::struct_(schema->fields());

    auto maybe_array = arrow::json::ArrayFromJSONString(struct_type, json);
    if (!maybe_array.ok()) {
        CHECK_STATUS(maybe_array.status());
    }
    auto struct_array = maybe_array.ValueOrDie();

    // Then, convert it to a record batch
    auto maybe_rb = arrow::RecordBatch::FromStructArray(struct_array);
    if (!maybe_rb.ok()) {
        CHECK_STATUS(maybe_rb.status());
    }

    auto rb = maybe_rb.ValueOrDie();

    return rb;
}

std::shared_ptr<arrow::Table> table_from_json(const std::shared_ptr<arrow::Schema> &schema,
                                              const std::vector<std::string> &json) {
    std::vector<std::shared_ptr<arrow::RecordBatch>> rbs;
    rbs.reserve(json.size());
    for (const auto &batch_json : json) {
        rbs.push_back(rb_from_json(schema, batch_json));
    }
    auto maybe_table = arrow::Table::FromRecordBatches(schema, std::move(rbs));
    if (!maybe_table.ok()) {
        CHECK_STATUS(maybe_table.status());
    }
    return maybe_table.ValueOrDie();
}
}  // namespace maximus
