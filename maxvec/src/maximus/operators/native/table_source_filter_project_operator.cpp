#include <maximus/error_handling.hpp>
#include <maximus/frontend/dataset_api.hpp>
#include <maximus/io/parquet.hpp>
#include <maximus/operators/native/table_source_filter_project_operator.hpp>
#include <maximus/utils/utils.hpp>

namespace maximus::native {

TableSourceFilterProjectOperator::TableSourceFilterProjectOperator(
    std::shared_ptr<MaximusContext>& ctx,
    std::shared_ptr<TableSourceFilterProjectProperties> properties)
        : AbstractTableSourceFilterProjectOperator(ctx, std::move(properties)) {
    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::NATIVE);

    assert(ctx_ && "TableSourceFilterProjectOperator's context must not be null");
    auto pool = ctx_->get_memory_pool();
    assert(pool && "TableSourceFilterProjectOperator's memory pool must not be null");

    auto maybe_scanner = build_scanner_from_files(this->ctx_,
                                                  this->properties->paths,
                                                  this->properties->schema->get_schema(),
                                                  this->properties->include_columns,
                                                  this->properties->filter_expr,
                                                  this->properties->exprs,
                                                  this->properties->column_names);
    CHECK_STATUS(maybe_scanner.status());
    auto scanner = maybe_scanner.ValueOrDie();

    auto maybe_reader = scanner->ToRecordBatchReader();
    CHECK_STATUS(maybe_reader.status());
    reader_ = std::move(maybe_reader.ValueOrDie());

    assign_input_schemas({this->properties->schema});
    assign_output_schema(std::make_shared<Schema>(reader_->schema()));

    assert(input_schemas.size() == 1 && input_schemas[0]);
    assert(output_schema);
}

void TableSourceFilterProjectOperator::read_next() {
    assert(!finished_ &&
           "TableSourceFilterProjectOperator::read_next() called after the end of the file");
    assert(!output_ && "TableSourceFilterProjectOperator::read_next() called when the "
                       "output_ is not empty");

    std::shared_ptr<arrow::RecordBatch> next;
    if (reader_) {
        // read the next batch from the reader_
        CHECK_STATUS(reader_->ReadNext(&next));
    }

    if (next) {
        output_ = std::move(next);
    }
}

bool TableSourceFilterProjectOperator::has_more_batches_impl(bool blocking) {
    if (output_) return true;

    if (finished_) return false;

    assert(input_schemas[0]);
    assert(output_schema);

    read_next();

    if (!output_) {
        finished_ = true;
    }

    return !finished_;
}

DeviceTablePtr TableSourceFilterProjectOperator::export_next_batch_impl() {
    assert(output_);
    return DeviceTablePtr(std::move(output_));
}

}  // namespace maximus::native
