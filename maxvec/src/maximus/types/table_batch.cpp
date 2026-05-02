#include <iostream>
#include <maximus/error_handling.hpp>
#include <maximus/types/table_batch.hpp>
#include <maximus/utils/arrow_helpers.hpp>
#include <maximus/utils/json_helpers.hpp>
#include <memory>

namespace maximus {

Status TableBatch::from_record_batch(const std::shared_ptr<MaximusContext> &ctx,
                                     std::shared_ptr<arrow::RecordBatch> table,
                                     TableBatchPtr &tableOut) {
    const auto &schema = table->schema();
    auto schema_status = are_types_supported(schema);
    if (!schema_status.ok()) {
        return schema_status;
    }
    tableOut = std::make_shared<TableBatch>(ctx, std::move(table));
    return Status::OK();
}

Status TableBatch::from_json(const std::shared_ptr<MaximusContext> &ctx,
                             const std::shared_ptr<Schema> &schema,
                             std::string_view json,
                             TableBatchPtr &tableOut) {
    auto record_batch = rb_from_json(schema->get_schema(), json);
    return TableBatch::from_record_batch(ctx, std::move(record_batch), tableOut);
}

Status TableBatch::from_exec_batch(const std::shared_ptr<MaximusContext> &ctx,
                                   std::shared_ptr<Schema> schema,
                                   std::shared_ptr<arrow::compute::ExecBatch> batch,
                                   ::arrow::MemoryPool *pool,
                                   TableBatchPtr &tableOut) {
    auto maybe_record_batch = batch->ToRecordBatch(schema->get_schema(), pool);
    if (!maybe_record_batch.ok()) {
        return {maybe_record_batch.status()};
    }
    tableOut = std::make_shared<TableBatch>(ctx, maybe_record_batch.ValueOrDie());
    return Status::OK();
}

int TableBatch::num_columns() const {
    if (table_) return table_->num_columns();
    return 0;
}

std::vector<std::string> TableBatch::column_names() {
    std::vector<std::string> names;
    if (table_) {
        for (int32_t i = 0; i < table_->num_columns(); i++) {
            names.push_back(table_->column_name(i));
        }
        return names;
    }
    return {};
}

int64_t TableBatch::num_rows() const {
    if (table_) return table_->num_rows();
    return 0;
}

bool TableBatch::empty() const {
    if (!table_) {
        return true;
    }
    return table_->num_rows() == 0;
}

Status TableBatch::to_record_batch(std::shared_ptr<arrow::RecordBatch> &out) {
    assert(table_);
    out = table_;
    return Status::OK();
}

Status TableBatch::to_exec_batch(std::shared_ptr<arrow::compute::ExecBatch> &output) {
    assert(table_);
    output = std::make_shared<arrow::compute::ExecBatch>(*table_);
    return Status::OK();
}

std::string TableBatch::to_string(
    int64_t first_row, int64_t end_row, int first_col, int end_col, const std::string delim) const {
    if (!table_) return "";

    std::stringstream stream;

    auto schema_fields   = table_->schema()->fields();
    auto default_headers = table_->schema()->field_names();

    auto print_headers = [&](const std::vector<std::string> &headers) {
        for (int col = first_col; col < end_col; ++col) {
            stream << headers[col] << "(" << schema_fields[col]->type()->name() << ")";
            stream << (col < end_col - 1 ? delim : "\n");
        }
    };

    print_headers(default_headers);

    auto print_row = [&](int64_t row) {
        for (int col = first_col; col < end_col; ++col) {
            auto col_data      = table_->column(col);
            int64_t row_offset = 0;
            if (row_offset <= row && row < row_offset + col_data->length()) {
                stream << maximus::array_string(col_data, row - row_offset);
                if (col < end_col - 1) stream << delim;
            }
            row_offset += col_data->length();
        }
        stream << '\n';
    };

    for (int64_t rowIdx = first_row; rowIdx < end_row; ++rowIdx) {
        print_row(rowIdx);
    }

    return stream.str();
}

std::string TableBatch::to_string() const {
    return to_string(0, num_rows(), 0, num_columns(), ", ");
}

std::string TableBatch::to_arrow_string() const {
    return table_->ToString();
}

void TableBatch::print() const {
    std::cout << to_string() << std::endl;
}

Status TableBatch::add_column(int32_t position,
                              const std::string &column_name,
                              std::shared_ptr<arrow::Array> input_column) {
    if (input_column->length() != table_->num_rows()) {
        return {maximus::ErrorCode::MaximusError,
                "New column length must match the number of rows in the table"};
    }
    auto field       = arrow::field(column_name, input_column->type());
    auto maybe_table = table_->AddColumn(position, std::move(field), std::move(input_column));
    if (!maybe_table.ok()) {
        return {maybe_table.status()};
    }
    table_ = maybe_table.ValueOrDie();
    return Status::OK();
}

const std::shared_ptr<maximus::MaximusContext> &TableBatch::get_context() const {
    return ctx_;
}

TableBatch::TableBatch(const std::shared_ptr<MaximusContext> &ctx, InternalTableBatchPtr tab)
        : ctx_(ctx), table_(std::move(tab)) {
}

void TableBatch::clear() {
    // CHECK_ARROW_STATUS(free_table(table_));
    table_.reset();
}

std::shared_ptr<Schema> TableBatch::get_schema() const {
    return std::make_shared<Schema>(table_->schema());
}

bool TableBatch::operator==(const TableBatch &other) const {
    if (table_ == other.table_) {
        return true;
    }
    if (!table_ || !other.table_) {
        return false;
    }
    // return table_->Equals(*other.table_);
    return to_string() == other.to_string();
}
InternalTableBatchPtr TableBatch::get_table_batch() const {
    return table_;
}

InternalTableBatchPtr TableBatch::get_table_batch() {
    return table_;
}

TableBatchPtr TableBatch::slice(int64_t offset, int64_t length) const {
    auto table = table_->Slice(offset, length);
    return std::make_shared<TableBatch>(ctx_, table);
}


TableBatchPtr TableBatch::clone(bool prefer_pinned_pool) const {
    assert(ctx_);
    ::arrow::MemoryPool *pool = ctx_->get_memory_pool();
    if (prefer_pinned_pool) {
        pool = ctx_->get_pinned_memory_pool_if_available();
    }
    auto out = arrow_clone(table_, pool);
    return std::make_shared<TableBatch>(ctx_, out);
}
}  // namespace maximus
