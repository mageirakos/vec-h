#include <arrow/table.h>

#include <iostream>
#include <maximus/io/csv.hpp>
#include <maximus/io/parquet.hpp>
#include <maximus/types/table.hpp>
#include <maximus/types/table_batch.hpp>
#include <maximus/utils/arrow_helpers.hpp>
#include <maximus/utils/json_helpers.hpp>
#include <maximus/utils/utils.hpp>
#include <memory>

namespace maximus {

Status Table::from_table(const std::shared_ptr<MaximusContext> &ctx,
                         InternalTablePtr table,
                         TablePtr &tableOut) {
    const auto &schema = table->schema();
    auto type_status   = are_types_supported(schema);
    if (!type_status.ok()) {
        return type_status;
    }
    tableOut = std::make_shared<Table>(ctx, std::move(table));
    return Status::OK();
}

Status Table::from_table_batches(const std::shared_ptr<MaximusContext> &ctx,
                                 std::vector<TableBatchPtr> table_batches,
                                 TablePtr &tableOut) {
    std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches(table_batches.size());
    assert(table_batches.size() == record_batches.size() &&
           "table_batches and record_batches should have the same size");
    for (unsigned int i = 0; i < table_batches.size(); i++) {
        auto status = table_batches[i]->to_record_batch(record_batches[i]);
        if (!status.ok()) {
            return status;
        }
    }
    auto maybe_table = InternalTable::FromRecordBatches(std::move(record_batches));

    if (!maybe_table.ok()) {
        return {maybe_table.status()};
    }
    tableOut = std::make_shared<Table>(ctx, maybe_table.ValueOrDie());
    return Status::OK();
}

Status Table::from_json(const std::shared_ptr<MaximusContext> &ctx,
                        const std::shared_ptr<Schema> &schema,
                        const std::vector<std::string> &json,
                        TablePtr &tableOut) {
    auto table = table_from_json(schema->get_schema(), json);
    return Table::from_table(ctx, std::move(table), tableOut);
}

Status Table::from_csv(const std::shared_ptr<MaximusContext> &ctx,
                       const std::string &path,
                       const std::shared_ptr<Schema> &schema,
                       const std::vector<std::string> &include_columns,
                       TablePtr &tableOut) {
    return read_csv(ctx, path, schema, include_columns, tableOut);
}

Status Table::from_parquet(const std::shared_ptr<MaximusContext> &ctx,
                           const std::string &path,
                           const std::shared_ptr<Schema> &schema,
                           const std::vector<std::string> &include_columns,
                           TablePtr &tableOut) {
    return read_parquet(ctx, path, schema, include_columns, tableOut);
}

int Table::num_columns() const {
    if (table_) return table_->num_columns();
    return 0;
}

std::vector<std::string> Table::column_names() {
    std::vector<std::string> names;
    if (table_) {
        return table_->ColumnNames();
    }
    return {};
}

int64_t Table::num_rows() const {
    if (table_) return table_->num_rows();
    return 0;
}

bool Table::empty() const {
    if (!table_) {
        return true;
    }
    return table_->num_rows() == 0;
}

std::string Table::to_string(
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
            // std::cout << "Printing row " << row << ", column " << col << std::endl;
            auto col_data      = table_->column(col);
            int64_t row_offset = 0;
            for (int chunk_idx = 0; chunk_idx < col_data->num_chunks(); ++chunk_idx) {
                auto chunk = col_data->chunk(chunk_idx);
                if (row_offset <= row && row < row_offset + chunk->length()) {
                    stream << maximus::array_string(chunk, row - row_offset);
                    if (col < end_col - 1) stream << delim;
                    break;
                }
                row_offset += chunk->length();
            }
        }
        stream << '\n';
    };

    for (int64_t rowIdx = first_row; rowIdx < end_row; ++rowIdx) {
        print_row(rowIdx);
    }

    return stream.str();
}

std::string Table::to_string() const {
    return to_string(0, num_rows(), 0, num_columns(), ", ");
}

std::string Table::to_arrow_string() const {
    return table_->ToString();
}

void Table::print() const {
    std::cout << to_string() << std::endl;
}

Status Table::add_column(const int32_t position,
                         const std::string &column_name,
                         std::shared_ptr<arrow::Array> input_column) {
    assert(table_);
    if (input_column->length() != table_->num_rows()) {
        return {maximus::ErrorCode::MaximusError,
                "New column length must match the number of rows in the table"};
    }
    auto field         = arrow::field(column_name, input_column->type());
    auto chunked_array = std::make_shared<arrow::ChunkedArray>(std::move(input_column));
    auto maybe_table   = table_->AddColumn(position, std::move(field), std::move(chunked_array));
    if (!maybe_table.ok()) {
        return {ErrorCode::ArrowError, maybe_table.status().message()};
    }
    table_ = maybe_table.ValueOrDie();
    return Status::OK();
}

Status Table::add_const_column(const int32_t position,
                               const std::string &column_name,
                               const int32_t value) {
    assert(ctx_);

    auto num_rows    = this->num_rows();
    auto num_columns = this->num_columns();

    // create a new column with the same value
    arrow::Int32Builder builder(arrow::int32(), ctx_->get_memory_pool());
    auto arrow_status = builder.AppendValues(std::vector<int32_t>(num_rows, value));
    if (!arrow_status.ok()) {
        return Status(ErrorCode::ArrowError, arrow_status.message());
    }
    std::shared_ptr<arrow::Array> new_column;
    arrow_status = builder.Finish(&new_column);
    if (!arrow_status.ok()) {
        return Status(ErrorCode::ArrowError, arrow_status.message());
    }

    auto maximus_status = add_column(num_columns, column_name, new_column);
    if (!maximus_status.ok()) {
        return maximus_status;
    }

    return Status::OK();
}

Status Table::remove_column(const int32_t position) {
    assert(table_);
    assert(position < table_->num_columns());

    auto maybe_table = table_->RemoveColumn(position);
    if (!maybe_table.status().ok()) {
        return Status(ErrorCode::ArrowError, maybe_table.status().message());
    }

    table_ = maybe_table.ValueOrDie();

    return Status::OK();
}

const std::shared_ptr<maximus::MaximusContext> &Table::get_context() const {
    return ctx_;
}

Table::Table(const std::shared_ptr<MaximusContext> &ctx, InternalTablePtr tab)
        : ctx_(ctx), table_(std::move(tab)) {
}

void Table::clear() {
    // CHECK_ARROW_STATUS(free_table(table_));
    table_.reset();
}

std::shared_ptr<Schema> Table::get_schema() const {
    return std::make_shared<Schema>(table_->schema());
}

bool Table::operator==(const Table &other) const {
    if (table_ == other.table_) {
        return true;
    }
    if (!table_ || !other.table_) {
        return false;
    }
    return table_->Equals(*other.table_);
    // return to_string() == other.to_string();
}

InternalTablePtr Table::get_table() const {
    return table_;
}

InternalTablePtr Table::get_table() {
    return table_;
}

TablePtr Table::slice(int64_t offset, int64_t length) const {
    assert(table_);
    auto sliced_table = table_->Slice(offset, length);
    return std::make_shared<Table>(ctx_, sliced_table);
}


TablePtr Table::clone(bool prefer_pinned_pool, bool to_single_chunk) const {
    assert(ctx_);
    ::arrow::MemoryPool *pool = ctx_->get_memory_pool();
    if (prefer_pinned_pool) {
        pool = ctx_->get_pinned_memory_pool_if_available();
    }
    std::shared_ptr<arrow::Table> out;
    if (to_single_chunk) {
        out = arrow_clone_to_single_chunk(table_, pool);
    } else {
        out = arrow_clone(table_, pool);
    }
    return std::make_shared<Table>(ctx_, out);
}

TablePtr Table::select_columns(const std::vector<std::string> &column_names) const {
    assert(table_);

    std::vector<int> indices;
    indices.reserve(column_names.size());
    for (const auto &name : column_names) {
        auto maybe_index = table_->schema()->GetFieldIndex(name);
        if (maybe_index == -1) {
            // Handle error: column name not found
            throw std::runtime_error("Column name " + name + " not found in table schema.");
        }
        indices.push_back(maybe_index);
    }

    std::vector<std::shared_ptr<arrow::ChunkedArray>> selected_columns;
    std::vector<std::shared_ptr<arrow::Field>> selected_fields;

    for (int index : indices) {
        selected_columns.push_back(table_->column(index));
        selected_fields.push_back(table_->field(index));
    }

    assert(selected_columns.size() > 0);
    assert(selected_fields.size() > 0);

    auto new_schema = std::make_shared<arrow::Schema>(selected_fields);
    auto new_table  = arrow::Table::Make(new_schema, selected_columns);

    return std::make_shared<Table>(ctx_, std::move(new_table));
}

}  // namespace maximus
