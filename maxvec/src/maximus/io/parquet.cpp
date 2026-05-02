#include <numeric>

#include <arrow/compute/cast.h>
#include <parquet/arrow/reader.h>

#include <maximus/io/file.hpp>
#include <maximus/io/parquet.hpp>

namespace maximus {
Status streaming_reader_parquet(const std::shared_ptr<MaximusContext> &ctx,
                                const std::string &path,
                                const std::shared_ptr<Schema> &schema,
                                std::vector<std::string> include_columns,
                                std::shared_ptr<arrow::RecordBatchReader> &reader) {
    assert(ends_with(path, ".parquet") && "streaming_reader_parquet only supports parquet files");

    assert(ctx);
    auto pool = ctx->get_memory_pool();
    assert(pool);

    // Configure general Parquet reader settings
    auto reader_properties = parquet::ReaderProperties(pool);
    reader_properties.set_buffer_size(4096 * 4);
    reader_properties.enable_buffered_stream();

    // Configure Arrow-specific Parquet reader settings
    auto arrow_reader_props = parquet::ArrowReaderProperties();
    arrow_reader_props.set_batch_size(128 * 1024);  // default 64 * 1024
    arrow_reader_props.set_use_threads(true);

    parquet::arrow::FileReaderBuilder reader_builder;
    auto status = reader_builder.OpenFile(path, /*memory_map=*/false, reader_properties);
    if (!status.ok()) {
        return Status(ErrorCode::ArrowError, status.message());
    }
    reader_builder.memory_pool(pool);
    reader_builder.properties(arrow_reader_props);

    /*
    auto maybe_arrow_reader = reader_builder.Build();
    if (!maybe_arrow_reader.ok()) {
        return Status(ErrorCode::ArrowError, maybe_arrow_reader.status().message());
    }
    std::unique_ptr<parquet::arrow::FileReader> arrow_reader =
        std::move(maybe_arrow_reader.ValueOrDie());

    status = arrow_reader->GetRecordBatchReader(&reader);
    if (!status.ok()) {
        return Status(ErrorCode::ArrowError, status.message());
    }
    return Status::OK();
    */
    auto maybe_arrow_reader = reader_builder.Build();
    if (!maybe_arrow_reader.ok()) {
        return Status(ErrorCode::ArrowError, maybe_arrow_reader.status().message());
    }

    std::unique_ptr<parquet::arrow::FileReader> arrow_reader =
        std::move(maybe_arrow_reader).ValueOrDie();

    // Use the new API: returns Result<std::unique_ptr<RecordBatchReader>>
    auto maybe_reader = arrow_reader->GetRecordBatchReader();
    if (!maybe_reader.ok()) {
        return Status(ErrorCode::ArrowError, maybe_reader.status().message());
    }

    // Convert unique_ptr -> shared_ptr for compatibility with old code
    reader = std::move(maybe_reader).ValueOrDie();

    return Status::OK();
}

Status read_parquet(const std::shared_ptr<MaximusContext> &ctx,
                    const std::string &path,
                    const std::shared_ptr<Schema> &schema,
                    std::vector<std::string> include_columns,
                    TablePtr &tableOut) {
    assert(ends_with(path, ".parquet") && "read_parquet(...) only supports parquet files");

    assert(ctx && "The context must not be null in Table::from_parquet.");
    auto pool = ctx->get_memory_pool();
    assert(pool && "The memory pool must not be null in Table::from_parquet.");

    std::shared_ptr<arrow::io::RandomAccessFile> input;
    auto status = input_file(ctx, path, input);
    if (!status.ok()) {
        return status;
    }

    auto maybe_reader = parquet::arrow::OpenFile(input, pool);
    if (!maybe_reader.ok()) {
        CHECK_STATUS(maybe_reader.status());
    }
    auto reader = std::move(maybe_reader.ValueOrDie());

    // Map column names to indices
    std::shared_ptr<arrow::Schema> file_schema;
    CHECK_STATUS(reader->GetSchema(&file_schema));

    std::vector<int> column_indices;

    for (const auto &col_name : include_columns) {
        auto idx = file_schema->GetFieldIndex(col_name);
        if (idx == -1) {
            return Status(ErrorCode::ArrowError, "Column not found in schema: " + col_name);
        }
        column_indices.push_back(idx);
    }

    if (column_indices.size() == 0) {
        assert(include_columns.empty());
        // if the column_indices are empty, it means the include columns were empty
        // in that case, we add all the columns
        // the column_indices should contain values 0, 1, 2, ..., num_fields-1
        column_indices.resize(file_schema->num_fields());
        std::iota(column_indices.begin(), column_indices.end(), 0);

        // in this special case, add all these fields to the include_columns vector
        for (const auto& field : file_schema->fields()) {
            include_columns.push_back(field->name());
        }

        assert(column_indices.size() == include_columns.size());
    }

    std::shared_ptr<arrow::Table> selected_table;
    CHECK_STATUS(reader->ReadTable(column_indices, &selected_table));

    // now look at the input schema and the include_columns
    // and create the target schema that only contains the selected columns
    // moreover, relax the schema to allow for nullability
    // iterate over all fields in the selected table and create a new schema
    std::vector<std::shared_ptr<arrow::Field>> fields;
    fields.reserve(selected_table->num_columns());
    for (const auto& field : selected_table->fields()) {
        assert(field);
        if (!schema) {
            fields.push_back(field->WithNullable(true));
        } else {
            assert(schema);
            assert(schema->get_schema());
            auto schema_field = schema->get_schema()->GetFieldByName(field->name());
            assert(schema_field);
            fields.push_back(schema_field->WithNullable(true));
        }
    }

    auto target_schema = std::make_shared<arrow::Schema>(fields);

    auto maybe_table = arrow::PromoteTableToSchema(selected_table, target_schema, pool);
    CHECK_STATUS(maybe_table.status());
    auto table = maybe_table.ValueOrDie();
    assert(table);

    // Restore the original schema's nullability
    if (schema) {
        std::vector<std::shared_ptr<arrow::Field>> correct_fields;
        for (const auto& field : table->fields()) {
            auto schema_field = schema->get_schema()->GetFieldByName(field->name());
            if (schema_field) {
                correct_fields.push_back(field->WithType(schema_field->type())->WithNullable(schema_field->nullable()));
            } else {
                correct_fields.push_back(field);
            }
        }
        auto correct_schema = std::make_shared<arrow::Schema>(correct_fields);
        table = arrow::Table::Make(correct_schema, table->columns());
    }

    tableOut = std::make_shared<Table>(ctx, table);
    assert(tableOut);
    return Status::OK();
}

Status read_parquet(const std::shared_ptr<MaximusContext> &ctx,
                    const std::string &path,
                    TablePtr &tableOut) {
    return read_parquet(ctx, path, nullptr, {}, tableOut);
}
}  // namespace maximus
