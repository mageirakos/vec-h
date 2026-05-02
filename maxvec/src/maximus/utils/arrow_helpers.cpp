#include <iostream>
#include <arrow/compute/api.h>
#include <maximus/error_handling.hpp>
#include <maximus/utils/arrow_helpers.hpp>
#include <random>

namespace maximus {

arrow::Result<std::shared_ptr<arrow::ArrayData>> arrow_clone_array_data(
    const std::shared_ptr<arrow::ArrayData> &source_data, arrow::MemoryPool *memory_pool) {
    if (!source_data) {
        return nullptr;
    }

    // Clone direct buffers
    std::vector<std::shared_ptr<arrow::Buffer>> cloned_buffers;
    cloned_buffers.reserve(source_data->buffers.size());
    for (const auto &buffer : source_data->buffers) {
        if (buffer) {
            auto buffer_copy = buffer->CopySlice(0, buffer->size(), memory_pool);
            check_status(buffer_copy.status());
            cloned_buffers.push_back(std::move(buffer_copy.ValueOrDie()));
        } else {
            cloned_buffers.push_back(nullptr);
        }
    }

    // Clone child data recursively
    std::vector<std::shared_ptr<arrow::ArrayData>> cloned_children;
    cloned_children.reserve(source_data->child_data.size());
    for (const auto &child_data : source_data->child_data) {
        auto cloned_child = arrow_clone_array_data(child_data, memory_pool);
        check_status(cloned_child.status());
        cloned_children.push_back(std::move(cloned_child.ValueOrDie()));
    }

    std::shared_ptr<arrow::ArrayData> cloned_dictionary = nullptr;
    if (source_data->dictionary) {
        auto dict_result = arrow_clone_array_data(source_data->dictionary, memory_pool);
        check_status(dict_result.status());
        cloned_dictionary = dict_result.ValueOrDie();
    }

    return arrow::ArrayData::Make(source_data->type,
                                  source_data->length,
                                  std::move(cloned_buffers),
                                  std::move(cloned_children),
                                  cloned_dictionary,
                                  source_data->null_count.load(),
                                  source_data->offset);
}

std::shared_ptr<arrow::RecordBatch> arrow_clone(
    const std::shared_ptr<arrow::RecordBatch> &input_batch, arrow::MemoryPool *memory_pool) {
    std::vector<std::shared_ptr<arrow::Array>> cloned_arrays;
    cloned_arrays.reserve(input_batch->num_columns());

    for (int i = 0; i < input_batch->num_columns(); ++i) {
        auto column       = input_batch->column(i);
        auto &column_data = column->data();

        std::vector<std::shared_ptr<arrow::Buffer>> copied_buffers;
        copied_buffers.reserve(column_data->buffers.size());

        for (auto &buffer : column_data->buffers) {
            if (buffer) {
                auto buffer_result = buffer->CopySlice(0, buffer->size(), memory_pool);
                if (!buffer_result.ok()) {
                    check_status(buffer_result.status());
                }
                copied_buffers.push_back(buffer_result.ValueOrDie());
            } else {
                copied_buffers.emplace_back(nullptr);
            }
        }

        auto new_array_data =
            arrow::ArrayData::Make(column->type(), column->length(), std::move(copied_buffers));
        cloned_arrays.emplace_back(arrow::MakeArray(new_array_data));
    }

    return arrow::RecordBatch::Make(
        input_batch->schema(), input_batch->num_rows(), std::move(cloned_arrays));
}

std::shared_ptr<arrow::Table> arrow_clone(const std::shared_ptr<arrow::Table> &source_table,
                                          arrow::MemoryPool *memory_pool) {
    std::vector<std::shared_ptr<arrow::ChunkedArray>> cloned_columns;
    cloned_columns.reserve(source_table->num_columns());

    for (int col_idx = 0; col_idx < source_table->num_columns(); ++col_idx) {
        auto chunked_array = source_table->column(col_idx);
        std::vector<std::shared_ptr<arrow::Array>> cloned_chunks;
        cloned_chunks.reserve(chunked_array->num_chunks());

        for (const auto &chunk : chunked_array->chunks()) {
            // Recursively clone the ArrayData for the chunk
            auto clone_result = arrow_clone_array_data(chunk->data(), memory_pool);
            check_status(clone_result.status());              // Check status
            auto new_array_data = clone_result.ValueOrDie();  // Get the cloned data

            cloned_chunks.push_back(arrow::MakeArray(new_array_data));
        }

        auto new_chunked_array =
            std::make_shared<arrow::ChunkedArray>(std::move(cloned_chunks), chunked_array->type());
        cloned_columns.emplace_back(std::move(new_chunked_array));
    }

    return arrow::Table::Make(source_table->schema(), std::move(cloned_columns));
}

int max_num_of_chunks(const std::shared_ptr<arrow::Table> &table) {
    int max_chunks = 0;

    // Iterate over each column in the table
    for (int i = 0; i < table->num_columns(); ++i) {
        // Get the chunked array for the column
        std::shared_ptr<arrow::ChunkedArray> chunked_array = table->column(i);

        // Get the number of chunks in this chunked array
        int num_chunks = chunked_array->num_chunks();

        max_chunks = std::max(max_chunks, num_chunks);
    }

    return max_chunks;
}

std::shared_ptr<arrow::Table> concatenate_chunks(std::shared_ptr<arrow::Table> table,
                                                 arrow::MemoryPool *pool) {
    assert(pool);

    int num_chunks = max_num_of_chunks(table);

    // if there is only one chunk, and the pinned memory usage is enforced, copy the table to pinned memory
    if (num_chunks == 1) {
        auto new_table = arrow_clone(table, pool);
        return std::move(new_table);
    }

    // otherwise, combine chunks will anyway perform a full copy, using the provided memory pool
    auto maybe_batch = table->CombineChunksToBatch(pool);
    if (!maybe_batch.ok()) {
        check_status(maybe_batch.status());
    }
    auto batch       = std::move(maybe_batch).ValueOrDie();
    auto maybe_table = arrow::Table::FromRecordBatches({std::move(batch)});
    if (!maybe_table.ok()) {
        check_status(maybe_table.status());
    }
    return std::move(maybe_table).ValueOrDie();
}

std::shared_ptr<arrow::Table> arrow_clone_to_single_chunk(
    const std::shared_ptr<arrow::Table> &table, arrow::MemoryPool *pool) {
    return concatenate_chunks(table, pool);
}

std::shared_ptr<arrow::RecordBatch> to_record_batch(std::shared_ptr<arrow::Table> table,
                                                    arrow::MemoryPool *pool) {
    auto table_reader  = arrow::TableBatchReader(table);
    auto maybe_batches = table_reader.ToRecordBatches();

    if (!maybe_batches.ok()) {
        check_status(maybe_batches.status());
    }

    auto batches = maybe_batches.ValueOrDie();

    if (batches.size() == 0) {
        return nullptr;
    }

    if (batches.size() == 1) {
        return std::move(batches[0]);
    }

    assert(pool);
    auto maybe_batch = table->CombineChunksToBatch(pool);
    if (!maybe_batch.ok()) {
        check_status(maybe_batch.status());
    }
    auto batch = std::move(maybe_batch.ValueOrDie());
    return std::move(batch);
}

std::shared_ptr<arrow::RecordBatch> to_record_batch(
    std::vector<std::shared_ptr<arrow::RecordBatch>> &batches, arrow::MemoryPool *pool) {
    if (batches.size() == 0) {
        return nullptr;
    }
    if (batches.size() == 1) {
        return std::move(batches[0]);
    }

    auto maybe_table = arrow::Table::FromRecordBatches(batches);
    if (!maybe_table.ok()) {
        check_status(maybe_table.status());
    }

    auto table = std::move(maybe_table.ValueOrDie());

    assert(pool);

    auto maybe_batch = table->CombineChunksToBatch(pool);
    if (!maybe_batch.ok()) {
        check_status(maybe_batch.status());
    }
    auto batch = std::move(maybe_batch.ValueOrDie());
    return std::move(batch);
}

std::shared_ptr<arrow::Table> to_table(std::vector<std::shared_ptr<arrow::RecordBatch>> &batches,
                                       arrow::MemoryPool *pool) {
    //
    // Convert the RecordBatch to a Table
    std::shared_ptr<arrow::Table> table;
    auto maybe_table = arrow::Table::FromRecordBatches(batches);
    if (!maybe_table.ok()) {
        check_status(maybe_table.status());
    }
    return maybe_table.ValueOrDie();
}

// Helper function template for numeric types
template<typename TYPE>
std::string to_string_numeric(const std::shared_ptr<arrow::Array> &array, int index) {
    auto casted_array = std::static_pointer_cast<arrow::NumericArray<TYPE>>(array);
    return casted_array->IsNull(index) ? "" : std::to_string(casted_array->Value(index));
}

std::string to_string_list(const std::shared_ptr<arrow::Array> &array, int index) {
    if (array->IsNull(index)) {
        return "";
    }

    // Handle FixedSizeListArray, ListArray, and LargeListArray generically
    std::shared_ptr<arrow::Array> values;
    int64_t offset = 0;
    int64_t length = 0;

    switch (array->type_id()) {
        case arrow::Type::FIXED_SIZE_LIST: {
            auto list_array = std::static_pointer_cast<arrow::FixedSizeListArray>(array);
            values = list_array->values();
            int32_t list_size = list_array->list_type()->list_size();
            offset = index * list_size;
            length = list_size;
            break;
        }
        case arrow::Type::LIST: {
            auto list_array = std::static_pointer_cast<arrow::ListArray>(array);
            values = list_array->values();
            offset = list_array->value_offset(index);
            length = list_array->value_length(index);
            break;
        }
        case arrow::Type::LARGE_LIST: {
            auto list_array = std::static_pointer_cast<arrow::LargeListArray>(array);
            values = list_array->values();
            offset = static_cast<int64_t>(list_array->value_offset(index));
            length = static_cast<int64_t>(list_array->value_length(index));
            break;
        }
        default:
            throw std::invalid_argument("Array is not a ListArray, LargeListArray, or FixedSizeListArray");
    }

    // Format result (sample a few elements for brevity)
    std::string result = "[";
    if (length > 0)
        result += array_string(values, offset);
    if (length > 1)
        result += ", " + array_string(values, offset + 1);
    if (length > 3)
        result += ", ...";
    if (length > 2)
        result += ", " + array_string(values, offset + length - 1);
    result += "]";
    return result;
}

std::string to_string_bool(const std::shared_ptr<arrow::Array> &array, int index) {
    auto casted_array = std::static_pointer_cast<arrow::BooleanArray>(array);
    if (casted_array->IsNull(index)) {
        return "";
    }
    return casted_array->Value(index) ? "true" : "false";
}

// Type mapping using a function pointer approach
using ToStringFunc = std::function<std::string(const std::shared_ptr<arrow::Array> &, int)>;

std::unordered_map<arrow::Type::type, ToStringFunc> create_to_string_map() {
    return {{arrow::Type::UINT8, to_string_numeric<arrow::UInt8Type>},
            {arrow::Type::INT8, to_string_numeric<arrow::Int8Type>},
            {arrow::Type::UINT16, to_string_numeric<arrow::UInt16Type>},
            {arrow::Type::INT16, to_string_numeric<arrow::Int16Type>},
            {arrow::Type::UINT32, to_string_numeric<arrow::UInt32Type>},
            {arrow::Type::INT32, to_string_numeric<arrow::Int32Type>},
            {arrow::Type::UINT64, to_string_numeric<arrow::UInt64Type>},
            {arrow::Type::INT64, to_string_numeric<arrow::Int64Type>},
            {arrow::Type::HALF_FLOAT, to_string_numeric<arrow::HalfFloatType>},
            {arrow::Type::FLOAT, to_string_numeric<arrow::FloatType>},
            {arrow::Type::DOUBLE, to_string_numeric<arrow::DoubleType>},
            {arrow::Type::DATE32, to_string_numeric<arrow::Date32Type>},
            {arrow::Type::DATE64, to_string_numeric<arrow::Date64Type>},
            {arrow::Type::TIMESTAMP, to_string_numeric<arrow::TimestampType>},
            {arrow::Type::TIME32, to_string_numeric<arrow::Time32Type>},
            {arrow::Type::TIME64, to_string_numeric<arrow::Time64Type>},
            {arrow::Type::STRING,
             [](const std::shared_ptr<arrow::Array> &array, int index) {
                 return std::static_pointer_cast<arrow::StringArray>(array)->GetString(index);
             }},
            {arrow::Type::FIXED_SIZE_LIST, to_string_list},
            {arrow::Type::LIST, to_string_list},
            {arrow::Type::LARGE_LIST, to_string_list},
            {arrow::Type::BOOL, to_string_bool}};
}

std::string array_string(const std::shared_ptr<arrow::Array> &array, int64_t index) {
    static const auto to_string_map = create_to_string_map();

    if (!array) return "NA";

    auto it = to_string_map.find(array->type()->id());
    if (it != to_string_map.end()) {
        return it->second(array, index);
    }

    return "NA";  // Default case for unsupported types
}

std::shared_ptr<arrow::RecordBatch> generate_batch(const arrow::FieldVector &fields,
                                                   int64_t size,
                                                   int seed,
                                                   arrow::MemoryPool *memory_pool) {
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> float_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<double> double_dist(-1.0, 1.0);
    std::uniform_int_distribution<int> char_dist('a', 'z');
    std::uniform_int_distribution<int> length_dist(5, 20);

    std::uniform_int_distribution<int> binary_length_dist(
        10, 50);  // Random binary length between 10 and 50// Random string length between 5 and 20

    for (const auto &field : fields) {
        auto type = field->type()->id();
        std::shared_ptr<arrow::Array> array;
        switch (type) {
            case arrow::Type::BOOL: {
                arrow::BooleanBuilder builder(memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    bool value = rng() % 2;
                    check_status(builder.Append(value));
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::UINT8: {
                arrow::UInt8Builder builder(memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    check_status(builder.Append(rng() % 256));
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::INT8: {
                arrow::Int8Builder builder(memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    check_status(builder.Append(rng() % 256 - 128));
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::UINT16: {
                arrow::UInt16Builder builder(memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    check_status(builder.Append(rng() % 65536));
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::INT16: {
                arrow::Int16Builder builder(memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    check_status(builder.Append(rng() % 65536 - 32768));
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::UINT32: {
                arrow::UInt32Builder builder(memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    check_status(builder.Append(rng()));
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::INT32: {
                arrow::Int32Builder builder(memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    check_status(builder.Append(static_cast<int32_t>(rng())));
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::UINT64: {
                arrow::UInt64Builder builder(memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    check_status(builder.Append(static_cast<uint64_t>(rng())));
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::INT64: {
                arrow::Int64Builder builder(memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    check_status(builder.Append(static_cast<int64_t>(rng())));
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::HALF_FLOAT: {
                arrow::HalfFloatBuilder builder(memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    check_status(builder.Append(float_dist(rng)));
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::FLOAT: {
                arrow::FloatBuilder builder(memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    check_status(builder.Append(float_dist(rng)));
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::DOUBLE: {
                arrow::DoubleBuilder builder(memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    check_status(builder.Append(double_dist(rng)));
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::DATE32: {
                arrow::Date32Builder builder(memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    check_status(builder.Append(rng() % (365 * 50)));
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::DATE64: {
                arrow::Date64Builder builder(memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    check_status(builder.Append(rng() % (365LL * 50 * 24 * 60 * 60 * 1000)));
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::TIMESTAMP: {
                arrow::TimestampBuilder builder(field->type(), memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    check_status(builder.Append(rng()));
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::STRING: {
                arrow::StringBuilder builder(memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    int length = length_dist(rng);  // Random string length between 5 and 20
                    std::string value;
                    value.reserve(length);
                    for (int j = 0; j < length; ++j) {
                        value.push_back(static_cast<char>(char_dist(rng)));  // Random char
                    }
                    check_status(builder.Append(value));  // Append as a string
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::FIXED_SIZE_BINARY: {
                int64_t byte_size =
                    field->type()->byte_width();  // Get the fixed size of the binary field
                arrow::FixedSizeBinaryBuilder builder(field->type(), memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    std::string value(byte_size, ' ');
                    for (int j = 0; j < byte_size; ++j) {
                        value[j] = static_cast<char>(rng() % 256);  // Random byte between 0 and 255
                    }
                    check_status(builder.Append(value));
                }
                check_status(builder.Finish(&array));
                break;
            }
            case arrow::Type::BINARY: {
                arrow::BinaryBuilder builder(memory_pool);
                for (int64_t i = 0; i < size; ++i) {
                    int length =
                        binary_length_dist(rng);  // Random binary length between 10 and 50 bytes
                    std::string value;
                    value.resize(length);  // Resize string to the random length
                    for (int j = 0; j < length; ++j) {
                        value[j] = static_cast<char>(rng() % 256);  // Random byte between 0 and 255
                    }
                    check_status(builder.Append(value));
                }
                check_status(builder.Finish(&array));
                break;
            }
            default:
                throw std::runtime_error("Type not yet implemented");
        }
        arrays.push_back(array);
    }
    return arrow::RecordBatch::Make(arrow::schema(fields), size, arrays);
}

Status cast_table_to_schema(std::shared_ptr<arrow::Table> &table,
                            std::shared_ptr<arrow::Schema> new_schema) {
    if (!table) {
        return Status(ErrorCode::MaximusError, "Casting failed: the table is NULL");
    }
    if (!new_schema) {
        return Status(ErrorCode::MaximusError, "Casting failed: the schema is NULL");
    }

    std::vector<std::shared_ptr<arrow::ChunkedArray>> new_columns;
    arrow::compute::CastOptions cast_options;  // Default cast options

    for (int i = 0; i < new_schema->num_fields(); ++i) {
        auto field = new_schema->field(i);
        auto column = table->GetColumnByName(field->name());

        if (!column) {
            return Status(ErrorCode::MaximusError,
                          "Column " + field->name() + " not found in the table.");
        }

        // Check if type already matches (zero-copy case)
        if (column->type()->Equals(field->type())) {
            new_columns.push_back(column);  // Reuse the existing column
            continue;
        }

        // Cast each chunk of the ChunkedArray
        std::vector<std::shared_ptr<arrow::Array>> casted_chunks;
        for (const auto& chunk : column->chunks()) {
            arrow::Result<std::shared_ptr<arrow::Array>> cast_result =
                arrow::compute::Cast(*chunk, field->type(), cast_options);

            if (!cast_result.ok()) {
                return Status(ErrorCode::MaximusError,
                              "Casting failed for column: " + field->name());
            }

            casted_chunks.push_back(cast_result.ValueOrDie());
        }

        // Create a new ChunkedArray from the casted chunks
        new_columns.push_back(std::make_shared<arrow::ChunkedArray>(casted_chunks, field->type()));
    }

    // Create a new table with the updated schema
    table = arrow::Table::Make(new_schema, new_columns);
    return Status::OK();
}

Status cast_record_batch_to_schema(std::shared_ptr<arrow::RecordBatch> &batch,
                                   std::shared_ptr<arrow::Schema> new_schema) {
    if (!batch) {
        return Status(ErrorCode::MaximusError, "Casting failed: the batch is NULL");
    }
    if (!new_schema) {
        return Status(ErrorCode::MaximusError, "Casting failed: the schema is NULL");
    }

    std::vector<std::shared_ptr<arrow::Array>> new_columns;

    for (int i = 0; i < new_schema->num_fields(); ++i) {
        auto field = new_schema->field(i);
        auto column = batch->GetColumnByName(field->name());

        if (!column) {
            return Status(ErrorCode::MaximusError,
                          "Column " + field->name() + " not found in the batch.");
        }

        // Check if type already matches (zero-copy case)
        if (column->type()->Equals(field->type())) {
            new_columns.push_back(column);  // Reuse the existing column
            continue;
        }

        // Perform casting if types differ (not zero-copy)
        arrow::Result<std::shared_ptr<arrow::Array>> cast_result =
            arrow::compute::Cast(*column, field->type());

        if (!cast_result.ok()) {
            return Status(ErrorCode::MaximusError, "Casting failed for column: " + field->name());
        }

        new_columns.push_back(cast_result.ValueOrDie());
    }

    // Create a new RecordBatch with the updated schema
    batch = arrow::RecordBatch::Make(new_schema, batch->num_rows(), new_columns);
    return Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Table>> concatenate_nonempty_tables(
    const std::vector<std::shared_ptr<arrow::Table>>& tables,
    arrow::MemoryPool* pool,
    const arrow::ConcatenateTablesOptions& options) {

    std::vector<std::shared_ptr<arrow::Table>> nonempty_tables;
    nonempty_tables.reserve(tables.size());

    for (const auto& t : tables) {
        if (t && t->num_rows() > 0) {
            nonempty_tables.push_back(t);
        }
    }

    if (nonempty_tables.empty()) {
        // Return an empty table with the schema of the first input (if available)
        if (!tables.empty() && tables[0]) {
            return arrow::Table::MakeEmpty(tables[0]->schema(), pool);
        } else {
            return arrow::Status::Invalid("No tables provided to concatenate_nonempty_tables()");
        }
    }

    if (nonempty_tables.size() == 1) {
        // Only one non-empty table - no need to concatenate
        return nonempty_tables.front();
    }

    // Concatenate the remaining non-empty tables
    return arrow::ConcatenateTables(nonempty_tables, options, pool);
}

}  // namespace maximus
