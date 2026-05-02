#pragma once

#include <arrow/api.h>

namespace maximus {
std::shared_ptr<arrow::RecordBatch> arrow_clone(const std::shared_ptr<arrow::RecordBatch>& rb,
                                                arrow::MemoryPool* pool);

std::shared_ptr<arrow::Table> arrow_clone(const std::shared_ptr<arrow::Table>& table,
                                          arrow::MemoryPool* pool);

std::shared_ptr<arrow::Table> arrow_clone_to_single_chunk(
    const std::shared_ptr<arrow::Table>& table, arrow::MemoryPool* pool);

std::shared_ptr<arrow::RecordBatch> to_record_batch(std::shared_ptr<arrow::Table> table,
                                                    arrow::MemoryPool* pool);

std::shared_ptr<arrow::RecordBatch> to_record_batch(
    std::vector<std::shared_ptr<arrow::RecordBatch>>& batches, arrow::MemoryPool* pool);

std::shared_ptr<arrow::Table> to_table(std::vector<std::shared_ptr<arrow::RecordBatch>>& batches,
                                       arrow::MemoryPool* pool);

// converts an arrow::Array value at given index to string
std::string array_string(const std::shared_ptr<arrow::Array>& array, int64_t index);

std::shared_ptr<arrow::RecordBatch> generate_batch(
    const arrow::FieldVector& fields,
    int64_t size,
    int seed,
    arrow::MemoryPool* memory_pool = arrow::default_memory_pool());

Status cast_table_to_schema(std::shared_ptr<arrow::Table>& table,
                            std::shared_ptr<arrow::Schema> new_schema);

Status cast_record_batch_to_schema(std::shared_ptr<arrow::RecordBatch>& batch,
                                   std::shared_ptr<arrow::Schema> new_schema);

template<typename T>
std::shared_ptr<T> arrow_array(const std::shared_ptr<arrow::ChunkedArray>& array) {
    static_assert(std::is_base_of<arrow::Array, T>::value, "T must be a subclass of arrow::Array");
    auto result = arrow::Concatenate(array->chunks(), arrow::default_memory_pool());
    CHECK_STATUS(result.status());
    return std::static_pointer_cast<T>(result.ValueUnsafe());
}

// Utility to concatenate only non-empty Arrow tables.
arrow::Result<std::shared_ptr<arrow::Table>> concatenate_nonempty_tables(
    const std::vector<std::shared_ptr<arrow::Table>>& tables,
    arrow::MemoryPool* pool = arrow::default_memory_pool(),
    const arrow::ConcatenateTablesOptions& options = arrow::ConcatenateTablesOptions::Defaults());

}  // namespace maximus
