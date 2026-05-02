#pragma once
#include <arrow/c/bridge.h>

#include <cudf/interop.hpp>
#include <maximus/gpu/cudf/cudf_types.hpp>
#include <maximus/types/schema.hpp>
#include <maximus/utils/arrow_helpers.hpp>

namespace maximus {

// Convert Arrow RecordBatch to cuDF Table using Arrow C Interface
std::unique_ptr<cudf::table> cudf_from_arrow_batch(std::shared_ptr<arrow::RecordBatch> arrow_batch,
                                                   std::shared_ptr<Schema> schema,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr,
                                                   arrow::MemoryPool* pool);

std::unique_ptr<cudf::table> cudf_from_arrow_table(std::shared_ptr<arrow::Table> arrow_table,
                                                   std::shared_ptr<Schema> schema,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr,
                                                   arrow::MemoryPool* pool);

// Convert cuDF Table to Arrow RecordBatch using Arrow C Interface
std::shared_ptr<arrow::RecordBatch> cudf_to_arrow_batch(cudf::table_view cudf_table,
                                                        std::shared_ptr<Schema> schema,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr,
                                                        arrow::MemoryPool* pool);

// Convert cuDF Table to Arrow Table using Arrow C Interface
std::shared_ptr<arrow::Table> cudf_to_arrow_table(cudf::table_view cudf_table,
                                                  std::shared_ptr<Schema> schema,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr,
                                                  arrow::MemoryPool* pool);

}  // namespace maximus
