#include <maximus/gpu/cudf/interop.hpp>
#include <maximus/profiler/profiler.hpp>

namespace maximus {

// Convert Arrow RecordBatch to cuDF Table using Arrow C Interface
std::unique_ptr<cudf::table> cudf_from_arrow_batch(std::shared_ptr<arrow::RecordBatch> arrow_batch,
                                                   std::shared_ptr<Schema> schema,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr,
                                                   arrow::MemoryPool* pool) {
    if (!arrow_batch) return nullptr;

    assert(pool);
    assert(schema);
    assert(arrow_batch);

    assert(arrow_batch->schema()->Equals(schema->get_schema()));

    // Convert the record batch to ArrowArray (C Data Interface)
    struct ArrowArray c_array;
    struct ArrowSchema c_schema;

    CHECK_STATUS(arrow::ExportRecordBatch(*arrow_batch, &c_array, &c_schema));

    // Convert from Arrow C Interface to cuDF Table
    auto cudf_table = cudf::from_arrow(&c_schema, &c_array, stream, mr);

    c_array.release(&c_array);
    c_schema.release(&c_schema);

    return cudf_table;
}

std::unique_ptr<cudf::table> cudf_from_arrow_table(std::shared_ptr<arrow::Table> arrow_table,
                                                   std::shared_ptr<Schema> schema,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr,
                                                   arrow::MemoryPool* pool) {
    if (!arrow_table) return nullptr;

    // convert the table to a record batch
    PE("to_record_batch");
    auto batch = to_record_batch(arrow_table, pool);
    PL("to_record_batch");

    PE("cudf::from_arrow");
    auto result = cudf_from_arrow_batch(batch, schema, stream, mr, pool);
    PL("cudf::from_arrow");

    return result;
}

// Convert cuDF Table to Arrow RecordBatch using Arrow C Interface
std::shared_ptr<arrow::RecordBatch> cudf_to_arrow_batch(cudf::table_view cudf_table,
                                                        std::shared_ptr<Schema> schema,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr,
                                                        arrow::MemoryPool* pool) {
    assert(schema);
    assert(pool);

    // Convert cuDF Table to Arrow C Interface
    auto column_metadata = gpu::to_cudf_column_metadata(schema->get_schema());
    auto cudf_schema     = cudf::to_arrow_schema(cudf_table, column_metadata);

    auto cudf_device_array = cudf::to_arrow_host(cudf_table, stream, mr);

    // Convert Arrow C Interface to Arrow Table
    auto maybe_arrow_batch = arrow::ImportRecordBatch(&cudf_device_array->array, cudf_schema.get());
    if (!maybe_arrow_batch.ok()) {
        CHECK_STATUS(maybe_arrow_batch.status());
    }

    auto arrow_batch = maybe_arrow_batch.ValueOrDie();

    // cast the record batch from cudf_schema to the target schema
    // this is necessary since cudf might convert all decimal types to decimal 128, as the widest type
    CHECK_STATUS(cast_record_batch_to_schema(arrow_batch, schema->get_schema()));

    // cudf_schema and cudf_device_array will automatically be released
    // since they are unique_ptr with custom deleter

    return arrow_batch;
}

// Convert cuDF Table to Arrow Table using Arrow C Interface
std::shared_ptr<arrow::Table> cudf_to_arrow_table(cudf::table_view cudf_table,
                                                  std::shared_ptr<Schema> schema,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr,
                                                  arrow::MemoryPool* pool) {
    auto batch = cudf_to_arrow_batch(cudf_table, schema, stream, mr, pool);
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches{batch};

    return to_table(batches, pool);
}

}  // namespace maximus
