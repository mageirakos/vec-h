#include <faiss/IndexFlat.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>

#include <maximus/operators/faiss/interop.hpp>
#include <maximus/operators/faiss/project_distance_operator.hpp>

namespace maximus::faiss {

ProjectDistanceOperator::ProjectDistanceOperator(
    std::shared_ptr<MaximusContext> &ctx,
    std::vector<std::shared_ptr<Schema>> input_schemas,
    std::shared_ptr<VectorProjectDistanceProperties> properties)
        : AbstractVectorProjectDistanceOperator(
              ctx, std::move(input_schemas), std::move(properties)) {

    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::FAISS);

    for (int i : {0, 1}) {
        _buffer_table[i] = std::make_shared<Table>(
            ctx_,
            arrow::Table::MakeEmpty(this->input_schemas[i]->get_schema(), ctx_->get_memory_pool())
                .ValueOrDie());
    }
}

void ProjectDistanceOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);
    assert(port == 0 || port == 1);
    device_input.convert_to<TablePtr>(ctx_, input_schemas[port]);
    TablePtr input_table = device_input.as_table();
    assert(input_table);
    int opposing_port    = 1 - port;

    if (!_buffer_table[opposing_port]->empty()) {
        TablePtr result_batch =
            (port == 0) ? run_kernel(ctx_, input_table, _buffer_table[1],*properties)
                        : run_kernel(ctx_, _buffer_table[0], input_table,*properties);
        assert(result_batch && !result_batch->empty());
        outputs_.push_back(DeviceTablePtr(std::move(result_batch)));
    }
    if (needs_input(opposing_port)) {
        // since _buffer_table[port] is initially set to contain an empty batch, concatenating it
        // with the input_table, using the built-in arrow's ConcatenateTables, would create a new table
        // that has the first batch empty, followed by the input_table batches, as follows:
        // [[], input_table_batches]
        // for this reason, we only concatenate non-empty tables from the list
        auto merged_result = concatenate_nonempty_tables(
            {_buffer_table[port]->get_table(), input_table->get_table()},
            ctx_->get_memory_pool(),
            arrow::ConcatenateTablesOptions::Defaults());
        CHECK_STATUS(merged_result.status());
        _buffer_table[port] = std::make_shared<Table>(ctx_, merged_result.ValueOrDie());
        assert(_buffer_table[port] && !_buffer_table[port]->empty());
        // std::cout << "buffer_table[" << port << "] = " << _buffer_table[port]->to_string() << std::endl;
    }
}

TablePtr ProjectDistanceOperator::run_kernel(std::shared_ptr<MaximusContext> &ctx,
                                             TablePtr &left_table,
                                             TablePtr &right_table,
                                             VectorProjectDistanceProperties &properties) {
    auto left_at     = left_table->get_table();
    auto right_at    = right_table->get_table();
    auto left_array  = left_at->GetColumnByName(*properties.left_vector_column.name());
    auto right_array = right_at->GetColumnByName(*properties.right_vector_column.name());

    if (left_array->length() == 0 || right_array->length() == 0) return nullptr;
    if (left_array->num_chunks() == 0 || right_array->num_chunks() == 0) return nullptr;
    assert(left_array->num_chunks() > 0 && right_array->num_chunks() > 0);

    if (left_array->chunk(0)->length() == 0 || right_array->chunk(0)->length() == 0) return nullptr;

    assert(left_array->chunk(0)->length() > 0);
    assert(right_array->chunk(0)->length() > 0);

    auto D = embedding_dimension(left_array->chunk(0));
    assert(D == embedding_dimension(right_array->chunk(0)));

    size_t nl        = left_at->num_rows();
    size_t nr        = right_at->num_rows();
    size_t ntotal    = nl * nr;

    FloatArrayPtr distances =
        compute_pairwise_distances(left_array, right_array, D, ctx->get_memory_pool());

    if (!properties.keep_left_vector_column) {
        int qv_idx = left_table->get_schema()->get_schema()->GetFieldIndex(
            *properties.left_vector_column.name());
        left_at = left_at->RemoveColumn(qv_idx).ValueOrDie();
    }
    if (!properties.keep_right_vector_column) {
        int dv_idx = right_table->get_schema()->get_schema()->GetFieldIndex(
            *properties.right_vector_column.name());
        right_at = right_at->RemoveColumn(dv_idx).ValueOrDie();
    }

    std::vector<std::shared_ptr<arrow::ChunkedArray>> joined_columns;
    std::vector<std::shared_ptr<arrow::Field>> joined_fields;

    // 1. Right side (Query) - FIRST
    arrow::Int64Builder right_index_builder;
    for (int i = 0; i < right_at->num_columns(); ++i) {
        ChunkedArrayPtr original_right_col_array = right_at->column(i);
        arrow::ArrayVector chunks_for_this_col;
        chunks_for_this_col.reserve(nl);
        for (int64_t k = 0; k < nl; ++k) {
            for (auto &chunk : original_right_col_array->chunks()) {
                chunks_for_this_col.push_back(chunk);
            }
        }
        auto chunked_col = std::make_shared<arrow::ChunkedArray>(chunks_for_this_col,
                                                                 original_right_col_array->type());
        joined_columns.push_back(chunked_col);
        joined_fields.push_back(right_at->schema()->field(i));
    }

    // 2. Left side (Data) - SECOND
    arrow::Int64Builder left_index_builder;
    CHECK_STATUS(left_index_builder.Reserve(ntotal));
    for (int64_t i = 0; i < nl; ++i) {
        for (int64_t j = 0; j < nr; ++j) {
            left_index_builder.UnsafeAppend(i);
        }
    }
    std::shared_ptr<arrow::Array> left_index_array;
    CHECK_STATUS(left_index_builder.Finish(&left_index_array));
    auto left_take_result = arrow::compute::Take(left_at,
                                                 left_index_array,
                                                 arrow::compute::TakeOptions::Defaults(),
                                                 ctx->get_exec_context());
    CHECK_STATUS(left_take_result.status());
    auto taken_left = left_take_result.ValueOrDie().table();
    joined_columns.insert(
        joined_columns.end(), taken_left->columns().begin(), taken_left->columns().end());
    joined_fields.insert(joined_fields.end(),
                         taken_left->schema()->fields().begin(),
                         taken_left->schema()->fields().end());

    // Add distance column
    joined_columns.push_back(std::make_shared<arrow::ChunkedArray>(distances));
    joined_fields.push_back(arrow::field(properties.distance_column_name, arrow::float32()));
    auto output_arrow_table = arrow::Table::Make(arrow::schema(joined_fields), joined_columns);
    return std::make_shared<Table>(ctx, output_arrow_table);
}

FloatArrayPtr ProjectDistanceOperator::compute_pairwise_distances(
    const ChunkedArrayPtr &left_vectors,
    const ChunkedArrayPtr &right_vectors,
    int64_t D,
    arrow::MemoryPool *pool) {
    auto &left_chunks  = left_vectors->chunks();
    auto &right_chunks = right_vectors->chunks();
    size_t ntotal      = left_vectors->length() * right_vectors->length();

    auto buffer = arrow::AllocateBuffer(sizeof(float) * ntotal, pool);
    CHECK_STATUS(buffer.status());
    float *distance_ptr = buffer.ValueOrDie()->mutable_data_as<float>();
    auto distances = std::make_shared<arrow::FloatArray>(ntotal, std::move(buffer.ValueOrDie()));

    for (const auto &left_chunk : left_chunks) {
        const float *left_vectors_ptr = get_embedding_raw_ptr(left_chunk);
        for (const auto &right_chunk : right_chunks) {
            const float *right_vectors_ptr = get_embedding_raw_ptr(right_chunk);
            PE("faiss");
            ::faiss::pairwise_L2sqr(D,
                                    left_chunk->length(),
                                    left_vectors_ptr,
                                    right_chunk->length(),
                                    right_vectors_ptr,
                                    distance_ptr);
            PL("faiss");
            distance_ptr += left_chunk->length() * right_chunk->length();
        }
    }
    return std::move(distances);
}

}  // namespace maximus::faiss
