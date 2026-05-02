#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>

#include <maximus/operators/faiss/join_operator.hpp>


namespace maximus::faiss {

using RangeSearchResultPtr = std::unique_ptr<::faiss::RangeSearchResult>;

JoinOperator::JoinOperator(std::shared_ptr<MaximusContext> &ctx,
                           std::vector<std::shared_ptr<Schema>> input_schemas,
                           std::shared_ptr<VectorJoinProperties> properties)
        : AbstractVectorJoinOperator(ctx, std::move(input_schemas), properties) {
    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::FAISS);
}

void JoinOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);
    assert(port == 0 || port == 1);

    auto &target_table = (port == 0) ? _data_table : _query_table;
    const auto &schema = input_schemas[port];

    const auto& operator_name = name();
    profiler::close_regions({operator_name, "add_input"});
    device_input.convert_to<TablePtr>(ctx_, schema);
    profiler::open_regions({operator_name, "add_input"});

    if (target_table) {
        auto merged_result = arrow::ConcatenateTables(
            {target_table->get_table(), device_input.as_table()->get_table()},
            arrow::ConcatenateTablesOptions::Defaults(),
            ctx_->get_memory_pool());
        CHECK_STATUS(merged_result.status());
        target_table = std::make_shared<Table>(ctx_, merged_result.ValueOrDie());
    } else {
        target_table = device_input.as_table();
    }

    if (!needs_input(0) && _data_table && _query_table) {
        maximus::DeviceTablePtr output_table = run_kernel(ctx_, _query_table, _data_table);
        if (!output_table.empty()) outputs_.push_back(std::move(output_table));
        _query_table = nullptr;
    }
}


std::shared_ptr<arrow::Table> JoinOperator::build_join_side(
    const std::shared_ptr<arrow::Table> &table,
    const std::shared_ptr<arrow::ChunkedArray> &indices,
    const std::shared_ptr<MaximusContext> &ctx,
    const std::vector<std::string> skip_columns) {
    auto table_ = table;
    for (const auto &col_name : skip_columns) {
        int dv_idx = table_->schema()->GetFieldIndex(col_name);
        table_     = table_->RemoveColumn(dv_idx).ValueOrDie();
    }
    assert(indices->type()->id() == arrow::Type::INT64);
    auto result = arrow::compute::Take(
        table_, indices, arrow::compute::TakeOptions::Defaults(), ctx->get_exec_context());
    CHECK_STATUS(result.status());
    return result.ValueOrDie().table();
}

JoinOperator::SearchResult JoinOperator::parse_range_search_results(
    const std::vector<std::vector<RangeSearchResultPtr>> &all_results, int64_t nq_results) {
    // Map to Indices
    arrow::Int64Builder left_indices_builder, right_indices_builder;
    arrow::FloatBuilder distances_builder;
    CHECK_STATUS(left_indices_builder.Reserve(nq_results));
    CHECK_STATUS(right_indices_builder.Reserve(nq_results));
    CHECK_STATUS(distances_builder.Reserve(nq_results));
    int queries_seen = 0;
    for (int qbi = 0; qbi < all_results.size(); ++qbi) {
        for (const auto &result : all_results[qbi]) {
            CHECK_STATUS(
                right_indices_builder.AppendValues(result->labels, result->lims[result->nq]));
            CHECK_STATUS(
                distances_builder.AppendValues(result->distances, result->lims[result->nq]));
            for (int i = 0; i < result->nq; ++i) {
                int64_t start = result->lims[i];
                int64_t end   = result->lims[i + 1];
                for (int j = start; j < end; ++j) {
                    left_indices_builder.UnsafeAppend(queries_seen + i);
                }
            }
            queries_seen += result->nq;
        }
    }
    std::shared_ptr<arrow::Int64Array> left_indices_array, right_indices_array;
    std::shared_ptr<arrow::FloatArray> distances_array;
    CHECK_STATUS(left_indices_builder.Finish(&left_indices_array));
    CHECK_STATUS(right_indices_builder.Finish(&right_indices_array));
    CHECK_STATUS(distances_builder.Finish(&distances_array));
    return SearchResult{arrow::ChunkedArray::Make({left_indices_array}).ValueUnsafe(),
                        arrow::ChunkedArray::Make({right_indices_array}).ValueUnsafe(),
                        arrow::ChunkedArray::Make({distances_array}).ValueUnsafe()};
}


void JoinOperator::on_no_more_input(int port) {
    assert(port == 0 || port == 1);
    if (port == 0 && _data_table) {
        auto filter_bitmap = abstract_properties->filter_bitmap;
        if (filter_bitmap.has_value()) {
            auto chunked = _data_table->get_table()->GetColumnByName(*filter_bitmap.value().name());
            assert(chunked->type()->id() == arrow::Type::BOOL);
            auto array =
                arrow::Concatenate(chunked->chunks(), ctx_->get_memory_pool()).ValueOrDie();
            auto boolean_array  = std::static_pointer_cast<arrow::BooleanArray>(array);
            _id_filter_selector = std::make_unique<IDSelectorBitmap>(boolean_array);
        }
    }
    if (!needs_input(0) && _data_table && _query_table) {
        maximus::DeviceTablePtr output_table = run_kernel(ctx_, _query_table, _data_table);
        if (!output_table.empty()) outputs_.push_back(std::move(output_table));
        _query_table = nullptr;
    }
}

}  // namespace maximus::faiss
