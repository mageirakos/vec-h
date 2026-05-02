#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>

#include <maximus/operators/faiss/interop.hpp>
#include <maximus/operators/faiss/join_indexed_operator.hpp>


namespace maximus::faiss {

using RangeSearchResultPtr = std::unique_ptr<::faiss::RangeSearchResult>;

JoinIndexedOperator::JoinIndexedOperator(std::shared_ptr<MaximusContext> &ctx,
                                         std::vector<std::shared_ptr<Schema>> input_schemas,
                                         std::shared_ptr<VectorJoinIndexedProperties> properties)
        : JoinOperator(ctx, input_schemas, properties), properties(properties) {
    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::FAISS);

    _index = std::dynamic_pointer_cast<FaissIndex>(this->properties->index);
    
    _search_parameters =
        (this->properties->index_parameters)
            ? std::dynamic_pointer_cast<FaissSearchParameters>(this->properties->index_parameters)
                  ->params
            : nullptr;

    // Bind filter expression if provided
    if (properties->filter_expr) {
        auto data_schema = input_schemas[get_data_port()]->get_schema();
        auto result =
            properties->filter_expr->get_expression()->Bind(*data_schema, ctx->get_exec_context());
        CHECK_STATUS(result.status());
        bound_filter_expression = result.ValueOrDie();
    }

    assert(output_schema);
}

DeviceTablePtr JoinIndexedOperator::run_kernel(std::shared_ptr<MaximusContext> &ctx,
                                               const TablePtr &query_table,
                                               const TablePtr &data_table) {
    auto query_at            = query_table->get_table();
    auto data_at             = data_table->get_table();
    auto query_vector_column = abstract_properties->query_vector_column.name();
    auto data_vector_column  = abstract_properties->data_vector_column.name();
    auto query_arrays        = query_at->GetColumnByName(*query_vector_column);
    int nq_total             = query_at->num_rows();

    auto query_schema       = input_schemas[1]->get_schema();
    auto query_field_result = properties->query_vector_column.GetOne(*query_schema);
    CHECK_STATUS(query_field_result.status());
    auto query_field = query_field_result.ValueOrDie();

    // Move index to CPU if it's currently on GPU (lazy movement)
    // Close operator region before index movement to treat it like other data movement
    if (_index && _index->is_on_gpu()) {
        std::cout << "[CPU JoinIndexedOperator] Moving GPU index to CPU for execution" << std::endl;
        profiler::close_regions({name(), "no_more_input"});
        profiler::open_regions({"DataTransformation", "GPU->CPU", "index_movement"});
        _index = _index->to_cpu();
        profiler::close_regions({"DataTransformation", "GPU->CPU", "index_movement"});
        profiler::open_regions({name(), "no_more_input"});
    }

    // For indexed search with trained index, data_vector_column may be empty/not in schema
    // since the index already owns the data vectors
    auto data_schema = input_schemas[0]->get_schema();
    std::shared_ptr<arrow::Field> data_field = nullptr;
    if (data_vector_column && !data_vector_column->empty()) {
        auto data_field_result = properties->data_vector_column.GetOne(*data_schema);
        if (data_field_result.ok()) {
            data_field = data_field_result.ValueOrDie();
        }
    }

    SearchResult searchResult = (abstract_properties->K.has_value()) ? knn_search(query_arrays)
                                                                     : range_search(query_arrays);

    PE("build_result_cpu");
    auto query_side = build_join_side(query_at,
                                      searchResult.left_indices,
                                      ctx,
                                      abstract_properties->keep_query_vector_column
                                          ? std::vector<std::string>{}
                                          : std::vector<std::string>{*query_vector_column});
    
    // For data side, only skip the data_vector_column if it exists
    std::vector<std::string> skip_columns;
    if (abstract_properties->filter_bitmap.has_value()) {
        skip_columns.push_back(*abstract_properties->filter_bitmap->name());
    } else if (!abstract_properties->keep_data_vector_column && data_vector_column && !data_vector_column->empty()) {
        skip_columns.push_back(*data_vector_column);
    }
    auto data_side = build_join_side(data_at, searchResult.right_indices, ctx, skip_columns);

    // Concatenate taken query and data columns
    arrow::ChunkedArrayVector joined_columns;
    std::vector<std::shared_ptr<arrow::Field>> joined_fields;
    for (int i = 0; i < query_side->num_columns(); ++i) {
        auto &field = query_side->schema()->field(i);
        if (!properties->keep_query_vector_column && field->name() == query_field->name()) {
            continue;
        }
        joined_columns.push_back(query_side->column(i));
        joined_fields.push_back(field);
    }
    for (int i = 0; i < data_side->num_columns(); ++i) {
        auto &field = data_side->schema()->field(i);
        // Only skip data_vector_column if data_field exists
        if (data_field && !properties->keep_data_vector_column && field->name() == data_field->name()) {
            continue;
        }
        joined_columns.push_back(data_side->column(i));
        joined_fields.push_back(data_side->schema()->field(i));
    }

    // Optionally add distance column
    if (abstract_properties->distance_column.has_value()) {
        joined_columns.push_back(searchResult.distances);
        joined_fields.push_back(
            arrow::field(abstract_properties->distance_column.value(), arrow::float32()));
    }

    auto output_table = arrow::Table::Make(arrow::schema(joined_fields), joined_columns);
    PL("build_result_cpu");
    return DeviceTablePtr(output_table);
}

JoinOperator::SearchResult JoinIndexedOperator::knn_search(const ChunkedArrayPtr &query_vectors) {
    PE("ann_search_cpu");
    auto query_chunks      = query_vectors->chunks();
    int64_t nq_total       = query_vectors->length();
    int64_t K              = properties->K.value();
    auto search_parameters = _search_parameters.get();
    auto index             = _index->faiss_index.get();
    arrow::Int64Builder left_indices_builder;
    arrow::Int64Builder right_indices_builder;
    arrow::FloatBuilder distances_builder;
    CHECK_STATUS(left_indices_builder.Reserve(nq_total * K));
    CHECK_STATUS(right_indices_builder.Reserve(nq_total * K));
    CHECK_STATUS(distances_builder.Reserve(nq_total * K));

    int qi = 0;
    for (const auto &query_array : query_chunks) {
        const float *query_vectors_ptr = get_embedding_raw_ptr(query_array);
        int64_t nq = query_array->length();
        std::vector<::faiss::idx_t> labels(K * nq);
        std::vector<float> distances(K * nq);

        PE("faiss");
        index->search(nq, query_vectors_ptr, K, distances.data(), labels.data(), search_parameters);
        PL("faiss");

        for (int i = 0; i < nq; ++i) {
            for (int j = 0; j < K && labels[i * K + j] >= 0; ++j) {
                left_indices_builder.UnsafeAppend(qi + i);
                right_indices_builder.UnsafeAppend(labels[i * K + j]);
                distances_builder.UnsafeAppend(distances[i * K + j]);
            }
        }
        qi += nq;
    }

    std::shared_ptr<arrow::Int64Array> left_indices_array;
    std::shared_ptr<arrow::Int64Array> right_indices_array;
    std::shared_ptr<arrow::FloatArray> distances_array;
    CHECK_STATUS(left_indices_builder.Finish(&left_indices_array));
    CHECK_STATUS(right_indices_builder.Finish(&right_indices_array));
    CHECK_STATUS(distances_builder.Finish(&distances_array));
    PL("ann_search_cpu");
    return SearchResult{arrow::ChunkedArray::Make({left_indices_array}).ValueUnsafe(),
                        arrow::ChunkedArray::Make({right_indices_array}).ValueUnsafe(),
                        arrow::ChunkedArray::Make({distances_array}).ValueUnsafe()};
}

JoinOperator::SearchResult JoinIndexedOperator::range_search(const ChunkedArrayPtr &query_vectors) {
    PE("ann_range_search_cpu");
    auto query_chunks      = query_vectors->chunks();
    int64_t nq_total       = query_vectors->length();
    auto search_parameters = _search_parameters.get();
    auto index             = _index->faiss_index.get();
    float radius           = properties->radius.value();
    int64_t nq_results     = 0;

    // Results ordered by query chunk, data chunk
    std::vector<std::vector<RangeSearchResultPtr>> all_results(query_chunks.size());
    for (int i = 0; i < query_chunks.size(); ++i) {
        const auto &query_array = query_chunks[i];
        const float *query_vectors_ptr = get_embedding_raw_ptr(query_array);
        int64_t nq = query_array->length();

        all_results[i].push_back(std::make_unique<::faiss::RangeSearchResult>(nq, true));
        PE("faiss");
        index->range_search(
            nq, query_vectors_ptr, radius, all_results[i][0].get(), search_parameters);
        PL("faiss");
        nq_results += all_results[i][0]->lims[all_results[i][0]->nq];
    }

    PL("ann_range_search_cpu");
    return parse_range_search_results(all_results, nq_results);
}


void JoinIndexedOperator::on_no_more_input(int port) {
    assert(port == 0 || port == 1);
    if (port == 0 && _data_table) {
        if (properties->filter_expr) {
            _id_filter_selector =
                std::make_unique<IDSelectorCallback>(bound_filter_expression, _data_table);
            if (!_search_parameters)
                _search_parameters = std::make_shared<::faiss::SearchParameters>();
            _search_parameters->sel = _id_filter_selector.get();

        } else if (properties->filter_bitmap.has_value()) {
            auto chunked = _data_table->get_table()->GetColumnByName(
                *properties->filter_bitmap.value().name());
            assert(chunked->type()->id() == arrow::Type::BOOL);
            auto array =
                arrow::Concatenate(chunked->chunks(), ctx_->get_memory_pool()).ValueOrDie();
            auto boolean_array = std::static_pointer_cast<arrow::BooleanArray>(array);
            if (!_search_parameters)
                _search_parameters = std::make_shared<::faiss::SearchParameters>();
            _id_filter_selector     = std::make_unique<IDSelectorBitmap>(boolean_array);
            _search_parameters->sel = _id_filter_selector.get();
        }
    }
    if (!needs_input(0) && _data_table && _query_table) {
        maximus::DeviceTablePtr output_table = run_kernel(ctx_, _query_table, _data_table);
        if (!output_table.empty()) outputs_.push_back(std::move(output_table));
        _query_table = nullptr;
    }
}

}  // namespace maximus::faiss
