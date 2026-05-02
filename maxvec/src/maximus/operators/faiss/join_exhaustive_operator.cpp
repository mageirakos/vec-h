#include <faiss/IndexFlat.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>

#include <maximus/operators/faiss/join_exhaustive_operator.hpp>
#include <maximus/operators/faiss/interop.hpp>

namespace maximus::faiss {

JoinExhaustiveOperator::JoinExhaustiveOperator(
    std::shared_ptr<MaximusContext> &ctx,
    std::vector<std::shared_ptr<Schema>> input_schemas,
    std::shared_ptr<VectorJoinExhaustiveProperties> properties)
        : JoinOperator(ctx, input_schemas, properties), properties(properties) {
    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::FAISS);

    assert(output_schema);
}

DeviceTablePtr JoinExhaustiveOperator::run_kernel(std::shared_ptr<MaximusContext> &ctx,
                                                  const TablePtr &query_table,
                                                  const TablePtr &data_table) {
    auto query_at            = query_table->get_table();
    auto data_at             = data_table->get_table();
    auto query_vector_column = properties->query_vector_column.name();
    auto data_vector_column  = properties->data_vector_column.name();
    auto query_arrays        = query_at->GetColumnByName(*query_vector_column);
    auto data_arrays         = data_at->GetColumnByName(*data_vector_column);
    int nq_total             = query_at->num_rows();

    SearchResult searchResult = (abstract_properties->K.has_value())
                                    ? knn_search(query_arrays, data_arrays)
                                    : range_search(query_arrays, data_arrays);

    PE("build_result_cpu");
    std::vector<std::string> skip_columns;
    if (!properties->keep_data_vector_column) skip_columns.push_back(*data_vector_column);
    if (abstract_properties->filter_bitmap.has_value())
        skip_columns.push_back(*abstract_properties->filter_bitmap->name());
    auto query_side = build_join_side(query_at,
                                      searchResult.left_indices,
                                      ctx,
                                      abstract_properties->keep_query_vector_column
                                          ? std::vector<std::string>{}
                                          : std::vector<std::string>{*query_vector_column});
    auto data_side =
        build_join_side(data_at, searchResult.right_indices, ctx, std::move(skip_columns));

    // Concatenate taken query and data columns
    arrow::ChunkedArrayVector joined_columns;
    std::vector<std::shared_ptr<arrow::Field>> joined_fields;
    for (int i = 0; i < query_side->num_columns(); ++i) {
        joined_columns.push_back(query_side->column(i));
        joined_fields.push_back(query_side->schema()->field(i));
    }
    for (int i = 0; i < data_side->num_columns(); ++i) {
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

JoinExhaustiveOperator::SearchResult JoinExhaustiveOperator::knn_search(
    const ChunkedArrayPtr &query_vectors, const ChunkedArrayPtr &data_vectors) {
    PE("knn_search_cpu");
    auto query_chunks           = query_vectors->chunks();
    auto data_chunks            = data_vectors->chunks();
    int64_t nq_total            = query_vectors->length();
    int64_t K                   = properties->K.value();
    VectorDistanceMetric metric = properties->metric;
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
        auto D = embedding_dimension(query_array);

        knn_exhaustive_search(D,
                              data_chunks,
                              nq,
                              query_vectors_ptr,
                              K,
                              metric,
                              distances.data(),
                              labels.data(),
                              _id_filter_selector.get());

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
    PL("knn_search_cpu");
    return SearchResult{arrow::ChunkedArray::Make({left_indices_array}).ValueUnsafe(),
                        arrow::ChunkedArray::Make({right_indices_array}).ValueUnsafe(),
                        arrow::ChunkedArray::Make({distances_array}).ValueUnsafe()};
}

JoinExhaustiveOperator::SearchResult JoinExhaustiveOperator::range_search(
    const ChunkedArrayPtr &query_vectors, const ChunkedArrayPtr &data_vectors) {
    PE("range_search_cpu");
    auto query_chunks           = query_vectors->chunks();
    auto data_chunks            = data_vectors->chunks();
    int64_t nq_total            = query_vectors->length();
    float radius                = properties->radius.value();
    VectorDistanceMetric metric = properties->metric;
    int64_t nq_results          = 0;

    // Results ordered by query chunk, data chunk
    std::vector<std::vector<RangeSearchResultPtr>> all_results(query_chunks.size());
    for (int i = 0; i < query_chunks.size(); ++i) {
        const auto &query_array = query_chunks[i];
        const float *query_vectors_ptr = get_embedding_raw_ptr(query_array);
        int64_t nq = query_array->length();
        auto D = embedding_dimension(query_array);

        int data_offset = 0;
        for (const auto &data_array : data_chunks) {
            const float *data_vectors_ptr = get_embedding_raw_ptr(data_array);
            int64_t nb = data_array->length();
            all_results[i].push_back(std::make_unique<::faiss::RangeSearchResult>(nq, true));
            std::unique_ptr<::faiss::IDSelector> sel_shifted;
            if (_id_filter_selector)
                std::make_unique<IDSelectorOffset>(_id_filter_selector.get(), data_offset);

            PE("faiss");
            switch (metric) {
                case VectorDistanceMetric::L2:
                    ::faiss::range_search_L2sqr(query_vectors_ptr,
                                                data_vectors_ptr,
                                                D,
                                                nq,
                                                nb,
                                                radius,
                                                all_results[i].back().get(),
                                                sel_shifted.get());
                    break;
                case VectorDistanceMetric::INNER_PRODUCT:
                    ::faiss::range_search_inner_product(query_vectors_ptr,
                                                        data_vectors_ptr,
                                                        D,
                                                        nq,
                                                        nb,
                                                        radius,
                                                        all_results[i].back().get(),
                                                        sel_shifted.get());
                    break;
                default:
                    throw std::runtime_error("Unsupported metric type for join top operator");
            }
            PL("faiss");
            int results_from_batch = all_results[i].back()->lims[all_results[i].back()->nq];
            nq_results += results_from_batch;
            for (int j = 0; j < results_from_batch; ++j) {
                all_results[i].back()->labels[j] += data_offset;
            }
            data_offset += data_array->length();
        }
    }

    // Map to Indices
    PL("range_search_cpu");
    return parse_range_search_results(all_results, nq_results);
}

void JoinExhaustiveOperator::knn_exhaustive_search(
        const int64_t D,
        arrow::ArrayVector &data_vectors,
        const int64_t nq,
        const float *query_vectors_ptr,
        const int64_t K,
        const VectorDistanceMetric metric,
        float *distances,
        ::faiss::idx_t *labels,
        const ::faiss::IDSelector *sel) {
    // KNN search results for all data shards/chunks
    std::vector<::faiss::idx_t> all_ids(data_vectors.size() * nq * K);
    std::vector<float> all_distances(data_vectors.size() * nq * K);

    int data_offset = 0;
    for (int i = 0; i < data_vectors.size(); ++i) {
        const auto &data_array = data_vectors[i];
        const float *data_vectors_ptr = get_embedding_raw_ptr(data_array);
        int64_t nb       = data_array->length();
        auto sel_shifted = sel ? std::make_unique<IDSelectorOffset>(sel, data_offset) : nullptr;

        PE("faiss");
        switch (metric) {
            case VectorDistanceMetric::L2:
                ::faiss::knn_L2sqr(query_vectors_ptr,
                                   data_vectors_ptr,
                                   D,
                                   nq,
                                   nb,
                                   K,
                                   all_distances.data() + i * nq * K,
                                   all_ids.data() + i * nq * K,
                                   nullptr,
                                   sel_shifted.get());
                break;
            case VectorDistanceMetric::INNER_PRODUCT:
                ::faiss::knn_inner_product(query_vectors_ptr,
                   data_vectors_ptr,
                   D,
                   nq,
                   nb,
                   K,
                   all_distances.data() + i * nq * K,
                   all_ids.data() + i * nq * K,
                   // as opposed to the L2 distance, y_norm2 parameter is not supported here
                   sel_shifted.get());
                break;
            default:
                throw std::runtime_error("Unsupported metric type for join top operator");
        }

        PL("faiss");
        for (int j = 0; j < nq * K; ++j) {
            all_ids[i * nq * K + j] += data_offset;
        }
        data_offset += nb;
    }

    auto merge_knn = [&](auto Comparator) {
        ::faiss::merge_knn_results<::faiss::idx_t, decltype(Comparator)>(
            nq, K, data_vectors.size(),
            all_distances.data(), all_ids.data(),
            distances, labels);
    };

    // minimize if it's a distance and maximize if it's similarity
    if (metric == VectorDistanceMetric::L2) {
        merge_knn(::faiss::CMin<float, int32_t>{});
    } else {
        assert(metric == VectorDistanceMetric::INNER_PRODUCT);
        merge_knn(::faiss::CMax<float, int32_t>{});
    }
}

}  // namespace maximus::faiss
