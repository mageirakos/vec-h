#include <maximus/operators/faiss/gpu/faiss_kernels.hpp>

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/stream_compaction.hpp> // apply_boolean_mask
#include <cudf/binaryop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/copying.hpp> // gather
#include <cudf/types.hpp>

#include <faiss/gpu/GpuDistance.h>   // bfKnn, GpuDistanceParams
#include <faiss/gpu/StandardGpuResources.h>

#include <maximus/indexes/faiss/faiss_index.hpp>
#include <rmm/device_buffer.hpp>

#include <maximus/utils/cuda_helpers.hpp>
#include <maximus/utils/cudf_helpers.hpp>

namespace maximus::faiss::gpu {

const float* get_embedding_ptr(cudf::column_view const& list_col, int64_t D)
{
    assert(list_col.type().id() == cudf::type_id::LIST);
    auto const& child = list_col.child(cudf::lists_column_view::child_column_index);
    assert(child.type().id() == cudf::type_id::FLOAT32);
    assert(child.size() == list_col.size() * D);
    return reinterpret_cast<const float*>(child.head<float>());
}

int get_num_vectors(cudf::column_view const& list_col) {
    assert(list_col.type().id() == cudf::type_id::LIST);
    return list_col.size();
}

GpuKnnResult knn_exhaustive_gpu(
    cudf::column_view const& data_vectors,   // LIST<FLOAT32>
    cudf::column_view const& query_vectors,  // LIST<FLOAT32>
    const int64_t D,
    const int64_t K,
    const ::faiss::MetricType metric,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
    using idx_t = int64_t;

    const int64_t nb = data_vectors.size();
    const int64_t nq = query_vectors.size();

    // Extract raw device pointers (row-major)
    const float* db_ptr    = get_embedding_ptr(data_vectors, D);
    const float* query_ptr = get_embedding_ptr(query_vectors, D);

    // ----------------------------------------------------
    // Allocate flat FAISS outputs (nq * K)
    // ----------------------------------------------------
    auto flat_distances = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::FLOAT32},
        nq * K,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);

    auto flat_labels = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::INT64},
        nq * K,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);

    float* d_out_dist = flat_distances->mutable_view().data<float>();
    idx_t* d_out_idx  = flat_labels->mutable_view().data<idx_t>();

    // ----------------------------------------------------
    // FAISS GPU
    // ----------------------------------------------------
    // Thread-local StandardGpuResources to avoid re-creating CUDA handles/streams
    // and to enable Faiss's internal temp memory stack allocator.
    static thread_local ::faiss::gpu::StandardGpuResources gpu_res;
    int device = get_gpu_device();
    gpu_res.setDefaultStream(device, stream);

    ::faiss::gpu::GpuDistanceParams params;
    params.metric = metric;
    params.k = static_cast<int>(K);
    params.dims = static_cast<int>(D);

    params.vectors = reinterpret_cast<const void*>(db_ptr);
    params.vectorType = ::faiss::gpu::DistanceDataType::F32;
    params.vectorsRowMajor = true;
    params.numVectors = static_cast<::faiss::idx_t>(nb);

    params.queries = reinterpret_cast<const void*>(query_ptr);
    params.queryType = ::faiss::gpu::DistanceDataType::F32;
    params.queriesRowMajor = true;
    params.numQueries = static_cast<::faiss::idx_t>(nq);

    params.outDistances = d_out_dist;
    params.ignoreOutDistances = false;
    params.outIndicesType = ::faiss::gpu::IndicesDataType::I64;
    params.outIndices = reinterpret_cast<void*>(d_out_idx);
    params.device = device;
    PE("faiss");
    ::faiss::gpu::bfKnn(&gpu_res, params);
    PL("faiss");

    return GpuKnnResult{
        std::move(flat_distances),
        std::move(flat_labels)
    };
}

GpuSearchResult knn_search_gpu(
    cudf::column_view const& data_vectors,    // LIST<FLOAT32>
    cudf::column_view const& query_vectors,   // LIST<FLOAT32>
    const int64_t D,
    const int64_t K,
    const ::faiss::MetricType metric,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
    using idx_t = int64_t;

    // ----------------------------------------------------
    // Basic validation
    // ----------------------------------------------------
    assert(query_vectors.type().id() == cudf::type_id::LIST);
    assert(data_vectors.type().id() == cudf::type_id::LIST);

    const int64_t nq = query_vectors.size();
    const cudf::size_type total = static_cast<cudf::size_type>(nq * K);

    // ----------------------------------------------------
    // Run brute-force KNN (flat outputs)
    // ----------------------------------------------------
    GpuKnnResult bf = knn_exhaustive_gpu(
        data_vectors,
        query_vectors,
        D,
        K,
        metric,
        mr,
        stream);

    // bf.distances : FLOAT32, size nq*K
    // bf.labels    : INT64,   size nq*K

    // ----------------------------------------------------
    // Build left_indices = (0 .. nq*K-1) / K
    // ----------------------------------------------------
    auto row_index_col =
        ::maximus::make_sequence_column(total, 1, mr, stream); // 0,1,2,...

    cudf::numeric_scalar<int64_t> K_scalar(K, true, stream);

    auto left_indices_col = cudf::binary_operation(
        row_index_col->view(),
        K_scalar,
        cudf::binary_operator::DIV,
        cudf::data_type{cudf::type_id::INT64},
        stream,
        mr);

    // ----------------------------------------------------
    // Result (already flat)
    // ----------------------------------------------------
    return GpuSearchResult{
        std::move(left_indices_col),
        std::move(bf.labels),
        std::move(bf.distances)
    };
}

std::unique_ptr<cudf::column> pairwise_distances_gpu(
    cudf::column_view const& data_vectors,    // LIST<FLOAT32> nb x D
    cudf::column_view const& query_vectors,   // LIST<FLOAT32> nq x D
    const int64_t D,
    const ::faiss::MetricType metric,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
    const int64_t nq = query_vectors.size();
    const int64_t nb = data_vectors.size();
    const cudf::size_type total = nq * nb;

    // -------------------------------------------------------------------------
    // Run brute-force KNN with K = nb to get all distances
    // -------------------------------------------------------------------------
    GpuKnnResult bf = knn_exhaustive_gpu(
        data_vectors,
        query_vectors,
        D,
        nb,          // K = number of database vectors
        metric,
        mr,
        stream);

    // the output of this function is not ordered the same as e.g. the CPU's faiss::pairwise_L2sqr
    // here, for each query, the distances to nearest neighbors are sorted, as follows:
    // For each query q:
    //   distances[q] = sorted(distances to all db vectors)
    // for example:
    //   [q0-nearest, q0-2nd, q0-3rd, q0-farthest,
    //    q1 - nearest, q1 - 2nd, q1 - 3rd, q1 - farthest,
    //    q2 - nearest, q2 - 2nd, q2 - 3rd, q2 - farthest]

    // bellow, we reshuffle the values, so that the (i, j) entry, represents the (i, j) distance.
    // ---------------------------------------------------------------------
    // Allocate output (row-major q x b)
    // ---------------------------------------------------------------------
    auto output = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::FLOAT32},
        total,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);

    // ---------------------------------------------------------------------
    // Compute row offsets: q * nb
    // ---------------------------------------------------------------------
    auto row_ids =
        ::maximus::make_sequence_column(total, 1, mr, stream); // 0..total-1

    cudf::numeric_scalar<int64_t> nb_scalar(nb, true, stream);

    auto q_ids = cudf::binary_operation(
        row_ids->view(),
        nb_scalar,
        cudf::binary_operator::DIV,
        cudf::data_type{cudf::type_id::INT64},
        stream,
        mr);

    // q_ids[i] = query index

    // ---------------------------------------------------------------------
    // Compute target indices: q * nb + label
    // ---------------------------------------------------------------------
    auto q_offsets = cudf::binary_operation(
        q_ids->view(),
        nb_scalar,
        cudf::binary_operator::MUL,
        cudf::data_type{cudf::type_id::INT64},
        stream,
        mr);

    auto target_indices = cudf::binary_operation(
        q_offsets->view(),
        bf.labels->view(),
        cudf::binary_operator::ADD,
        cudf::data_type{cudf::type_id::INT64},
        stream,
        mr);

    // ---------------------------------------------------------------------
    // Scatter distances into correct positions
    // ---------------------------------------------------------------------
    // Wrap source and destination as single-column tables
    cudf::table_view src_table({ bf.distances->view() });
    cudf::table_view dst_table({ output->view() });

    // Scatter (returns a new table)
    auto scattered_table = cudf::scatter(
        src_table,
        target_indices->view(),
        dst_table,
        stream,
        mr);

    // Extract the scattered column
    return std::move(scattered_table->release()[0]);
}

GpuSearchResult ann_search_gpu(
    std::shared_ptr<FaissIndex>& index,
    cudf::column_view const& query_vectors,   // LIST<FLOAT32>
    const int64_t D,
    const int64_t K,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream,
    const ::faiss::SearchParameters* params)
{
    // ----------------------------------------------------
    // Validation
    // ----------------------------------------------------
    assert(query_vectors.type().id() == cudf::type_id::LIST);
    assert(index != nullptr);
    assert(index->faiss_index != nullptr);

    const int64_t nq = query_vectors.size();
    const cudf::size_type total = static_cast<cudf::size_type>(nq * K);

    // ----------------------------------------------------
    // Allocate ANN output columns
    // ----------------------------------------------------
    auto distances = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::FLOAT32},
        total,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);

    auto right_indices = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT64},
        total,
        cudf::mask_state::UNALLOCATED,
        stream,
        mr);

    // ----------------------------------------------------
    // Perform ANN search via FAISS GPU index
    // ----------------------------------------------------
    PE("faiss");
    index->search(
        query_vectors,
        static_cast<int>(K),
        *distances,
        *right_indices,
        params);
    PL("faiss");

    // ----------------------------------------------------
    // Build left_indices from dense (unfiltered) array
    // Must happen BEFORE -1 filtering so query assignments
    // remain correct when -1s are distributed unevenly
    // across queries (e.g. q0 gets 2 results, q1 gets 3)
    // ----------------------------------------------------
    auto row_index_col =
        maximus::make_sequence_column(total, 1, mr, stream);  // 0,1,2,...

    cudf::numeric_scalar<int64_t> K_scalar(K, true, stream);

    auto left_indices = cudf::binary_operation(
        row_index_col->view(),
        K_scalar,
        cudf::binary_operator::DIV,
        cudf::data_type{cudf::type_id::INT64},
        stream,
        mr);

    // ----------------------------------------------------
    // Filter out -1 results (Faiss returns -1 when fewer
    // than K neighbors found, e.g. small IVF clusters)
    // Filter all three columns together to keep them aligned
    // ----------------------------------------------------
    cudf::numeric_scalar<int64_t> neg_one(-1, true, stream);
    auto mask = cudf::binary_operation(
        right_indices->view(),
        neg_one,
        cudf::binary_operator::NOT_EQUAL,
        cudf::data_type{cudf::type_id::BOOL8},
        stream, mr);

    cudf::table_view to_filter({left_indices->view(), right_indices->view(), distances->view()});
    auto filtered = cudf::apply_boolean_mask(to_filter, mask->view(), stream, mr);
    auto filtered_cols = filtered->release();
    left_indices = std::move(filtered_cols[0]);
    right_indices = std::move(filtered_cols[1]);
    distances = std::move(filtered_cols[2]);

    // ----------------------------------------------------
    // Return final GPU search result
    // ----------------------------------------------------
    return GpuSearchResult{
        std::move(left_indices),
        std::move(right_indices),
        std::move(distances)
    };
}

std::shared_ptr<cudf::table> join_after_knn_search_gpu(
    GpuSearchResult& knn_result,
    CudfTablePtr data_table, CudfTablePtr query_table,
    std::shared_ptr<arrow::Schema>& data_schema, std::shared_ptr<arrow::Schema>& query_schema,
    std::shared_ptr<arrow::Schema>& output_schema,
    std::optional<std::string> distance_column,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream) {

    std::vector<std::unique_ptr<cudf::column>> output_columns;

    // ---------------------------------------------------------------------
    // Gather output columns
    // ---------------------------------------------------------------------
    for (int i = 0; i < output_schema->num_fields(); ++i) {
        auto field = output_schema->field(i);

        // Query-side column
        if (query_schema->GetFieldIndex(field->name()) != -1) {
            int q_idx = query_schema->GetFieldIndex(field->name());

            output_columns.push_back(
                ::maximus::gather_column(
                    query_table->view(),
                    q_idx,
                    knn_result.left_indices->view(),
                    cudf::out_of_bounds_policy::DONT_CHECK,
                    stream,
                    mr));
        }
        // Data-side column
        else if (data_schema->GetFieldIndex(field->name()) != -1) {
            int d_idx = data_schema->GetFieldIndex(field->name());

            output_columns.push_back(
                ::maximus::gather_column(
                    data_table->view(),
                    d_idx,
                    knn_result.right_indices->view(),
                    cudf::out_of_bounds_policy::DONT_CHECK,
                    stream,
                    mr));
        }
        // Distance column
        else if (distance_column &&
                 field->name() == distance_column.value()) {
            output_columns.push_back(std::move(knn_result.distances));
                 }
        else {
            throw std::runtime_error(
                "JoinExhaustiveOperator: unknown output field " + field->name());
        }
    }

    // ---------------------------------------------------------------------
    // Produce output table
    // ---------------------------------------------------------------------
    return std::make_shared<cudf::table>(std::move(output_columns));
}

}