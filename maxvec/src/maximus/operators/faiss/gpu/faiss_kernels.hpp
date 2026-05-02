#pragma once

#include <faiss/MetricType.h>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <faiss/Index.h>
#include <maximus/indexes/faiss/faiss_gpu_index.hpp>
#include <maximus/types/device_table_ptr.hpp>
#include <maximus/utils/cudf_helpers.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>

namespace maximus::faiss::gpu {
// result container
struct GpuKnnResult {
    std::unique_ptr<cudf::column> distances; // FLOAT32, size nq*K
    std::unique_ptr<cudf::column> labels;    // INT64,   size nq*K

    GpuKnnResult(
        std::unique_ptr<cudf::column> distances_,
        std::unique_ptr<cudf::column> labels_)
        : distances(std::move(distances_)),
          labels(std::move(labels_)) {

        assert(distances->type().id() == cudf::type_id::FLOAT32);
        assert(labels->type().id() == cudf::type_id::INT64);
        assert(distances->size() == labels->size());
    }
};

// GPUSearchResult: final returned filtered columns (device-resident cudf columns)
struct GpuSearchResult {
    std::unique_ptr<cudf::column> left_indices;  // INT64, size nq*K
    std::unique_ptr<cudf::column> right_indices; // INT64, size nq*K
    std::unique_ptr<cudf::column> distances;     // FLOAT32, size nq*K

    GpuSearchResult(
        std::unique_ptr<cudf::column> left_indices_,  // INT64, size nq*K
        std::unique_ptr<cudf::column> right_indices_, // INT64, size nq*K
        std::unique_ptr<cudf::column> distances_
    ): left_indices(std::move(left_indices_))
     , right_indices(std::move(right_indices_))
     , distances(std::move(distances_))
    {
        assert(left_indices->type().id() == cudf::type_id::INT64);
        assert(right_indices->type().id() == cudf::type_id::INT64);
        assert(distances->type().id() == cudf::type_id::FLOAT32);

        assert(left_indices->size() == right_indices->size());
        assert(distances->size() == left_indices->size());
    }
};

const float* get_embedding_ptr(cudf::column_view const& list_col, int64_t D);
int get_num_vectors(cudf::column_view const& list_col);

/*
 * D: embedding dimensionality
 * data_vectors: vector of cudf::column_view, each is (nb_shard * D) flat float32 (row-major)
 * nq: number of queries
 * query_vectors_ptr: device pointer to queries (nq x D, row-major)
 * K: number of neighbors
 * mr: rmm device memory resource for allocations (pool_mr etc.)
 * stream: CUDA stream to use
 *
 * Returns: GPU-resident distances and indices in cudf columns (row-major: nq x K flattened).
 */

// same as before, but assumes the input chunks are already concatenated in a single chunk
GpuKnnResult knn_exhaustive_gpu(
    cudf::column_view const& data_vectors,   // LIST<FLOAT32>
    cudf::column_view const& query_vectors,  // LIST<FLOAT32>
    const int64_t D,
    const int64_t K,
    const ::faiss::MetricType metric,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream);

/*
 * knn_search_gpu
 *
 * query_vectors: cudf::column_view with nq * D float32 (row-major)
 * data_vectors: vector<cudf::column_view> each shard: nb_shard * D float32
 * D, nq, K, metric, mr, stream: same as above
 *
 * Returns: GPUSearchResult with the same semantics as CPU knn_search:
 *   - rows where label >= 0 are kept, others removed
 *   - remaining rows are (left_query_index, label, distance)
 */
// same as before, but assumes the input chunks are already concatenated in a single chunk
GpuSearchResult knn_search_gpu(
    cudf::column_view const& data_vectors,    // LIST<FLOAT32> (nb lists of dim D)
    cudf::column_view const& query_vectors,   // LIST<FLOAT32> (nq lists of dim D)
    const int64_t D,
    const int64_t K,
    const ::faiss::MetricType metric,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream);

std::unique_ptr<cudf::column> pairwise_distances_gpu(
    cudf::column_view const& data_vectors,    // LIST<FLOAT32> nb x D
    cudf::column_view const& query_vectors,   // LIST<FLOAT32> nq x D
    const int64_t D,
    const ::faiss::MetricType metric,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream);

GpuSearchResult ann_search_gpu(
    std::shared_ptr<FaissIndex>& index,
    cudf::column_view const& query_vectors,   // LIST<FLOAT32>
    const int64_t D,
    const int64_t K,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream,
    const ::faiss::SearchParameters* params = nullptr);

std::shared_ptr<cudf::table> join_after_knn_search_gpu(
    GpuSearchResult& knn_result,
    CudfTablePtr data_table, CudfTablePtr query_table,
    std::shared_ptr<arrow::Schema>& data_schema, std::shared_ptr<arrow::Schema>& query_schema,
    std::shared_ptr<arrow::Schema>& output_schema,
    std::optional<std::string> distance_column,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream);

}