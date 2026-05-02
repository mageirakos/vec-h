#include <arrow/io/file.h>
#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuIndexCagra.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <gtest/gtest.h>
#include <parquet/arrow/reader.h>

#include <faiss/gpu/GpuIndexIVFPQ.h> // Ensure this header is available
#include <cmath>
#include <filesystem>
#include <maximus/database.hpp>
#include <maximus/database_catalogue.hpp>
#include <maximus/frontend/expressions.hpp>
#include <maximus/frontend/query_plan_api.hpp>
#include <maximus/indexes/faiss/faiss_gpu_index.hpp>
#include <maximus/indexes/faiss/faiss_index.hpp>
#include <maximus/indexes/faiss/gpu_resources.hpp>
#include <maximus/indexes/index.hpp>
#include <maximus/operators/faiss/gpu/faiss_kernels.hpp>
#include <maximus/operators/faiss/gpu/join_exhaustive_operator.hpp>
#include <maximus/operators/faiss/gpu/join_indexed_operator.hpp>
#include <maximus/operators/faiss/join_indexed_operator.hpp>
#include <maximus/operators/faiss/project_distance_operator.hpp>
#include <maximus/types/expression.hpp>
#include <maximus/utils/cudf_helpers.hpp>
#include <unordered_set>

// fixtures
#include "sift_fixture.hpp"

using idx_t = faiss::idx_t;

namespace test {

TEST(faiss, RawFaissRunsProperly) {
    int d  = 64;    // dimension
    int nb = 4000;  // database size
    int nq = 10;    // nb of queries

    // Create a random database and query set
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    std::vector<float> xb(d * nb);
    std::vector<float> xq(d * nq);
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++) xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++) xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    int nlist = 100;
    int k     = 4;

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatL2 index(&res, d);
    index.add(nb, xb.data());

    // Search queries
    std::vector<idx_t> I(k * nq);
    std::vector<float> D(k * nq);
    index.search(nq, xq.data(), k, D.data(), I.data());

    printf("I=\n");
    for (int i = nq - 5; i < nq; i++) {
        for (int j = 0; j < k; j++) printf("%5ld ", I[i * k + j]);
        printf("\n");
    }

    printf("D=\n");
    for (int i = nq - 5; i < nq; i++) {
        for (int j = 0; j < k; j++) printf("%5f ", D[i * k + j]);
        printf("\n");
    }
}

TEST(faiss, RawFaissWithCagraRunsProperly) {
    int d  = 64;    // dimension
    int nb = 4000;  // database size
    int nq = 10;    // nb of queries

    // Create a random database and query set
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    std::vector<float> xb(d * nb);
    std::vector<float> xq(d * nq);
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++) xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++) xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    int nlist = 100;
    int k     = 4;

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexCagra index(&res, d);
    index.add(nb, xb.data());

    // Search queries
    std::vector<idx_t> I(k * nq);
    std::vector<float> D(k * nq);
    index.search(nq, xq.data(), k, D.data(), I.data());

    printf("I=\n");
    for (int i = nq - 5; i < nq; i++) {
        for (int j = 0; j < k; j++) printf("%5ld ", I[i * k + j]);
        printf("\n");
    }

    printf("D=\n");
    for (int i = nq - 5; i < nq; i++) {
        for (int j = 0; j < k; j++) printf("%5f ", D[i * k + j]);
        printf("\n");
    }
}

TEST(faiss, RawFaissFlatCuvsWithMaximusResources) {
    int d  = 64;    // dimension
    int nb = 4000;  // database size
    int nq = 10;    // nb of queries

    // Create a random database and query set
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    std::vector<float> xb(d * nb);
    std::vector<float> xq(d * nq);
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++) xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++) xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    int k = 4;

    auto context = maximus::make_context();

    auto res = std::make_shared<maximus::MaximusFaissGpuResources>(context);
    faiss::gpu::GpuIndexFlatConfig cfg;
    cfg.use_cuvs = true;
    faiss::gpu::GpuIndexFlatL2 index(res, d, cfg);
    index.add(nb, xb.data());

    // Search queries
    std::vector<idx_t> I(k * nq);
    std::vector<float> D(k * nq);
    index.search(nq, xq.data(), k, D.data(), I.data());

    printf("I=\n");
    for (int i = nq - 5; i < nq; i++) {
        for (int j = 0; j < k; j++) printf("%5ld ", I[i * k + j]);
        printf("\n");
    }

    printf("D=\n");
    for (int i = nq - 5; i < nq; i++) {
        for (int j = 0; j < k; j++) printf("%5f ", D[i * k + j]);
        printf("\n");
    }
}


TEST(faiss, BuildGpuIndex) {
    auto context = maximus::make_context();
    int d        = 64;
    maximus::faiss::MetricType metric = maximus::faiss::MetricType::METRIC_L2;

    // 1. Flat
    {
        std::string desc = "GPU,Flat";
        auto index = maximus::faiss::FaissIndex::factory_make(context, d, desc, metric);
        ASSERT_NE(index, nullptr);
        EXPECT_EQ(index->device_type, maximus::DeviceType::GPU);
        auto gpu_index_wrapper = std::dynamic_pointer_cast<maximus::faiss::FaissGPUIndex>(index);
        EXPECT_NE(gpu_index_wrapper, nullptr);
        
        // Verify underlying type
        auto* raw_ptr = gpu_index_wrapper->faiss_index.get();
        ASSERT_NE(raw_ptr, nullptr);
        auto* flat_ptr = dynamic_cast<faiss::gpu::GpuIndexFlat*>(raw_ptr);
        EXPECT_NE(flat_ptr, nullptr) << "Expected GpuIndexFlat for 'GPU,Flat'";
    }

    // 2. IVF256,Flat
    {
        std::string desc = "GPU,IVF256,Flat";
        auto index = maximus::faiss::FaissIndex::factory_make(context, d, desc, metric);
        ASSERT_NE(index, nullptr);
        EXPECT_EQ(index->device_type, maximus::DeviceType::GPU);
        auto gpu_index_wrapper = std::dynamic_pointer_cast<maximus::faiss::FaissGPUIndex>(index);
        EXPECT_NE(gpu_index_wrapper, nullptr);

        // Verify underlying type
        auto* raw_ptr = gpu_index_wrapper->faiss_index.get();
        ASSERT_NE(raw_ptr, nullptr);
        auto* ivf_ptr = dynamic_cast<faiss::gpu::GpuIndexIVFFlat*>(raw_ptr);
        EXPECT_NE(ivf_ptr, nullptr) << "Expected GpuIndexIVFFlat for 'GPU,IVF256,Flat'";
    }

    // 3. IVF256,PQ8
    {
        std::string desc = "GPU,IVF256,PQ8";
        auto index = maximus::faiss::FaissIndex::factory_make(context, d, desc, metric);
        ASSERT_NE(index, nullptr);
        EXPECT_EQ(index->device_type, maximus::DeviceType::GPU);
        auto gpu_index_wrapper = std::dynamic_pointer_cast<maximus::faiss::FaissGPUIndex>(index);
        EXPECT_NE(gpu_index_wrapper, nullptr);

        // Verify underlying type
        auto* raw_ptr = gpu_index_wrapper->faiss_index.get();
        ASSERT_NE(raw_ptr, nullptr);
        auto* ivfpq_ptr = dynamic_cast<faiss::gpu::GpuIndexIVFPQ*>(raw_ptr);
        EXPECT_NE(ivfpq_ptr, nullptr) << "Expected GpuIndexIVFPQ for 'GPU,IVF256,PQ8'";
    }
}


TEST_F(Sift10KTest, WrapperVsRawSearchGPU) {
    // 1. Build Index (GPU, Flat) using training data
    std::string desc      = "GPU,Flat";
    bool use_cache        = false;
    std::string cache_dir = "./index_cache_gpu_test";

    auto training_data = db->get_table("base");

    auto index = maximus::faiss::FaissIndex::build(context,
                                                   training_data,
                                                   "vector",
                                                   desc,
                                                   maximus::VectorDistanceMetric::L2,
                                                   use_cache,
                                                   cache_dir);
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(index->device_type, maximus::DeviceType::GPU);
    auto gpu_index = std::dynamic_pointer_cast<maximus::faiss::FaissGPUIndex>(index);
    ASSERT_NE(gpu_index, nullptr);

    // 2. Prepare Query Data (Arrow -> DeviceTablePtr -> CudfTablePtr)
    // GetQueryTableBatch returns a TableBatch with "qid", "qvector"
    auto query_batch = GetQueryTableBatch();

    // Use DeviceTablePtr to handle conversion
    maximus::DeviceTablePtr device_table(query_batch);
    // Convert to Cudf Table (moves data to GPU)
    // query_schema has fields: "qid", "qvector"
    device_table.convert_to<maximus::CudfTablePtr>(context, query_schema);

    ASSERT_TRUE(device_table.is_cudf_table());
    auto cudf_table = device_table.as_cudf_table();

    // Extract the vector column view (column index 1 is "qvector")
    // Note: The schema order in GetQueryTableBatch/query_schema is qid, qvector.
    auto col_view = cudf_table->view().column(1);

    // 3. Wrapper Search (GPU)
    int k = 10;
    int nq = col_view.size();

    auto mr     = rmm::mr::get_current_device_resource();
    auto stream = cudf::get_default_stream().value();

    // Allocate output columns
    auto labels_col =
        maximus::make_device_column<int64_t>(std::vector<int64_t>(nq * k), mr, stream);
    auto dists_col =
        maximus::make_device_column<float>(std::vector<float>(nq * k), mr, stream);

    // Call wrapper search overloads for cuDF
    gpu_index->search(col_view, k, *dists_col, *labels_col);

    // 4. Raw FAISS Search (Host-side for verification)
    // We compare against the standard "search" method which we know works (and RawFaissRunsProperly tests verified basic FAISS)
    // We use the same query data but from Host (query_vectors from fixture)

    // Raw FAISS search
    std::vector<float> distances_r(nq * k);
    std::vector<faiss::idx_t> labels_r(nq * k);

    // query_vectors is std::vector<float> from fixture (flat)
    float* xq = query_vectors.data();
    ASSERT_NE(xq, nullptr);

    // FaissGPUIndex wraps a gpu::GpuIndex. Its search() method accepts host pointers.
    gpu_index->faiss_index->search(nq, xq, k, distances_r.data(), labels_r.data());

    // 5. Compare Results
    // Copy GPU results to host
    auto labels_w_host =
        maximus::copy_device_column_to_host<int64_t>(labels_col->view(), stream);
    auto dists_w_host =
        maximus::copy_device_column_to_host<float>(dists_col->view(), stream);
    maximus::sync_stream(stream);

    ASSERT_EQ(labels_w_host.size(), nq * k);
    ASSERT_EQ(dists_w_host.size(), nq * k);

    for (int i = 0; i < nq * k; ++i) {
        EXPECT_EQ(labels_w_host[i], static_cast<int64_t>(labels_r[i]))
            << "label mismatch at global index " << i;
        EXPECT_NEAR(dists_w_host[i], distances_r[i], 1e-4)
            << "distance mismatch at global index " << i;
    }
}



TEST(faiss, RawFaissFlatWithMaximusResources) {
    int d  = 64;    // dimension
    int nb = 4000;  // database size
    int nq = 10;    // nb of queries

    // Create a random database and query set
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    std::vector<float> xb(d * nb);
    std::vector<float> xq(d * nq);
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++) xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++) xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }
    int k = 4;

    auto context = maximus::make_context();

    auto res = std::make_shared<maximus::MaximusFaissGpuResources>(context);
    faiss::gpu::GpuIndexFlatConfig cfg;
    cfg.use_cuvs = false;
    faiss::gpu::GpuIndexFlatL2 index(res, d, cfg);
    index.add(nb, xb.data());

    // Search queries
    std::vector<idx_t> I(k * nq);
    std::vector<float> D(k * nq);
    index.search(nq, xq.data(), k, D.data(), I.data());

    printf("I=\n");
    for (int i = nq - 5; i < nq; i++) {
        for (int j = 0; j < k; j++) printf("%5ld ", I[i * k + j]);
        printf("\n");
    }

    printf("D=\n");
    for (int i = nq - 5; i < nq; i++) {
        for (int j = 0; j < k; j++) printf("%5f ", D[i * k + j]);
        printf("\n");
    }
}

TEST(faiss, JoinOperatorWithGPUIndex) {
    // Parameters
    std::string path = std::string(PROJECT_SOURCE_DIR) + "/tests/sift/parquet10K";

    // ===============================================
    //     CREATING THE INPUT
    // ===============================================
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    auto data_table_schema =
        std::make_shared<maximus::Schema>(std::vector<std::shared_ptr<arrow::Field>>{
            arrow::field("id", arrow::int32()),
            arrow::field("vector", maximus::embeddings_list(arrow::int32(), 128))});

    auto query_table_schema =
        std::make_shared<maximus::Schema>(std::vector<std::shared_ptr<arrow::Field>>{
            arrow::field("id", arrow::int32()),
            arrow::field("vector", maximus::embeddings_list(arrow::int32(), 128))});

    db->load_table("base", data_table_schema, {"id", "vector"}, maximus::DeviceType::CPU);
    db->load_table("query", query_table_schema, {"id", "vector"}, maximus::DeviceType::CPU);
    auto data_table                                             = db->get_table("base");
    auto query_table                                            = db->get_table("query");
    std::vector<std::shared_ptr<maximus::Schema>> input_schemas = {
        data_table.as_table()->get_schema(), query_table.as_table()->get_schema()};
    assert(
        data_table.as_table()->get_schema()->get_schema()->Equals(input_schemas[0]->get_schema()));
    assert(
        query_table.as_table()->get_schema()->get_schema()->Equals(input_schemas[1]->get_schema()));
    data_table.convert_to<maximus::TableBatchPtr>(db->get_context(), input_schemas[0]);
    query_table.convert_to<maximus::TableBatchPtr>(db->get_context(), input_schemas[1]);
    auto result_schema = std::make_shared<maximus::Schema>(
        arrow::FieldVector{arrow::field("id", arrow::int32()), arrow::field("id", arrow::int32())});

    // ===============================================
    //     SETTING UP THE GPU INDEX
    // ===============================================
    auto column  = data_table.as_table_batch()->get_table_batch()->GetColumnByName("vector");
    auto vectors = std::static_pointer_cast<maximus::EmbeddingsArray>(column);
    auto d       = maximus::embedding_dimension(column);

    faiss::gpu::StandardGpuResources res;
    auto raw_index = std::make_unique<faiss::gpu::GpuIndexCagra>(&res, d);

    auto ctx = db->get_context();

    auto vector_index =
        std::make_shared<maximus::faiss::FaissGPUIndex>(ctx, d, std::move(raw_index));

    vector_index->train(*vectors);
    vector_index->add(*vectors);

    // ===============================================
    //     CREATING THE OPERATOR
    // ===============================================
    auto properties = std::make_shared<maximus::VectorJoinIndexedProperties>(
        arrow::FieldRef("vector"),  // data vector column
        arrow::FieldRef("vector"),  // query vector column
        vector_index,               // index
        2,                          // k = 2,
        std::nullopt,               // radius
        nullptr);                   // index_parameters

    auto join_operator = std::make_shared<maximus::faiss::JoinIndexedOperator>(
        db->get_context(), std::move(input_schemas), properties);
    join_operator->next_op_type     = maximus::PhysicalOperatorType::TABLE_SINK;
    join_operator->next_engine_type = maximus::EngineType::NATIVE;

    // ===============================================
    //     PUSH THE BATCH TO THE OPERATOR
    // ===============================================
    join_operator->add_input(data_table, 0);
    join_operator->add_input(query_table, 1);
    join_operator->no_more_input(0);
    join_operator->no_more_input(1);

    maximus::DeviceTablePtr output;
    auto output_batch = join_operator->export_next_batch();
    output_batch.convert_to<maximus::TablePtr>(db->get_context(), result_schema);
    output_batch.as_table()->slice(0, 5)->print();
}

void print_knn_2d(std::vector<int64_t> const& labels,
                  std::vector<float> const& distances,
                  int64_t nq,
                  int64_t K) {
    std::cout << "\n===== KNN (nq=" << nq << ", K=" << K << ") =====\n";

    for (int q = 0; q < nq; q++) {
        std::cout << "Query " << q << ":\n";
        std::cout << "  k | label | distance\n";
        std::cout << "  ---------------------\n";

        for (int k = 0; k < K; k++) {
            int idx = q * K + k;

            std::cout << "  " << std::setw(1) << k << " | " << std::setw(5) << labels[idx] << " | "
                      << std::fixed << std::setprecision(6) << std::setw(9) << distances[idx]
                      << "\n";
        }
        std::cout << "\n";
    }
}

void print_knn_search_result(std::vector<int64_t> const& left_indices,
                             std::vector<int64_t> const& right_labels,
                             std::vector<float> const& distances) {
    if (left_indices.empty()) {
        std::cout << "KNN Search result is empty.\n";
        return;
    }

    std::cout << "\n===== KNN Search Result =====\n";
    std::cout << "q | label | distance\n";
    std::cout << "----------------------\n";

    for (size_t i = 0; i < left_indices.size(); ++i) {
        std::cout << std::setw(1) << left_indices[i] << " | " << std::setw(5) << right_labels[i]
                  << " | " << std::fixed << std::setprecision(6) << std::setw(9) << distances[i]
                  << "\n";
    }

    // Optional: group by query for clarity
    std::cout << "\nGrouped by query:\n";
    int current_query = left_indices[0];
    std::cout << "Query " << current_query << ":\n";
    std::cout << "  k | label | distance\n";
    std::cout << "  ---------------------\n";
    int k = 0;
    for (size_t i = 0; i < left_indices.size(); ++i) {
        if (left_indices[i] != current_query) {
            current_query = left_indices[i];
            k             = 0;
            std::cout << "\nQuery " << current_query << ":\n";
            std::cout << "  k | label | distance\n";
            std::cout << "  ---------------------\n";
        }
        std::cout << "  " << k++ << " | " << std::setw(5) << right_labels[i] << " | " << std::fixed
                  << std::setprecision(6) << std::setw(9) << distances[i] << "\n";
    }
    std::cout << "\n";
}

// Computes squared L2 distance between two vectors
float squared_l2(const std::vector<float>& a, const std::vector<float>& b, int64_t D) {
    float sum = 0.f;
    for (int64_t i = 0; i < D; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Compute KNN indices and distances on CPU
void cpu_knn(const std::vector<float>& db,
             int64_t nb,
             const std::vector<float>& queries,
             int64_t nq,
             int64_t D,
             int64_t K,
             std::vector<int64_t>& out_indices,
             std::vector<float>& out_distances) {
    out_indices.resize(nq * K);
    out_distances.resize(nq * K);

    for (int64_t q = 0; q < nq; q++) {
        std::vector<std::pair<float, int64_t>> dist_idx;
        for (int64_t i = 0; i < nb; i++) {
            std::vector<float> db_vec(db.begin() + i * D, db.begin() + (i + 1) * D);
            std::vector<float> q_vec(queries.begin() + q * D, queries.begin() + (q + 1) * D);
            float d = squared_l2(db_vec, q_vec, D);
            dist_idx.push_back({d, i});
        }
        // Sort by distance
        std::sort(dist_idx.begin(), dist_idx.end(), [](auto const& a, auto const& b) {
            return a.first < b.first;
        });
        // Pick top K
        for (int64_t k = 0; k < K; k++) {
            out_distances[q * K + k] = dist_idx[k].first;
            out_indices[q * K + k]   = dist_idx[k].second;
        }
    }
}

TEST(KnnExhaustiveGpu, Simple2D) {
    constexpr int64_t D  = 2;
    constexpr int64_t K  = 2;
    constexpr int64_t nq = 2;

    auto context        = maximus::make_context();
    auto& cuda_mr       = context->cuda_mr;
    auto& pool_mr       = context->pool_mr;
    cudaStream_t stream = cudf::get_default_stream().value();

    // -----------------------------------------
    // Database vectors as LIST<FLOAT32>
    // -----------------------------------------
    std::vector<float> db_flat = {0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 1.f, 1.f};  // 4 vectors, D=2

    // Create child column (FLOAT32)
    auto db_child = maximus::make_device_column<float>(db_flat, &pool_mr, stream);

    // Offsets for LIST column: 4 vectors -> offsets = [0,2,4,6,8]
    std::vector<int32_t> offsets = {0, 2, 4, 6, 8};
    auto offsets_col             = maximus::make_device_column<int32_t>(offsets, &pool_mr, stream);

    // LIST column of FLOAT32
    auto db_col = cudf::make_lists_column(4,  // 4 lists
                                          std::move(offsets_col),
                                          std::move(db_child),
                                          0,
                                          rmm::device_buffer{},
                                          stream,
                                          &pool_mr);

    // -----------------------------------------
    // Query vectors as LIST<FLOAT32>
    // -----------------------------------------
    std::vector<float> queries_flat = {0.f, 0.f, 1.f, 1.f};  // 2 queries, D=2
    auto queries_child = maximus::make_device_column<float>(queries_flat, &pool_mr, stream);

    std::vector<int32_t> q_offsets = {0, 2, 4};  // 2 queries -> offsets
    auto q_offsets_col = maximus::make_device_column<int32_t>(q_offsets, &pool_mr, stream);

    auto query_col = cudf::make_lists_column(2,
                                             std::move(q_offsets_col),
                                             std::move(queries_child),
                                             0,
                                             rmm::device_buffer{},
                                             stream,
                                             &pool_mr);

    // -----------------------------------------
    // Run GPU KNN
    // -----------------------------------------
    maximus::faiss::gpu::GpuSearchResult result = maximus::faiss::gpu::knn_search_gpu(
        db_col->view(), query_col->view(), D, K, faiss::METRIC_L2, &pool_mr, stream);

    maximus::sync_stream(stream);

    ASSERT_NE(result.distances, nullptr);
    ASSERT_NE(result.right_indices, nullptr);
    ASSERT_NE(result.left_indices, nullptr);

    // -----------------------------------------
    // Copy results back to host
    // -----------------------------------------
    auto h_dist = maximus::copy_device_column_to_host<float>(result.distances->view(), stream);
    auto h_lab = maximus::copy_device_column_to_host<int64_t>(result.right_indices->view(), stream);
    auto h_left = maximus::copy_device_column_to_host<int64_t>(result.left_indices->view(), stream);

    maximus::sync_stream(stream);

    // Print results (optional)
    print_knn_2d(h_lab, h_dist, nq, K);

    // -----------------------------------------
    // CPU reference
    // -----------------------------------------
    std::vector<int64_t> cpu_labels;
    std::vector<float> cpu_distances;
    cpu_knn(db_flat, db_flat.size() / D, queries_flat, nq, D, K, cpu_labels, cpu_distances);

    // Compare GPU results to CPU results
    ASSERT_EQ(h_lab.size(), cpu_labels.size());
    ASSERT_EQ(h_dist.size(), cpu_distances.size());

    for (size_t i = 0; i < h_lab.size(); i++) {
        EXPECT_EQ(h_lab[i], cpu_labels[i]);
        EXPECT_NEAR(h_dist[i], cpu_distances[i], 1e-5);
    }
}

TEST(KnnSearchGpu, BasicFiltering) {
    constexpr int64_t D  = 2;  // dimensions
    constexpr int64_t nq = 3;  // number of queries
    constexpr int64_t K  = 3;  // nearest neighbors

    auto context        = maximus::make_context();
    auto& cuda_mr       = context->cuda_mr;
    auto& pool_mr       = context->pool_mr;
    cudaStream_t stream = cudf::get_default_stream().value();

    // -----------------------------------------
    // Database vectors (4 points)
    // -----------------------------------------
    std::vector<float> db_flat = {0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 1.f, 1.f};
    auto db_child              = maximus::make_device_column<float>(db_flat, &pool_mr, stream);

    // Offsets for LIST column: 4 vectors -> offsets = [0,2,4,6,8]
    std::vector<int32_t> db_offsets = {0, 2, 4, 6, 8};
    auto db_offsets_col = maximus::make_device_column<int32_t>(db_offsets, &pool_mr, stream);

    auto db_col = cudf::make_lists_column(4,  // 4 lists
                                          std::move(db_offsets_col),
                                          std::move(db_child),
                                          0,
                                          rmm::device_buffer{},
                                          stream,
                                          &pool_mr);

    // -----------------------------------------
    // Query vectors (3 points)
    // -----------------------------------------
    std::vector<float> queries_flat = {0.f, 0.f, 1.f, 1.f, 2.f, 2.f};
    auto query_child = maximus::make_device_column<float>(queries_flat, &pool_mr, stream);

    // Offsets for LIST column: 3 queries -> offsets = [0,2,4,6]
    std::vector<int32_t> q_offsets = {0, 2, 4, 6};
    auto q_offsets_col = maximus::make_device_column<int32_t>(q_offsets, &pool_mr, stream);

    auto query_col = cudf::make_lists_column(3,  // 3 lists
                                             std::move(q_offsets_col),
                                             std::move(query_child),
                                             0,
                                             rmm::device_buffer{},
                                             stream,
                                             &pool_mr);

    // -----------------------------------------
    // Run GPU KNN (single LIST column)
    // -----------------------------------------
    maximus::faiss::gpu::GpuSearchResult result = maximus::faiss::gpu::knn_search_gpu(
        db_col->view(), query_col->view(), D, K, faiss::METRIC_L2, &pool_mr, stream);

    maximus::sync_stream(stream);

    ASSERT_NE(result.left_indices, nullptr);
    ASSERT_NE(result.right_indices, nullptr);
    ASSERT_NE(result.distances, nullptr);

    // -----------------------------------------
    // Copy results back to host
    // -----------------------------------------
    auto h_left = maximus::copy_device_column_to_host<int64_t>(result.left_indices->view(), stream);
    auto h_right =
        maximus::copy_device_column_to_host<int64_t>(result.right_indices->view(), stream);
    auto h_dist = maximus::copy_device_column_to_host<float>(result.distances->view(), stream);
    maximus::sync_stream(stream);

    print_knn_search_result(h_left, h_right, h_dist);

    ASSERT_EQ(h_left.size(), h_right.size());
    ASSERT_EQ(h_left.size(), h_dist.size());

    // -----------------------------------------
    // Check that all labels >= 0
    // -----------------------------------------
    for (auto lbl : h_right) {
        EXPECT_TRUE(lbl >= 0);
    }

    // Check left indices are valid (0 <= idx < nq)
    for (auto li : h_left) {
        EXPECT_TRUE(li >= 0);
        EXPECT_TRUE(li < nq);
    }

    // Ensure distances per query are non-decreasing
    int pos = 0;
    for (int q = 0; q < nq; q++) {
        std::vector<float> d;
        while (pos < h_left.size() && h_left[pos] == q) {
            d.push_back(h_dist[pos]);
            pos++;
        }
        for (size_t i = 1; i < d.size(); i++) {
            EXPECT_LE(d[i - 1], d[i] + 1e-6)
                << "Query " << q << ", k=" << i << ": distance not sorted (" << d[i - 1] << " > "
                << d[i] << ")";
        }
    }

    // -----------------------------------------
    // Compute CPU reference
    // -----------------------------------------
    std::vector<int64_t> cpu_indices;
    std::vector<float> cpu_distances;
    cpu_knn(db_flat, db_flat.size() / D, queries_flat, nq, D, K, cpu_indices, cpu_distances);

    // Compare GPU results to CPU results
    for (size_t i = 0; i < h_left.size(); i++) {
        EXPECT_EQ(h_right[i], cpu_indices[i]);
        EXPECT_NEAR(h_dist[i], cpu_distances[i], 1e-5);
    }
}

TEST(PairwiseDistancesGpu, Simple2D) {
    constexpr int64_t D  = 2;
    constexpr int64_t nb = 4;
    constexpr int64_t nq = 3;

    auto context        = maximus::make_context();
    auto& cuda_mr       = context->cuda_mr;
    auto& pool_mr       = context->pool_mr;
    cudaStream_t stream = cudf::get_default_stream().value();

    // -----------------------------------------
    // Database vectors (LIST<FLOAT32>)
    // -----------------------------------------
    std::vector<float> db_flat = {0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 1.f, 1.f};
    auto db_child              = maximus::make_device_column<float>(db_flat, &pool_mr, stream);

    std::vector<int32_t> db_offsets = {0, 2, 4, 6, 8};
    auto db_offsets_col = maximus::make_device_column<int32_t>(db_offsets, &pool_mr, stream);

    auto db_col = cudf::make_lists_column(nb,
                                          std::move(db_offsets_col),
                                          std::move(db_child),
                                          0,
                                          rmm::device_buffer{},
                                          stream,
                                          &pool_mr);

    // -----------------------------------------
    // Query vectors (LIST<FLOAT32>)
    // -----------------------------------------
    std::vector<float> queries_flat = {0.f, 0.f, 1.f, 1.f, 2.f, 2.f};
    auto query_child = maximus::make_device_column<float>(queries_flat, &pool_mr, stream);

    std::vector<int32_t> q_offsets = {0, 2, 4, 6};
    auto q_offsets_col = maximus::make_device_column<int32_t>(q_offsets, &pool_mr, stream);

    auto query_col = cudf::make_lists_column(nq,
                                             std::move(q_offsets_col),
                                             std::move(query_child),
                                             0,
                                             rmm::device_buffer{},
                                             stream,
                                             &pool_mr);

    // -----------------------------------------
    // Run GPU pairwise distances
    // -----------------------------------------
    auto distances_col = maximus::faiss::gpu::pairwise_distances_gpu(
        db_col->view(), query_col->view(), D, faiss::METRIC_L2, &pool_mr, stream);

    maximus::sync_stream(stream);
    ASSERT_NE(distances_col, nullptr);

    // -----------------------------------------
    // NEW: Validate flat output (NOT LIST)
    // -----------------------------------------
    EXPECT_EQ(distances_col->type().id(), cudf::type_id::FLOAT32);
    EXPECT_EQ(distances_col->size(), nq * nb);

    // -----------------------------------------
    // Copy flat distances to host
    // -----------------------------------------
    auto h_distances = maximus::copy_device_column_to_host<float>(distances_col->view(), stream);
    maximus::sync_stream(stream);

    // -----------------------------------------
    // CPU reference
    // -----------------------------------------
    std::vector<float> cpu_distances(nq * nb);
    for (int q = 0; q < nq; q++) {
        for (int b = 0; b < nb; b++) {
            float dx                  = queries_flat[q * D + 0] - db_flat[b * D + 0];
            float dy                  = queries_flat[q * D + 1] - db_flat[b * D + 1];
            cpu_distances[q * nb + b] = dx * dx + dy * dy;
        }
    }

    // -----------------------------------------
    // Compare GPU vs CPU (row-major)
    // -----------------------------------------
    ASSERT_EQ(h_distances.size(), cpu_distances.size());
    for (size_t i = 0; i < h_distances.size(); i++) {
        EXPECT_NEAR(h_distances[i], cpu_distances[i], 1e-5);
    }
}

TEST(faiss, KeepVectorColumnFlagsGPU) {
    // Test that keep_data_vector_column and keep_query_vector_column flags work correctly
    // for both JoinExhaustiveOperator and JoinIndexedOperator across all 4 combinations.
    // We don't have to provide functionality for dropping vectors cols as it can be a project
    // operator afterwards, but it's a very common use case so it's easiest this way
    // ===============================================
    //     CREATING THE INPUT
    // ===============================================
    auto data_schema = std::make_shared<maximus::Schema>(
        arrow::FieldVector{arrow::field("dvector", maximus::embeddings_list(arrow::float32(), 5)),
                           arrow::field("category", arrow::utf8()),
                           arrow::field("data_id", arrow::int32())});
    auto query_schema = std::make_shared<maximus::Schema>(
        arrow::FieldVector{arrow::field("qid", arrow::int32()),
                           arrow::field("qvector", maximus::embeddings_list(arrow::float32(), 5)),
                           arrow::field("query_label", arrow::utf8())});

    auto context = maximus::make_context();

    // Helper to create fresh copies of input tables for each test case
    auto make_data_table = [&]() {
        maximus::TableBatchPtr table;
        auto status = maximus::TableBatch::from_json(context,
                                                     data_schema,
                                                     {R"([
                [[1.00, 1.00, 1.00, 1.00, 1.00], "a", 100],
                [[2.00, 2.00, 2.00, 2.00, 2.00], "b", 200]
            ])"},
                                                     table);
        CHECK_STATUS(status);
        return table;
    };

    auto make_query_table = [&]() {
        maximus::TableBatchPtr table;
        auto status = maximus::TableBatch::from_json(context,
                                                     query_schema,
                                                     {R"([
                [0, [1.01, 0.99, 1.00, 1.02, 0.98], "query_0"]
            ])"},
                                                     table);
        CHECK_STATUS(status);
        return table;
    };

    // Build a simple GPU flat index for indexed operator tests
    const int d = 5;
    faiss::gpu::StandardGpuResources res;
    auto raw_index = std::make_unique<faiss::gpu::GpuIndexFlat>(&res, d, faiss::METRIC_L2);
    auto vector_index =
        std::make_shared<maximus::faiss::FaissGPUIndex>(context, d, std::move(raw_index));
    auto sample_data = make_data_table();
    auto column      = sample_data->get_table_batch()->GetColumnByName("dvector");
    auto vectors     = std::static_pointer_cast<maximus::EmbeddingsArray>(column);
    // train should do nothing for "Flat" index, but call it anyway to test. If it fails you need to handle it.
    vector_index->train(*vectors);
    vector_index->add(*vectors);
    auto index_params = std::make_shared<maximus::faiss::FaissSearchParameters>(
        std::make_shared<::faiss::SearchParameters>());

    // Test configurations: (keep_data_vector, keep_query_vector)
    std::vector<std::pair<bool, bool>> configs = {
        {false, false}, {true, false}, {false, true}, {true, true}};

    for (auto [keep_dvec, keep_qvec] : configs) {
        SCOPED_TRACE(testing::Message()
                     << "Testing config: keep_dvec=" << keep_dvec << ", keep_qvec=" << keep_qvec);
        // ===============================================
        //     TEST GPU INDEXED OPERATOR
        // ===============================================
        {
            auto data_table  = make_data_table();
            auto query_table = make_query_table();

            auto properties =
                std::make_shared<maximus::VectorJoinIndexedProperties>(arrow::FieldRef("dvector"),
                                                                       arrow::FieldRef("qvector"),
                                                                       vector_index,
                                                                       1,             // k = 1
                                                                       std::nullopt,  // radius none
                                                                       index_params,
                                                                       keep_dvec,
                                                                       keep_qvec);

            std::vector<std::shared_ptr<maximus::Schema>> input_schemas = {data_schema,
                                                                           query_schema};

            auto join_operator = std::make_shared<maximus::faiss::gpu::JoinIndexedOperator>(
                context, std::move(input_schemas), properties);
            join_operator->next_op_type     = maximus::PhysicalOperatorType::TABLE_SINK;
            join_operator->next_engine_type = maximus::EngineType::NATIVE;

            join_operator->add_input(maximus::DeviceTablePtr(std::move(data_table)), 0);
            join_operator->add_input(maximus::DeviceTablePtr(std::move(query_table)), 1);
            join_operator->no_more_input(0);
            join_operator->no_more_input(1);

            maximus::DeviceTablePtr output = join_operator->export_next_batch();
            // Convert GPU output to Arrow table for inspection
            output.convert_to<maximus::ArrowTablePtr>(context, join_operator->output_schema);
            auto arrow_table   = output.as_arrow_table();
            auto output_schema = arrow_table->schema();

            // 1. Input columns besides the vectors should always be present
            std::vector<std::string> required_cols = {"qid", "query_label", "category", "data_id"};
            for (const auto& col : required_cols) {
                EXPECT_NE(output_schema->GetFieldByName(col), nullptr)
                    << "Missing required column: " << col;
            }

            // 2. Check the conditional columns (keep specific logic separate)
            EXPECT_EQ(output_schema->GetFieldByName("dvector") != nullptr, keep_dvec);
            EXPECT_EQ(output_schema->GetFieldByName("qvector") != nullptr, keep_qvec);
        }

        // ===============================================
        //     TEST GPU EXHAUSTIVE OPERATOR
        // ===============================================
        {
            auto data_table  = make_data_table();
            auto query_table = make_query_table();

            auto properties = std::make_shared<maximus::VectorJoinExhaustiveProperties>(
                arrow::FieldRef("dvector"),
                arrow::FieldRef("qvector"),
                1,  // k = 1
                std::nullopt,
                maximus::VectorDistanceMetric::L2,
                keep_dvec,
                keep_qvec);

            std::vector<std::shared_ptr<maximus::Schema>> input_schemas = {data_schema,
                                                                           query_schema};

            auto join_operator = std::make_shared<maximus::faiss::gpu::JoinExhaustiveOperator>(
                context, std::move(input_schemas), properties);
            join_operator->next_op_type     = maximus::PhysicalOperatorType::TABLE_SINK;
            join_operator->next_engine_type = maximus::EngineType::NATIVE;

            join_operator->add_input(maximus::DeviceTablePtr(std::move(data_table)), 0);
            join_operator->add_input(maximus::DeviceTablePtr(std::move(query_table)), 1);
            join_operator->no_more_input(0);
            join_operator->no_more_input(1);

            maximus::DeviceTablePtr output = join_operator->export_next_batch();
            // Convert GPU output to Arrow table for inspection
            output.convert_to<maximus::ArrowTablePtr>(context, join_operator->output_schema);
            auto arrow_table   = output.as_arrow_table();
            auto output_schema = arrow_table->schema();


            // 1. Input columns besides the vectors should always be present
            std::vector<std::string> required_cols = {"qid", "query_label", "category", "data_id"};
            for (const auto& col : required_cols) {
                EXPECT_NE(output_schema->GetFieldByName(col), nullptr)
                    << "Missing required column: " << col;
            }

            // 2. Check the conditional columns (keep specific logic separate)
            EXPECT_EQ(output_schema->GetFieldByName("dvector") != nullptr, keep_dvec);
            EXPECT_EQ(output_schema->GetFieldByName("qvector") != nullptr, keep_qvec);
        }
    }
}


// =========================================================================
// SIFT10K GPU Index Tests - Validates against real groundtruth
// =========================================================================
// Recommended method = Use the FaissIndexBuilder class

// Here I am testing various ways of creating GPU FAISS indexes
// and verifying they all produce correct results on SIFT10K data.
// - They are all valid but at different levels of abstraction.

// METHOD 1: Raw FAISS FlatL2 with StandardGpuResources
// Expected: 100% recall (exact search)
TEST_F(Sift10KTest, RawFaissStandardResources) {
    std::cout << "Testing: Raw FAISS FlatL2 + StandardGpuResources" << std::endl;

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatL2 index(&res, d);
    index.add(nb, db_vectors.data());

    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);
    index.search(nq, query_vectors.data(), k, distances.data(), labels.data());

    VerifyDistancesSorted(distances, k);
    VerifyExactResults(labels, k);
}

// METHOD 2: Raw FAISS FlatL2 with MaximusFaissGpuResources
// Expected: 100% recall (exact search)
TEST_F(Sift10KTest, RawFaissMaximusResources) {
    std::cout << "Testing: Raw FAISS FlatL2 + MaximusFaissGpuResources" << std::endl;

    auto res = std::make_shared<maximus::MaximusFaissGpuResources>(context);
    faiss::gpu::GpuIndexFlatConfig cfg;
    cfg.use_cuvs = true;
    faiss::gpu::GpuIndexFlatL2 index(res, d, cfg);
    index.verbose = false;
    index.add(nb, db_vectors.data());

    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);
    index.search(nq, query_vectors.data(), k, distances.data(), labels.data());

    VerifyDistancesSorted(distances, k);
    VerifyExactResults(labels, k);
}

// METHOD 3: FaissGPUIndex wrapper with raw FAISS index + custom config
// Expected: 100% recall (exact search)
TEST_F(Sift10KTest, WrappedFaissWithConfig) {
    std::cout << "Testing: FaissGPUIndex wrapper + GpuIndexFlat with use_cuvs=true" << std::endl;

    auto res = std::make_shared<maximus::MaximusFaissGpuResources>(context);
    faiss::gpu::GpuIndexFlatConfig cfg;
    // data internally stored interleaved for cuVS but input queries still contiguous
    // index building should be largest win w/ cuvs enabled
    cfg.use_cuvs = true;

    auto raw_index = std::make_unique<faiss::gpu::GpuIndexFlat>(res, d, faiss::METRIC_L2, cfg);
    auto vector_index =
        std::make_shared<maximus::faiss::FaissGPUIndex>(context, d, std::move(raw_index));

    vector_index->train(*db_array);
    vector_index->add(*db_array);

    auto dist_builder  = std::make_shared<arrow::FloatBuilder>();
    auto label_builder = std::make_shared<arrow::Int64Builder>();
    CHECK_STATUS(dist_builder->AppendEmptyValues(nq * k));
    CHECK_STATUS(label_builder->AppendEmptyValues(nq * k));
    std::shared_ptr<arrow::FloatArray> dists;
    std::shared_ptr<arrow::Int64Array> labs;
    CHECK_STATUS(dist_builder->Finish(&dists));
    CHECK_STATUS(label_builder->Finish(&labs));

    vector_index->search(*query_array, k, *dists, *labs, nullptr);

    VerifyArrowResults(*labs, *dists, /*is_exact=*/true, "FlatL2");
}

// METHOD 4: Factory String "GPU,Flat"
// Expected: 100% recall (exact search)
TEST_F(Sift10KTest, FactoryStringFlat) {
    std::cout << "Testing: FaissGPUIndex factory string 'GPU,Flat'" << std::endl;

    auto vector_index =
        std::make_shared<maximus::faiss::FaissGPUIndex>(context, d, "GPU,Flat", faiss::METRIC_L2);

    vector_index->train(*db_array);
    vector_index->add(*db_array);

    auto dist_builder  = std::make_shared<arrow::FloatBuilder>();
    auto label_builder = std::make_shared<arrow::Int64Builder>();
    CHECK_STATUS(dist_builder->AppendEmptyValues(nq * k));
    CHECK_STATUS(label_builder->AppendEmptyValues(nq * k));
    std::shared_ptr<arrow::FloatArray> dists;
    std::shared_ptr<arrow::Int64Array> labs;
    CHECK_STATUS(dist_builder->Finish(&dists));
    CHECK_STATUS(label_builder->Finish(&labs));

    vector_index->search(*query_array, k, *dists, *labs, nullptr);

    VerifyArrowResults(*labs, *dists, /*is_exact=*/true, "GPU,Flat");
}

// METHOD 5: Factory String "GPU,Cagra" - Approximate search
// Expected: >99% recall with default search parameters w/ Cagra on sift10k
TEST_F(Sift10KTest, FactoryStringCagra) {
    std::cout << "Testing: FaissGPUIndex factory string 'GPU,Cagra'" << std::endl;

    auto vector_index = std::make_shared<maximus::faiss::FaissGPUIndex>(
        context, d, "GPU,Cagra,64,32,IVF_PQ", faiss::METRIC_L2);

    vector_index->train(*db_array);
    vector_index->add(*db_array);

    auto dist_builder  = std::make_shared<arrow::FloatBuilder>();
    auto label_builder = std::make_shared<arrow::Int64Builder>();
    CHECK_STATUS(dist_builder->AppendEmptyValues(nq * k));
    CHECK_STATUS(label_builder->AppendEmptyValues(nq * k));
    std::shared_ptr<arrow::FloatArray> dists;
    std::shared_ptr<arrow::Int64Array> labs;
    CHECK_STATUS(dist_builder->Finish(&dists));
    CHECK_STATUS(label_builder->Finish(&labs));

    vector_index->search(*query_array, k, *dists, *labs, nullptr);

    VerifyArrowResults(*labs, *dists, /*is_exact=*/false, "GPU,Cagra");
}

// METHOD 6: Raw FAISS Cagra with high itopk for better recall
// Expected: >95% recall with itopk_size=256
TEST_F(Sift10KTest, RawCagraWithHighItopk) {
    std::cout << "Testing: Raw FAISS GpuIndexCagra with itopk_size=256" << std::endl;

    faiss::gpu::StandardGpuResources res;

    // Build with default parameters
    faiss::gpu::GpuIndexCagraConfig cfg;
    cfg.intermediate_graph_degree = 64;
    cfg.graph_degree              = 32;

    faiss::gpu::GpuIndexCagra index(&res, d, faiss::METRIC_L2, cfg);
    index.add(nb, db_vectors.data());

    // Search with high itopk for better recall
    faiss::gpu::SearchParametersCagra search_params;
    search_params.itopk_size = 256;  // Higher = better recall, slower search

    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);
    index.search(nq, query_vectors.data(), k, distances.data(), labels.data(), &search_params);

    VerifyDistancesSorted(distances, k);
    ReportApproximateResults(labels, k, "Cagra (itopk=256)", /*min_expected_recall=*/95.0);
}

// METHOD 7: JoinExhaustiveOperator (exhaustive KNN search)
// Tests the operator-level interface used in query execution.
// Expected: 100% recall (exact search)
TEST_F(Sift10KTest, JoinExhaustiveOperator) {
    std::cout << "Testing: JoinExhaustiveOperator with SIFT10K" << std::endl;

    // Get pre-loaded tables from fixture (The GetDataTableBatch does the renaming)
    auto data_table_batch  = GetDataTableBatch();
    auto query_table_batch = GetQueryTableBatch();

    // Create JoinExhaustiveOperator
    auto properties = std::make_shared<maximus::VectorJoinExhaustiveProperties>(
        arrow::FieldRef("dvector"),  // data vector column
        arrow::FieldRef("qvector"),  // query vector column
        k,                           // k neighbors
        std::nullopt,                // no distance column
        maximus::VectorDistanceMetric::L2,
        false,   // keep_data_vector_column
        false);  // keep_query_vector_column

    std::vector<std::shared_ptr<maximus::Schema>> input_schemas = {data_schema, query_schema};

    auto join_operator = std::make_shared<maximus::faiss::gpu::JoinExhaustiveOperator>(
        context, std::move(input_schemas), properties);
    join_operator->next_op_type     = maximus::PhysicalOperatorType::TABLE_SINK;
    join_operator->next_engine_type = maximus::EngineType::NATIVE;

    // Add inputs and execute
    join_operator->add_input(maximus::DeviceTablePtr(std::move(data_table_batch)), 0);
    join_operator->add_input(maximus::DeviceTablePtr(std::move(query_table_batch)), 1);
    join_operator->no_more_input(0);
    join_operator->no_more_input(1);

    maximus::DeviceTablePtr output = join_operator->export_next_batch();

    // Convert to Arrow table for inspection
    output.convert_to<maximus::ArrowTablePtr>(context, join_operator->output_schema);
    auto arrow_table = output.as_arrow_table();

    std::cout << "  Output schema: " << arrow_table->schema()->ToString() << std::endl;
    std::cout << "  Output rows: " << arrow_table->num_rows() << " (expected: " << (nq * k) << ")"
              << std::endl;

    ASSERT_EQ(arrow_table->num_rows(), nq * k) << "Expected nq * k output rows";

    // Extract result labels from output
    auto qid_col     = arrow_table->GetColumnByName("qid");
    auto data_id_col = arrow_table->GetColumnByName("data_id");

    ASSERT_NE(qid_col, nullptr) << "Missing qid column in output";
    ASSERT_NE(data_id_col, nullptr) << "Missing data_id column in output";
    ASSERT_EQ(qid_col->num_chunks(), 1) << "Expected single-chunk qid column";
    ASSERT_EQ(data_id_col->num_chunks(), 1) << "Expected single-chunk data_id column";

    auto qid_array     = std::static_pointer_cast<arrow::Int32Array>(qid_col->chunk(0));
    auto data_id_array = std::static_pointer_cast<arrow::Int32Array>(data_id_col->chunk(0));

    // Extract results and verify ordering
    std::vector<idx_t> results(nq * k);
    for (int64_t i = 0; i < nq * k; i++) {
        results[i] = data_id_array->Value(i);
    }

    // Verify ordering assumption: qid should repeat k times for each query
    for (int64_t q = 0; q < nq; q++) {
        for (int ki = 0; ki < k; ki++) {
            int64_t row_idx = q * k + ki;
            ASSERT_EQ(qid_array->Value(row_idx), q)
                << "Unexpected qid at row " << row_idx << ": expected " << q << ", got "
                << qid_array->Value(row_idx);
        }
    }

    VerifyExactResults(results, k);
}

// METHOD 8: Use the FaissIndexBuilder class
// ninja -C ./build/Debug/ && ./build/Debug/tests/test.faiss_gpu --gtest_filter='Sift10KTest.FaissIndexBuilderGPUCagra'
TEST_F(Sift10KTest, FaissIndexBuilderGPUCagra) {
    // Test FaissIndexBuilder with SIFT10K dataset - GPU Flat index
    // This validates end-to-end index building with real data
    auto training_data    = db->get_table("base");
    bool use_cache        = false;
    std::string cache_dir = "./index_cache";
    auto index            = maximus::faiss::FaissIndex::build(context,
                                                   training_data,
                                                   "vector",
                                                   "GPU,Cagra,64,32,IVF_PQ",
                                                   maximus::VectorDistanceMetric::L2,
                                                   use_cache,
                                                   cache_dir);

    ASSERT_NE(index, nullptr);
    EXPECT_EQ(index->device_type, maximus::DeviceType::GPU);

    // Search using the built index
    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

    assert(typeid(*(index->faiss_index)) == typeid(faiss::gpu::GpuIndexCagra));

    index->faiss_index->search(nq, query_vectors.data(), k, distances.data(), labels.data());

    VerifyDistancesSorted(distances, k);
    ReportApproximateResults(labels, k, "Cagra (itopk=256)", /*min_expected_recall=*/95.0);
}


TEST_F(Sift10KTest, IndexCacheSaveLoadGPU) {
    // 1) Prepare training data (use fixture's preloaded table)
    auto training_data = db->get_table("base");

    // 2) Build the index with caching enabled
    bool use_cache        = true;
    std::string cache_dir = "/tmp/index_cache";
    std::string desc      = "GPU,Flat";
    // std::string desc      = "GPU,Cagra,64,32,IVF_PQ"; // TODO: Does not work
    // std::string desc      = "GPU,IVF256,Flat"; // TODO : We need to add support for GPU,IVFFlat

    std::shared_ptr<maximus::faiss::FaissIndex> index1;
    index1 = maximus::faiss::FaissIndex::build(context,
                                               training_data,
                                               "vector",
                                               desc,
                                               maximus::VectorDistanceMetric::L2,
                                               use_cache,
                                               cache_dir);


    ASSERT_NE(index1, nullptr);
    EXPECT_EQ(index1->device_type, maximus::DeviceType::GPU);

    // 3) Build/load again with the same parameters (should load from cache)
    std::shared_ptr<maximus::faiss::FaissIndex> index2;
    index2 = maximus::faiss::FaissIndex::build(context,
                                               training_data,
                                               "vector",
                                               desc,
                                               maximus::VectorDistanceMetric::L2,
                                               use_cache,
                                               cache_dir);

    ASSERT_NE(index2, nullptr);
    EXPECT_EQ(index2->device_type, maximus::DeviceType::GPU);

    // 4) Run identical searches on both indices and compare outputs
    std::vector<float> distances1(nq * k);
    std::vector<faiss::idx_t> labels1(nq * k);

    std::vector<float> distances2(nq * k);
    std::vector<faiss::idx_t> labels2(nq * k);

    // Use the underlying faiss index search (it accepts CPU query arrays)
    index1->faiss_index->search(nq, query_vectors.data(), k, distances1.data(), labels1.data());
    index2->faiss_index->search(nq, query_vectors.data(), k, distances2.data(), labels2.data());

    // 5) Verify results match exactly
    for (size_t i = 0; i < labels1.size(); ++i) {
        EXPECT_EQ(labels1[i], labels2[i]) << "mismatch at index " << i;
        EXPECT_NEAR(distances1[i], distances2[i], 1e-6) << "distance mismatch at index " << i;
    }
    //Cleanup cache directory
    std::filesystem::remove_all(cache_dir);
    std::cout << "Cleaned up cache directory: " << cache_dir << std::endl;
}

TEST_F(Sift10KTest, FrontendIndexedVectorJoinGPU) {
    // Test the indexed_vector_join frontend with GPU index description
    auto ctx           = db->get_context();
    auto training_data = db->get_table("base");
    auto data_source =
        table_source(db, "base", data_schema, {"id", "vector"}, maximus::DeviceType::GPU);
    auto query_source =
        table_source(db, "query", query_schema, {"id", "vector"}, maximus::DeviceType::GPU);

    auto data_renamed  = rename(data_source, {"id", "vector"}, {"data_id", "vector"});
    auto query_renamed = rename(query_source, {"id", "vector"}, {"query_id", "vector"});

    // index
    auto search_params = std::make_shared<maximus::faiss::FaissSearchParameters>(
        std::make_shared<::faiss::SearchParameters>());

    bool use_cache        = false;
    std::string cache_dir = "./index_cache";
    auto index            = maximus::faiss::FaissIndex::build(ctx,
                                                   training_data,
                                                   "vector",  // The column in training_data to use
                                                   "GPU,Flat",  // The index description
                                                   maximus::VectorDistanceMetric::L2,
                                                   use_cache,
                                                   cache_dir);
    // query node
    auto join_node = maximus::indexed_vector_join(data_renamed,
                                                  query_renamed,
                                                  "vector",      // data vector column name
                                                  "vector",      // query vector column name
                                                  index,         // <--- PASS THE BUILT INDEX HERE
                                                  k,             // K = 10
                                                  std::nullopt,  // no radius
                                                  search_params, // search params (required for GPU)
                                                  false,         // don't keep data vector
                                                  false,         // don't keep query vector
                                                  "distance",    // distance column name
                                                  maximus::DeviceType::GPU);

    auto qp = query_plan(table_sink(join_node));
    db->schedule(qp);
    ctx->barrier();
    auto tables = db->execute();
    ctx->barrier();

    ASSERT_EQ(tables.size(), 1);
    auto result = tables[0];
    EXPECT_EQ(result->num_rows(), nq * k);
    auto labels_col = result->get_table()->GetColumnByName("data_id");
    ASSERT_NE(labels_col, nullptr);

    auto dist_col = result->get_table()->GetColumnByName("distance");
    ASSERT_NE(dist_col, nullptr);

    // 1st way to verify results:
    std::shared_ptr<arrow::Int64Array> labels_array;
    // Cast 32 -> 64 array
    if (labels_col->type()->id() == arrow::Type::INT32) {
        auto tmp32 = maximus::arrow_array<arrow::Int32Array>(labels_col);
        arrow::Result<arrow::Datum> cast_res =
            arrow::compute::Cast(arrow::Datum(tmp32), arrow::int64());
        ASSERT_TRUE(cast_res.ok()) << cast_res.status().ToString();
        labels_array =
            std::static_pointer_cast<arrow::Int64Array>(cast_res.ValueOrDie().make_array());
    } else {
        labels_array = maximus::arrow_array<arrow::Int64Array>(labels_col);
    }
    // auto labels_array = maximus::arrow_array<arrow::Int64Array>(labels_col);
    auto dist_array = maximus::arrow_array<arrow::FloatArray>(dist_col);

    // Verify distances are non-negative and exact results for GPU Flat
    VerifyArrowResults(*labels_array, *dist_array, /*is_exact=*/true, "GPU Flat");

    // 2nd way to verify results:
    // Or if you plan to verify recall manually with std::vectr
    // Option 2: You need a) concat to single chunk and b) copy to std::vector c) use the recall verifier
    // - note that VerifyArrowResults already does a,b,c but in a different way. This is just for illustration:
    // But this second way is faster as it's a single copy, instead of per-element access in VerifyArrowResults

    // BRIDGE CODE: Arrow Arrays -> std::vectors (for recall)
    const float* raw_dists = dist_array->raw_values();
    const int64_t* raw_ids = labels_array->raw_values();
    std::vector<float> distances(raw_dists, raw_dists + dist_array->length());
    std::vector<::faiss::idx_t> labels(raw_ids, raw_ids + labels_array->length());

    VerifyDistancesSorted(distances, k);
    VerifyExactResults(labels, k);
}

TEST(faiss, LargeListArrayGPUSearchThrows) {
    // Test that GPU index search throws when using LargeListArray (int64 offsets not supported by cuDF)
    auto large_list_schema = std::make_shared<maximus::Schema>(arrow::FieldVector{
        arrow::field("vector", arrow::large_list(arrow::float32())),
        arrow::field("label", arrow::utf8())});

    auto context = maximus::make_context();
    maximus::TableBatchPtr table;
    auto status = maximus::TableBatch::from_json(context,
                                                 large_list_schema,
                                                 {R"([
            [[1.0, 2.0, 3.0, 4.0, 5.0], "a"],
            [[2.0, 3.0, 4.0, 5.0, 6.0], "b"],
            [[3.0, 4.0, 5.0, 6.0, 7.0], "c"]
        ])"},
                                                 table);
    CHECK_STATUS(status);

    // Build GPU index from DeviceTablePtr (should work with warning)
    maximus::DeviceTablePtr data_table(table);
    auto index = maximus::faiss::FaissIndex::build(context,
                                                   data_table,
                                                   "vector",
                                                   "GPU,Flat",
                                                   maximus::VectorDistanceMetric::L2,
                                                   false);
    ASSERT_NE(index, nullptr);
    EXPECT_EQ(index->device_type, maximus::DeviceType::GPU);
    EXPECT_EQ(index->faiss_index->ntotal, 3);

    // Get vector column as LargeListArray for search
    auto vector_col = table->get_table_batch()->GetColumnByName("vector");
    auto large_list_array = std::static_pointer_cast<arrow::LargeListArray>(vector_col);

    // Allocate output buffers
    int k = 2;
    int nq = large_list_array->length();
    auto pool = arrow::default_memory_pool();
    
    arrow::FloatBuilder dist_builder(pool);
    CHECK_STATUS(dist_builder.AppendEmptyValues(nq * k));
    std::shared_ptr<arrow::FloatArray> distances;
    CHECK_STATUS(dist_builder.Finish(&distances));
    
    arrow::Int64Builder label_builder(pool);
    CHECK_STATUS(label_builder.AppendEmptyValues(nq * k));
    std::shared_ptr<arrow::Int64Array> labels;
    CHECK_STATUS(label_builder.Finish(&labels));

    // Search should throw for GPU index with LargeListArray
    EXPECT_THROW({
        index->search(*large_list_array, k, *distances, *labels);
    }, std::runtime_error);
}

}  // namespace test
