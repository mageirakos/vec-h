#pragma once
#include <gtest/gtest.h>

// =========================================================================
// SIFT10K GPU Index Test Fixture
// Uses real SIFT10K dataset with groundtruth validation.
// Loads data via Database/DatabaseCatalogue for minimal boilerplate.
// =========================================================================
namespace test {
using idx_t = faiss::idx_t;

class Sift10KTest : public ::testing::Test {
protected:
    static constexpr const char* SIFT_PATH = "/tests/sift/parquet10K";

    int d       = 128;
    int64_t nb  = 10000;
    int64_t nq  = 100;
    int gt_k    = 0;
    const int k = 10;  // Test with k=10 neighbors

    // Database infrastructure
    std::shared_ptr<maximus::Database> db;
    maximus::Context context;

    // Raw vectors for FAISS tests that need float pointers
    std::vector<float> db_vectors;
    std::vector<float> query_vectors;
    std::vector<int32_t> groundtruth;

    // Arrow arrays for FaissGPUIndex wrapper tests
    std::shared_ptr<maximus::EmbeddingsArray> db_array;
    std::shared_ptr<maximus::EmbeddingsArray> query_array;

    // Schema for tables (id renamed to avoid conflicts in joins)
    std::shared_ptr<maximus::Schema> data_schema;
    std::shared_ptr<maximus::Schema> query_schema;

    void SetUp() {
        std::string path  = std::string(PROJECT_SOURCE_DIR) + SIFT_PATH;
        auto db_catalogue = maximus::make_catalogue(path);
        db                = maximus::make_database(db_catalogue);
        context           = db->get_context();

        // Schema matching parquet file structure: id (int32), vector (fixed_size_list<float32>[128])
        // Note: We load with original column names and rename when needed for joins
        auto base_schema = std::make_shared<maximus::Schema>(arrow::FieldVector{
            arrow::field("id", arrow::int32()),
            arrow::field("vector", maximus::embeddings_list(arrow::float32(), 128))});

        auto gt_schema = std::make_shared<maximus::Schema>(arrow::FieldVector{
            arrow::field("id", arrow::int32()),
            arrow::field("vector", maximus::embeddings_list(arrow::uint32(), 100))});

        // Load tables with original column names
        db->load_table("base", base_schema, {"id", "vector"}, maximus::DeviceType::CPU);
        db->load_table("query", base_schema, {"id", "vector"}, maximus::DeviceType::CPU);
        db->load_table("groundtruth", gt_schema, {"id", "vector"}, maximus::DeviceType::CPU);

        // Define schemas with renamed columns for join tests (to avoid column name conflicts)
        data_schema = std::make_shared<maximus::Schema>(arrow::FieldVector{
            arrow::field("data_id", arrow::int32()),
            arrow::field("dvector", maximus::embeddings_list(arrow::float32(), 128))});

        query_schema = std::make_shared<maximus::Schema>(arrow::FieldVector{
            arrow::field("qid", arrow::int32()),
            arrow::field("qvector", maximus::embeddings_list(arrow::float32(), 128))});

        // Extract raw float vectors for FAISS tests
        ExtractRawVectors();
    }

    // Helper to rename columns in an Arrow RecordBatch
    static std::shared_ptr<arrow::RecordBatch> RenameColumns(
        const std::shared_ptr<arrow::RecordBatch>& batch,
        const std::vector<std::string>& new_names) {
        return batch->RenameColumns(new_names).ValueOrDie();
    }

    // Helper functions to get data/query tables with renamed columns...
    // assumes single RecordBatch which is true for our test data so don't copy paste this anywhere else
    maximus::TableBatchPtr GetDataTableBatch() {
        auto table       = db->get_table("base");
        auto arrow_table = table.as_table()->get_table();
        auto batch       = arrow_table->CombineChunksToBatch().ValueOrDie();
        // Renames: id -> data_id, vector -> dvector
        auto renamed = RenameColumns(batch, {"data_id", "dvector"});
        maximus::TableBatchPtr result;
        CHECK_STATUS(maximus::TableBatch::from_record_batch(context, renamed, result));
        return result;
    }

    maximus::TableBatchPtr GetQueryTableBatch() {
        auto table       = db->get_table("query");
        auto arrow_table = table.as_table()->get_table();
        auto batch       = arrow_table->CombineChunksToBatch().ValueOrDie();
        // Renames: id -> qid, vector -> qvector
        auto renamed = RenameColumns(batch, {"qid", "qvector"});
        maximus::TableBatchPtr result;
        CHECK_STATUS(maximus::TableBatch::from_record_batch(context, renamed, result));
        return result;
    }

    // Calculate set-based recall: what fraction of results are in groundtruth top-k
    double CalculateRecall(const std::vector<idx_t>& results, int result_k) {
        int set_matches = 0;
        int total       = nq * result_k;

        for (int64_t q = 0; q < nq; q++) {
            std::unordered_set<int32_t> gt_set;
            for (int ki = 0; ki < result_k && ki < gt_k; ki++) {
                gt_set.insert(groundtruth[q * gt_k + ki]);
            }

            for (int ki = 0; ki < result_k; ki++) {
                if (gt_set.count(static_cast<int32_t>(results[q * result_k + ki])) > 0) {
                    set_matches++;
                }
            }
        }
        return 100.0 * set_matches / total;
    }

    // Verify exact search (for Exhaustive/Flat)
    void VerifyExactResults(const std::vector<idx_t>& results, int result_k) {
        double recall = CalculateRecall(results, result_k);
        std::cout << "  Recall@" << result_k << " = " << recall << "%" << std::endl;
        EXPECT_EQ(recall, 100.0) << "FlatL2 exact search should have 100% recall";
    }

    // Report approximate search (for ANN indexes)
    void ReportApproximateResults(const std::vector<idx_t>& results,
                                  int result_k,
                                  const std::string& index_type,
                                  double min_expected_recall = 99.0) {
        double recall = CalculateRecall(results, result_k);
        std::cout << "  " << index_type << " Recall@" << result_k << " = " << recall << "%"
                  << std::endl;
        EXPECT_GE(recall, min_expected_recall)
            << index_type << " should have at least " << min_expected_recall << "% recall";
    }

    // Verify distances are sorted per query
    void VerifyDistancesSorted(const std::vector<float>& distances, int result_k) {
        for (int64_t q = 0; q < nq; q++) {
            for (int ki = 1; ki < result_k; ki++) {
                EXPECT_LE(distances[q * result_k + ki - 1], distances[q * result_k + ki] + 1e-6f)
                    << "Distances not sorted for query " << q;
            }
        }
    }

    // Helper for Arrow-based tests
    void VerifyArrowResults(const arrow::Int64Array& labels,
                            const arrow::FloatArray& distances,
                            bool is_exact,
                            const std::string& index_type = "Index") {
        ASSERT_EQ(labels.length(), nq * k);


        // Copy to vector for recall calculation
        std::vector<idx_t> results(nq * k);
        for (int64_t i = 0; i < labels.length(); i++) {
            results[i] = labels.Value(i);
            // ensure non-negative ids, and distances, as well as in-bounds ids
            EXPECT_GE(labels.Value(i), 0) << "Invalid label at index " << i;
            EXPECT_LT(labels.Value(i), nb) << "Label out of bounds at index " << i;
            EXPECT_GE(distances.Value(i), 0.0f) << "Negative distance at index " << i;
        }

        if (is_exact) {
            VerifyExactResults(results, k);
        } else {
            ReportApproximateResults(results, k, index_type);
        }
    }

private:
    // Extract raw float vectors from loaded tables for raw FAISS tests
    void ExtractRawVectors() {
        auto base_table  = db->get_table("base");
        auto query_table = db->get_table("query");
        auto gt_table    = db->get_table("groundtruth");

        // Get Arrow tables (columns have original names: id, vector)
        auto base_arrow  = base_table.as_table()->get_table();
        auto query_arrow = query_table.as_table()->get_table();
        auto gt_arrow    = gt_table.as_table()->get_table();

        ASSERT_EQ(base_arrow->num_rows(), nb) << "Base table row count mismatch";
        ASSERT_EQ(query_arrow->num_rows(), nq) << "Query table row count mismatch";

        // Extract base vectors
        auto base_col = base_arrow->GetColumnByName("vector");
        ASSERT_NE(base_col, nullptr) << "Missing vector column in base table";
        ASSERT_EQ(base_col->num_chunks(), 1) << "Expected single-chunk base vectors";
        auto base_list   = std::static_pointer_cast<arrow::ListArray>(base_col->chunk(0));
        d                = base_list->value_length(0);  // Get dimension from first vector
        auto base_values = std::static_pointer_cast<arrow::FloatArray>(base_list->values());
        db_vectors.assign(base_values->raw_values(), base_values->raw_values() + nb * d);

        // Create db_array for wrapper tests
        db_array = std::static_pointer_cast<maximus::EmbeddingsArray>(base_col->chunk(0));

        // Extract query vectors
        auto query_col = query_arrow->GetColumnByName("vector");
        ASSERT_NE(query_col, nullptr) << "Missing vector column in query table";
        ASSERT_EQ(query_col->num_chunks(), 1) << "Expected single-chunk query vectors";
        auto query_list   = std::static_pointer_cast<arrow::ListArray>(query_col->chunk(0));
        auto query_values = std::static_pointer_cast<arrow::FloatArray>(query_list->values());
        query_vectors.assign(query_values->raw_values(), query_values->raw_values() + nq * d);

        // Create query_array for wrapper tests
        query_array = std::static_pointer_cast<maximus::EmbeddingsArray>(query_col->chunk(0));

        // Extract groundtruth (vector column contains uint32 indices)
        auto gt_col = gt_arrow->GetColumnByName("vector");
        ASSERT_NE(gt_col, nullptr) << "Missing groundtruth vector column";
        ASSERT_EQ(gt_col->num_chunks(), 1) << "Expected single-chunk groundtruth";
        auto gt_list   = std::static_pointer_cast<arrow::ListArray>(gt_col->chunk(0));
        gt_k           = gt_list->value_length(0);  // Get k from first row
        auto gt_values = std::static_pointer_cast<arrow::UInt32Array>(gt_list->values());
        groundtruth.resize(nq * gt_k);
        for (int64_t i = 0; i < nq * gt_k; i++) {
            groundtruth[i] = static_cast<int32_t>(gt_values->Value(i));
        }
    }
};
}  // namespace test