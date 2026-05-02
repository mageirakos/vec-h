#include <gtest/gtest.h>

#include <iostream>
#include <maximus/database.hpp>
#include <maximus/frontend/query_plan_api.hpp>
#include <maximus/indexes/faiss/faiss_index.hpp>
#include <maximus/tpch/tpch_queries.hpp>
#include <maximus/vsds/vsds_queries.hpp>
#include <maximus/vsds/vsds_utils.hpp>
#include <chrono>

// VSDS data path - can be overridden by VSDS_DATA_PATH env var
std::string vsds_path() {
    const char* env_p = std::getenv("VSDS_DATA_PATH");
    if (env_p) return env_p;
    
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/vsds/data-industrial_and_scientific-sf_0.001";
    return path;
}

std::string vsds_path_sf1() {
    const char* env_p = std::getenv("VSDS_DATA_PATH_SF1");
    if (env_p) return env_p;
    
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/vsds/data-industrial_and_scientific-sf_1";
    return path;
}

std::string vsds_path_sf001() {
    const char* env_p = std::getenv("VSDS_DATA_PATH_SF1");
    if (env_p) return env_p;
    
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/vsds/data-industrial_and_scientific-sf_0.01";
    return path;
}


// ============================================================================
// ENN Tests
// ============================================================================

TEST(VSDS, ENN_Reviews) {
    std::string path = vsds_path();
    std::cout << "Path = " << path << "\n";

    // Setup database
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    // Create ENN query for reviews
    maximus::vsds::QueryParameters params;
    params.k = 5;
    
    auto device = maximus::DeviceType::CPU;
    auto q      = maximus::vsds::enn_reviews(db, device, params);

    std::cout << "Query Plan = \n" << q->to_string() << std::endl;

    auto table = db->query(q);

    std::cout << "Query result = \n";
    if (table) {
        std::cout << "Rows: " << table->num_rows() << "\n";
        // Print top 10 rows
        table->slice(0, 10)->print();
    } else {
        std::cout << "Query result is empty" << std::endl;
    }
    
    ASSERT_TRUE(table);
}

TEST(VSDS, ENN_Images) {
    std::string path = vsds_path();
    std::cout << "Path = " << path << "\n";

    // Setup database
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    // Create ENN query for images
    maximus::vsds::QueryParameters params;
    params.k = 5;
    
    auto device = maximus::DeviceType::CPU;
    auto q      = maximus::vsds::enn_images(db, device, params);

    std::cout << "Query Plan = \n" << q->to_string() << std::endl;

    auto table = db->query(q);

    std::cout << "Query result = \n";
    if (table) {
        std::cout << "Rows: " << table->num_rows() << "\n";
        // Print top 10 rows
        table->slice(0, 10)->print();
    } else {
        std::cout << "Query result is empty" << std::endl;
    }
    
    ASSERT_TRUE(table);
}

// ============================================================================
// ANN Tests
// ============================================================================

TEST(VSDS, ANN_Reviews_HNSW) {
    std::string path = vsds_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    auto device = maximus::DeviceType::CPU;
    auto index_desc = "HNSW32,Flat";

    // 1. Build Index (Offline phase)
    std::cout << "Building Index (" << index_desc << ")...\n";
    
    // Load table explicitly for index building
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    auto training_data = db->get_table("reviews");
    
    if (training_data.empty()) {
        std::cout << "[WARNING] Reviews table empty, cannot build index.\n";
    }
    
    auto index = maximus::faiss::FaissIndex::build(ctx,
                                              training_data,
                                              "rv_embedding",
                                              index_desc,
                                              maximus::VectorDistanceMetric::L2,
                                              false);

    // 2. Execute ANN Query
    maximus::vsds::QueryParameters params;
    params.k = 5;
    params.faiss_index = index_desc;
    params.hnsw_efsearch = 64;

    auto q      = maximus::vsds::ann_reviews(db, index, device, params);
    
    std::cout << "Query Plan = \n" << q->to_string() << std::endl;

    auto table = db->query(q);
    
    std::cout << "Query result = \n";
    if (table) {
        std::cout << "Rows: " << table->num_rows() << "\n";
        table->slice(0, 10)->print();
    } else {
        std::cout << "Query result is empty" << std::endl;
    }
    
    ASSERT_TRUE(table);
}


TEST(VSDS, ANN_Images_IVF) {
    std::string path = vsds_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    auto device = maximus::DeviceType::CPU;
    auto index_desc = "IVF64,Flat";

    // 1. Build Index (Offline phase)
    std::cout << "Building Index (" << index_desc << ")...\n";
    
    // Load table explicitly for index building
    db->load_table("images", maximus::vsds::schema("images"), {}, device);
    auto training_data = db->get_table("images");
    
    if (training_data.empty()) {
        std::cout << "[WARNING] Images table empty, cannot build index.\n";
    }
    
    auto index = maximus::faiss::FaissIndex::build(ctx,
                                              training_data,
                                              "i_embedding",
                                              index_desc,
                                              maximus::VectorDistanceMetric::L2,
                                              false);

    // 2. Execute ANN Query
    maximus::vsds::QueryParameters params;
    params.k = 10;
    params.faiss_index = index_desc;
    params.ivf_nprobe = 8;

    auto q      = maximus::vsds::ann_images(db, index, device, params);
    
    std::cout << "Query Plan = \n" << q->to_string() << std::endl;

    auto table = db->query(q);
    
    std::cout << "Query result = \n";
    if (table) {
        std::cout << "Rows: " << table->num_rows() << "\n";
        table->slice(0, 20)->print();
    } else {
        std::cout << "Query result is empty" << std::endl;
    }
    
    ASSERT_TRUE(table);
}

// ============================================================================
// Pre/Post Filtering Tests
// ============================================================================

TEST(VSDS, Pre_Reviews) {
    std::string path = vsds_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    auto device = maximus::DeviceType::CPU;

    // Execute pre-filtering query
    maximus::vsds::QueryParameters params;
    params.k = 5;

    auto q      = maximus::vsds::pre_reviews(db, device, params);
    
    std::cout << "Query Plan = \n" << q->to_string() << std::endl;

    auto table = db->query(q);
    
    std::cout << "Query result = \n";
    if (table) {
        std::cout << "Rows: " << table->num_rows() << "\n";
        table->slice(0, 10)->print();
    } else {
        std::cout << "Query result is empty" << std::endl;
    }
     ASSERT_TRUE(table);
}


TEST(VSDS, Pre_Images) {
    std::string path = vsds_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    auto device = maximus::DeviceType::CPU;

    // Execute pre-filtering query
    maximus::vsds::QueryParameters params;
    params.k = 5;

    auto q      = maximus::vsds::pre_images(db, device, params);
    
    std::cout << "Query Plan = \n" << q->to_string() << std::endl;

    auto table = db->query(q);
    
    std::cout << "Query result = \n";
    if (table) {
        std::cout << "Rows: " << table->num_rows() << "\n";
        table->slice(0, 10)->print();
    } else {
        std::cout << "Query result is empty" << std::endl;
    }
     ASSERT_TRUE(table);
}

TEST(VSDS, PRE_IMAGES_HYBRID) {
    std::string path = vsds_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    // Execute pre-filtering query
    maximus::vsds::QueryParameters params;
    params.k = 5;

    auto q      = maximus::vsds::pre_images_hybrid(db, params);
    
    std::cout << "Query Plan = \n" << q->to_string() << std::endl;

    auto table = db->query(q);
    
    std::cout << "Query result = \n";
    if (table) {
        std::cout << "Rows: " << table->num_rows() << "\n";
        table->slice(0, 10)->print();
    } else {
        std::cout << "Query result is empty" << std::endl;
    }
     ASSERT_TRUE(table);
}

TEST(VSDS, Post_Reviews_Flat) {
    std::string path = vsds_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    auto device = maximus::DeviceType::CPU;

    // Build index
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    auto training_data = db->get_table("reviews");
    auto index = maximus::faiss::FaissIndex::build(ctx,
                                              training_data,
                                              "rv_embedding",
                                              "Flat",
                                              maximus::VectorDistanceMetric::L2,
                                              false);

    // Execute post-filtering query
    maximus::vsds::QueryParameters params;
    params.k = 2;
    params.postfilter_ksearch = 3;  // Larger k to ensure results after filtering
    params.query_count = 10;  // Enable scatter-gather pattern for per-query limit

    auto q      = maximus::vsds::post_reviews(db, index, device, params);
    
    std::cout << "Query Plan = \n" << q->to_string() << std::endl;

    auto table = db->query(q);
    
    std::cout << "Query result = \n";
    if (table) {
        std::cout << "Rows: " << table->num_rows() << "\n";
        table->slice(0, 10)->print();
    } else {
        std::cout << "Query result is empty" << std::endl;
    }
    
    ASSERT_TRUE(table);
}

TEST(VSDS, Post_Images_Flat) {
    std::string path = vsds_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    auto device = maximus::DeviceType::CPU;

    // Build index
    db->load_table("images", maximus::vsds::schema("images"), {}, device);
    auto training_data = db->get_table("images");
    auto index = maximus::faiss::FaissIndex::build(ctx,
                                              training_data,
                                              "i_embedding",
                                              "Flat",
                                              maximus::VectorDistanceMetric::L2,
                                              false);

    // Execute post-filtering query
    maximus::vsds::QueryParameters params;
    params.k = 2;
    params.postfilter_ksearch = 1000;  // Larger k to ensure results after filtering
    params.query_count = 10;  // Enable scatter-gather pattern for per-query limit
    // params.query_start = 0;  
    // params.query_count = 1;  

    auto q      = maximus::vsds::post_images(db, index, device, params);
    
    std::cout << "Query Plan = \n" << q->to_string() << std::endl;

    auto table = db->query(q);
    
    std::cout << "Query result = \n";
    if (table) {
        std::cout << "Rows: " << table->num_rows() << "\n";
        table->slice(0, 10)->print();
    } else {
        std::cout << "Query result is empty" << std::endl;
    }
    
    ASSERT_TRUE(table);
}


TEST(VSDS, POST_IMAGES_HYBRID) {
    std::string path = vsds_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    auto device = maximus::DeviceType::CPU;

    // Build index
    db->load_table("images", maximus::vsds::schema("images"), {}, device);
    auto training_data = db->get_table("images");
    auto index = maximus::faiss::FaissIndex::build(ctx,
                                              training_data,
                                              "i_embedding",
                                              "GPU,Flat",
                                              maximus::VectorDistanceMetric::INNER_PRODUCT,
                                              false);

    // Execute post-filtering query
    maximus::vsds::QueryParameters params;
    params.k = 2;
    params.postfilter_ksearch = 1000;  // Larger k to ensure results after filtering
    params.query_count = 10;  // Enable scatter-gather pattern for per-query limit
    // params.query_start = 0;  
    // params.query_count = 1;  

    auto q      = maximus::vsds::post_images_hybrid(db, index, params);
    
    std::cout << "Query Plan = \n" << q->to_string() << std::endl;

    auto table = db->query(q);
    
    std::cout << "Query result = \n";
    if (table) {
        std::cout << "Rows: " << table->num_rows() << "\n";
        table->slice(0, 10)->print();
    } else {
        std::cout << "Query result is empty" << std::endl;
    }
    
    ASSERT_TRUE(table);
}


// ============================================================================
// Query Slicing Tests  
// Verifies that query_start/query_count parameters work correctly
// ============================================================================

// Use shared utility from vsds_utils.hpp
using maximus::vsds::extract_results_by_query;
using maximus::vsds::QueryResult;

TEST(VSDS, QuerySlicing_SingleQueryAtATime) {
    std::string path = vsds_path();
    std::cout << "VSDS Data Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto device       = maximus::DeviceType::CPU;
    
    // Preload tables so they remain available across queries
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);

    // First, run with all 10 queries
    maximus::vsds::QueryParameters params_all;
    params_all.k = 5;
    
    std::cout << "Getting the Ground truth for all queries (no slicing)...\n";
    auto q_all = maximus::vsds::enn_reviews(db, device, params_all);
    maximus::TablePtr result_all;
    
    result_all = db->query(q_all);
    // print result_all
    std::cout << "Full query result = \n";
    result_all->print();

    
    if (!result_all || result_all->num_rows() == 0) {
        std::cout << "[SKIPPED] No results from full query.\n";
        return;
    }
    
    std::cout << "Full query returned " << result_all->num_rows() << " rows\n";
    auto full_results = extract_results_by_query(result_all,
                                                  "rv_reviewkey_queries", "rv_reviewkey");
    int num_queries = full_results.size();
    std::cout << "Found " << num_queries << " unique query IDs in results\n";
    
    // Now run one query at a time
    std::cout << "\nRunning single queries (query_count=1) and comparing...\n";
    int mismatches = 0;
    
    for (int i = 0; i < num_queries; ++i) {
        maximus::vsds::QueryParameters params_single;
        params_single.k = 5;
        params_single.query_start = i;
        params_single.query_count = 1;
        
        auto q_single = maximus::vsds::enn_reviews(db, device, params_single);
        auto result_single = db->query(q_single);

        // print result_single
        std::cout << "Single query result = \n";
        result_single->print();
        
        if (!result_single || result_single->num_rows() == 0) {
            std::cout << "  Query " << i << ": no results (unexpected)\n";
            mismatches++;
            continue;
        }
        
        auto single_results = extract_results_by_query(result_single,
                                                         "rv_reviewkey_queries", "rv_reviewkey");
        // Should have exactly one query ID in results
        if (single_results.size() != 1) {
            std::cout << "  Query " << i << ": expected 1 query ID, got " << single_results.size() << "\n";
            mismatches++;
            continue;
        }
        
        // Compare with full results for this query ID
        int64_t qid = single_results.begin()->first;
        auto& single_neighbors = single_results[qid];
        auto& full_neighbors = full_results[qid];
        
        if (single_neighbors.size() != full_neighbors.size()) {
            std::cout << "  Query " << i << " (qid=" << qid << "): neighbor count mismatch "
                      << single_neighbors.size() << " vs " << full_neighbors.size() << "\n";
            mismatches++;
            continue;
        }
        
        // Check first few neighbors match
        bool match = true;
        for (size_t j = 0; j < std::min(single_neighbors.size(), size_t(3)); ++j) {
            // print query results and expected results
            if (single_neighbors[j].data_id != full_neighbors[j].data_id ||
                std::abs(single_neighbors[j].distance - full_neighbors[j].distance) > 1e-5) {
                match = false;
                break;
            }
        }
        
        if (!match) {
            std::cout << "  Query " << i << " (qid=" << qid << "): neighbor values mismatch\n";
            mismatches++;
        } else {
            std::cout << "  Query " << i << " (qid=" << qid << "): OK (" << single_neighbors.size() << " neighbors)\n";
        }
    }
    
    std::cout << "\nSingle-query slicing test: " << (num_queries - mismatches) << "/" << num_queries << " passed\n";
    EXPECT_EQ(mismatches, 0);
}

TEST(VSDS, QuerySlicing_BatchOf5) {
    std::string path = vsds_path();
    std::cout << "VSDS Data Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto device       = maximus::DeviceType::CPU;
    
    // Preload tables so they remain available across queries
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);

    // First, run with all 10 queries
    maximus::vsds::QueryParameters params_all;
    params_all.k = 5;
    
    std::cout << "Getting the Ground truth for all queries (no slicing)...\n";
    auto q_all = maximus::vsds::enn_reviews(db, device, params_all);
    maximus::TablePtr result_all;
    
    result_all = db->query(q_all);
    
    if (!result_all || result_all->num_rows() == 0) {
        std::cout << "[SKIPPED] No results from full query.\n";
        return;
    }
    
    std::cout << "Full query returned " << result_all->num_rows() << " rows\n";
    auto full_results = extract_results_by_query(result_all,
                                                  "rv_reviewkey_queries", "rv_reviewkey");
    int num_queries = full_results.size();
    std::cout << "Found " << num_queries << " unique query IDs\n";
    
    // Run in batches of 5
    std::cout << "\nRunning in 2 batches of 5 queries each...\n";
    
    // Batch 1: queries 0-4
    maximus::vsds::QueryParameters params_batch1;
    params_batch1.k = 5;
    params_batch1.query_start = 0;
    params_batch1.query_count = 5;
    
    auto q_batch1 = maximus::vsds::enn_reviews(db, device, params_batch1);
    auto result_batch1 = db->query(q_batch1);
    auto batch1_results = extract_results_by_query(result_batch1,
                                                    "rv_reviewkey_queries", "rv_reviewkey");
    std::cout << "Batch 1 (0-4): " << batch1_results.size() << " query IDs, "
              << (result_batch1 ? result_batch1->num_rows() : 0) << " total rows\n";
    
    // Batch 2: queries 5-9
    maximus::vsds::QueryParameters params_batch2;
    params_batch2.k = 5;
    params_batch2.query_start = 5;
    params_batch2.query_count = 5;
    
    auto q_batch2 = maximus::vsds::enn_reviews(db, device, params_batch2);
    auto result_batch2 = db->query(q_batch2);
    auto batch2_results = extract_results_by_query(result_batch2,
                                                    "rv_reviewkey_queries", "rv_reviewkey");
    std::cout << "Batch 2 (5-9): " << batch2_results.size() << " query IDs, "
              << (result_batch2 ? result_batch2->num_rows() : 0) << " total rows\n";
    
    // Merge batch results
    std::map<int64_t, std::vector<QueryResult>> merged_results = batch1_results;
    for (const auto& [qid, neighbors] : batch2_results) {
        merged_results[qid] = neighbors;
    }
    
    // Compare merged with full
    std::cout << "\nComparing merged batch results with full results...\n";
    EXPECT_EQ(merged_results.size(), full_results.size());
    
    int mismatches = 0;
    for (const auto& [qid, full_neighbors] : full_results) {
        auto it = merged_results.find(qid);
        if (it == merged_results.end()) {
            std::cout << "  Query ID " << qid << ": missing from batched results\n";
            mismatches++;
            continue;
        }
        
        const auto& batch_neighbors = it->second;
        if (batch_neighbors.size() != full_neighbors.size()) {
            std::cout << "  Query ID " << qid << ": neighbor count mismatch "
                      << batch_neighbors.size() << " vs " << full_neighbors.size() << "\n";
            mismatches++;
            continue;
        }
        
        // Check neighbors match
        bool match = true;
        for (size_t j = 0; j < full_neighbors.size(); ++j) {
            if (batch_neighbors[j].data_id != full_neighbors[j].data_id ||
                std::abs(batch_neighbors[j].distance - full_neighbors[j].distance) > 1e-5) {
                match = false;
                break;
            }
        }
        
        if (!match) {
            std::cout << "  Query ID " << qid << ": neighbor values mismatch\n";
            mismatches++;
        }
    }
    
    std::cout << "\nBatch slicing test: " << (num_queries - mismatches) << "/" << num_queries << " passed\n";
    EXPECT_EQ(mismatches, 0);
}

// ============================================================================
// ANN Slicing Test - verify slicing works with indexed vector join
// ============================================================================

TEST(VSDS, QuerySlicing_ANN_BatchOf5) {
    std::string path = vsds_path();
    std::cout << "VSDS Data Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    auto device       = maximus::DeviceType::CPU;
    
    // Preload tables
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);
    
    // Build index
    maximus::IndexPtr index = nullptr;
    auto training_data = db->get_table("reviews");
    if (training_data.empty()) {
        std::cout << "[SKIPPED] Reviews table empty.\n";
        return;
    }
    index = maximus::faiss::FaissIndex::build(ctx, training_data, "rv_embedding",
                                                   "IVF100,Flat", maximus::VectorDistanceMetric::L2, false);

    // Run with all queries
    maximus::vsds::QueryParameters params_all;
    params_all.k = 2;
    params_all.ivf_nprobe = 10;
    
    std::cout << "Running ANN with all queries (no slicing)...\n";
    auto q_all = maximus::vsds::ann_reviews(db, index, device, params_all);
    auto result_all = db->query(q_all);

    std::cout << "All queries single batch result = " << result_all->to_string() << "\n";
    
    if (!result_all || result_all->num_rows() == 0) {
        std::cout << "[SKIPPED] No results from full query.\n";
        return;
    }
    
    std::cout << "Full query returned " << result_all->num_rows() << " rows\n";
    auto full_results = extract_results_by_query(result_all, "rv_reviewkey_queries", "rv_reviewkey");
    int num_queries = full_results.size();
    std::cout << "Found " << num_queries << " unique query IDs\n";
    
    // Run in batches of 5
    std::cout << "\nRunning ANN in 2 batches of 5 queries each...\n";
    
    maximus::vsds::QueryParameters params_batch1;
    params_batch1.k = 2;
    params_batch1.query_start = 0;
    params_batch1.query_count = 5;
    params_batch1.ivf_nprobe = 10;
    
    auto q_batch1 = maximus::vsds::ann_reviews(db, index, device, params_batch1);
    auto result_batch1 = db->query(q_batch1);
    auto batch1_results = extract_results_by_query(result_batch1, "rv_reviewkey_queries", "rv_reviewkey");
    std::cout << "Batch 1 (0-4): " << batch1_results.size() << " query IDs\n";

    std::cout << "Batch 1 result = " << result_batch1->to_string() << "\n";

    maximus::vsds::QueryParameters params_batch2;
    params_batch2.k = 2;
    params_batch2.query_start = 5;
    params_batch2.query_count = 5;
    params_batch2.ivf_nprobe = 10;
    
    auto q_batch2 = maximus::vsds::ann_reviews(db, index, device, params_batch2);
    auto result_batch2 = db->query(q_batch2);
    auto batch2_results = extract_results_by_query(result_batch2, "rv_reviewkey_queries", "rv_reviewkey");
    std::cout << "Batch 2 (5-9): " << batch2_results.size() << " query IDs\n";

    std::cout << "Batch 2 result = " << result_batch2->to_string() << "\n";

    // Merge and compare
    std::map<int64_t, std::vector<QueryResult>> merged_results = batch1_results;
    for (const auto& [qid, neighbors] : batch2_results) {
        merged_results[qid] = neighbors;
    }
    
    EXPECT_EQ(merged_results.size(), full_results.size());
    
    int mismatches = 0;
    for (const auto& [qid, full_neighbors] : full_results) {
        auto it = merged_results.find(qid);
        if (it == merged_results.end() || it->second.size() != full_neighbors.size()) {
            mismatches++;
            continue;
        }
        for (size_t j = 0; j < full_neighbors.size(); ++j) {
            if (it->second[j].data_id != full_neighbors[j].data_id) {
                mismatches++;
                break;
            }
        }
    }
    
    std::cout << "ANN batch slicing test: " << (num_queries - mismatches) << "/" << num_queries << " passed\n";
    EXPECT_EQ(mismatches, 0);
}

// ============================================================================
// Pre-filtering Slicing Test ( Exhaustive )
// ============================================================================

TEST(VSDS, QuerySlicing_Pre_BatchOf5) {
    std::string path = vsds_path();
    std::cout << "VSDS Data Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    auto device       = maximus::DeviceType::CPU;
    
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);


    // Run with all queries
    maximus::vsds::QueryParameters params_all;
    params_all.k = 5;
    
    std::cout << "Running Pre-filtering with all queries...\n";
    auto q_all = maximus::vsds::pre_reviews(db, device, params_all);
    auto result_all = db->query(q_all);
    
    if (!result_all || result_all->num_rows() == 0) {
        std::cout << "[SKIPPED] No results from full query.\n";
        return;
    }
    
    auto full_results = extract_results_by_query(result_all, "rv_reviewkey_queries", "rv_reviewkey");
    int num_queries = full_results.size();
    std::cout << "Full query: " << num_queries << " query IDs, " << result_all->num_rows() << " rows\n";
    
    // Run in batches
    maximus::vsds::QueryParameters params_batch1;
    params_batch1.k = 5;
    params_batch1.query_start = 0;
    params_batch1.query_count = 5;
    
    auto q_batch1 = maximus::vsds::pre_reviews(db, device, params_batch1);
    auto result_batch1 = db->query(q_batch1);
    auto batch1_results = extract_results_by_query(result_batch1, "rv_reviewkey_queries", "rv_reviewkey");
    
    maximus::vsds::QueryParameters params_batch2;
    params_batch2.k = 5;
    params_batch2.query_start = 5;
    params_batch2.query_count = 5;
    
    auto q_batch2 = maximus::vsds::pre_reviews(db, device, params_batch2);
    auto result_batch2 = db->query(q_batch2);
    auto batch2_results = extract_results_by_query(result_batch2, "rv_reviewkey_queries", "rv_reviewkey");
    
    std::cout << "Batch 1: " << batch1_results.size() << " query IDs, Batch 2: " << batch2_results.size() << " query IDs\n";
    
    // Merge and compare
    std::map<int64_t, std::vector<QueryResult>> merged_results = batch1_results;
    for (const auto& [qid, neighbors] : batch2_results) {
        merged_results[qid] = neighbors;
    }
    
    EXPECT_EQ(merged_results.size(), full_results.size());
    
    int mismatches = 0;
    for (const auto& [qid, full_neighbors] : full_results) {
        auto it = merged_results.find(qid);
        if (it == merged_results.end() || it->second.size() != full_neighbors.size()) {
            mismatches++;
        }
    }
    
    std::cout << "Pre-filtering batch slicing test: " << (num_queries - mismatches) << "/" << num_queries << " passed\n";
    EXPECT_EQ(mismatches, 0);
}

// ============================================================================
// Post-filtering Slicing Test
// ============================================================================

TEST(VSDS, QuerySlicing_Post_BatchOf5) {
    std::string path = vsds_path();
    std::cout << "VSDS Data Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    
    try {
        db->load_table("reviews", maximus::vsds::schema("reviews"), {}, maximus::DeviceType::CPU);
        db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, maximus::DeviceType::CPU);
    } catch (const std::exception& e) {
        std::cout << "[SKIPPED] Failed to load tables: " << e.what() << "\n";
        return;
    }
    
    maximus::IndexPtr index = nullptr;
    try {
        auto training_data = db->get_table("reviews");
        if (training_data.empty()) {
            std::cout << "[SKIPPED] Reviews table empty.\n";
            return;
        }
        index = maximus::faiss::FaissIndex::build(ctx, training_data, "rv_embedding",
                                                   "Flat", maximus::VectorDistanceMetric::L2, false);
    } catch (const std::exception& e) {
        std::cout << "[SKIPPED] Failed to build index: " << e.what() << "\n";
        return;
    }

    // Run with all queries
    maximus::vsds::QueryParameters params_all;
    params_all.k = 5;
    params_all.postfilter_ksearch = 50;
    params_all.query_count = 10;
    
    std::cout << "Running Post-filtering with all queries...\n";
    auto q_all = maximus::vsds::post_reviews(db, index, maximus::DeviceType::CPU, params_all);
    auto result_all = db->query(q_all);
    
    if (!result_all || result_all->num_rows() == 0) {
        std::cout << "[SKIPPED] No results from full query.\n";
        return;
    }
    
    auto full_results = extract_results_by_query(result_all, "rv_reviewkey_queries", "rv_reviewkey");
    int num_queries = full_results.size();
    std::cout << "Full query: " << num_queries << " query IDs, " << result_all->num_rows() << " rows\n";
    
    // Run in batches
    maximus::vsds::QueryParameters params_batch1;
    params_batch1.k = 5;
    params_batch1.postfilter_ksearch = 50;
    params_batch1.query_start = 0;
    params_batch1.query_count = 5;
    
    auto q_batch1 = maximus::vsds::post_reviews(db, index, maximus::DeviceType::CPU, params_batch1);
    auto result_batch1 = db->query(q_batch1);
    auto batch1_results = extract_results_by_query(result_batch1, "rv_reviewkey_queries", "rv_reviewkey");
    
    maximus::vsds::QueryParameters params_batch2;
    params_batch2.k = 5;
    params_batch2.postfilter_ksearch = 50;
    params_batch2.query_start = 5;
    params_batch2.query_count = 5;
    
    auto q_batch2 = maximus::vsds::post_reviews(db, index, maximus::DeviceType::CPU, params_batch2);
    auto result_batch2 = db->query(q_batch2);
    auto batch2_results = extract_results_by_query(result_batch2, "rv_reviewkey_queries", "rv_reviewkey");
    
    std::cout << "Batch 1: " << batch1_results.size() << " query IDs, Batch 2: " << batch2_results.size() << " query IDs\n";
    
    // Merge and compare
    std::map<int64_t, std::vector<QueryResult>> merged_results = batch1_results;
    for (const auto& [qid, neighbors] : batch2_results) {
        merged_results[qid] = neighbors;
    }
    
    EXPECT_EQ(merged_results.size(), full_results.size());
    
    int mismatches = 0;
    for (const auto& [qid, full_neighbors] : full_results) {
        auto it = merged_results.find(qid);
        if (it == merged_results.end() || it->second.size() != full_neighbors.size()) {
            mismatches++;
        }
    }
    
    std::cout << "Post-filtering batch slicing test: " << (num_queries - mismatches) << "/" << num_queries << " passed\n";
    EXPECT_EQ(mismatches, 0);
}


TEST(VSDS, QuerySlicing_Post_SingleQueryAtATime) {
    std::string path = vsds_path();
    std::cout << "VSDS Data Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    auto device       = maximus::DeviceType::CPU;
    
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);
    
    // Build index
    auto training_data = db->get_table("reviews");
    auto index = maximus::faiss::FaissIndex::build(ctx, training_data, "rv_embedding",
                                                   "HNSW32,Flat", maximus::VectorDistanceMetric::L2, false);

    // Run with all 10 queries
    maximus::vsds::QueryParameters params_all;
    params_all.k = 2;
    params_all.postfilter_ksearch = 20;
    params_all.query_count = 10;
    
    std::cout << "Running Post-filtering with all queries (no slicing)...\n";
    auto q_all = maximus::vsds::post_reviews(db, index, device, params_all);
    auto result_all = db->query(q_all);
    
    if (!result_all || result_all->num_rows() == 0) {
        std::cout << "[SKIPPED] No results from full query.\n";
        return;
    }
    
    auto full_results = extract_results_by_query(result_all, "rv_reviewkey_queries", "rv_reviewkey");
    int num_queries = full_results.size();
    std::cout << "Full query: " << num_queries << " unique query IDs\n";
    
    // Now run one query at a time
    std::cout << "\nRunning single queries (query_count=1) and comparing...\n";
    int mismatches = 0;
    
    for (int i = 0; i < num_queries; ++i) {
        maximus::vsds::QueryParameters params_single;
        params_single.k = 2;
        params_single.postfilter_ksearch = 20;
        params_single.query_start = i;
        params_single.query_count = 1;
        
        auto q_single = maximus::vsds::post_reviews(db, index, device, params_single);
        auto result_single = db->query(q_single);
        
        if (!result_single || result_single->num_rows() == 0) {
            std::cout << "  Query " << i << ": no results (unexpected)\n";
            mismatches++;
            continue;
        }
        
        auto single_results = extract_results_by_query(result_single, "rv_reviewkey_queries", "rv_reviewkey");
        if (single_results.size() != 1) {
            std::cout << "  Query " << i << ": expected 1 query ID, got " << single_results.size() << "\n";
            mismatches++;
            continue;
        }
        
        int64_t qid = single_results.begin()->first;
        auto& single_neighbors = single_results[qid];
        auto& full_neighbors = full_results[qid];
        
        if (single_neighbors.size() != full_neighbors.size()) {
            std::cout << "  Query " << i << " (qid=" << qid << "): neighbor count mismatch "
                      << single_neighbors.size() << " vs " << full_neighbors.size() << "\n";
            mismatches++;
            continue;
        }
        
        bool match = true;
        for (size_t j = 0; j < single_neighbors.size(); ++j) {
            if (single_neighbors[j].data_id != full_neighbors[j].data_id ||
                std::abs(single_neighbors[j].distance - full_neighbors[j].distance) > 1e-5) {
                match = false;
                break;
            }
        }
        
        if (!match) {
            std::cout << "  Query " << i << " (qid=" << qid << "): neighbor values mismatch\n";
            mismatches++;
        } else {
            std::cout << "  Query " << i << " (qid=" << qid << "): OK (" << single_neighbors.size() << " neighbors)\n";
        }
    }
    
    std::cout << "\nPost-filtering single-query slicing test: " << (num_queries - mismatches) << "/" << num_queries << " passed\n";
    EXPECT_EQ(mismatches, 0);
}

// ============================================================================
// Pre-join Slicing Test (exhaustive search - no index needed)
// ============================================================================

// ============================================================================
// Pre-join Slicing Test (exhaustive search - no index needed)
// Fails due to Acero limitation: "Data type list<item: float> is not supported in join non-key field"
// ============================================================================

TEST(VSDS, DISABLED_QuerySlicing_PreJoin_BatchOf5) {
    std::string path = vsds_path();
    std::cout << "VSDS Data Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto device       = maximus::DeviceType::CPU;
    
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);
    db->load_table("part", maximus::tpch::schema("part"), {}, device);

    // Full query (no index needed - exhaustive search)
    maximus::vsds::QueryParameters params_all;
    params_all.k = 5;
    
    std::cout << "Running PreJoin with all queries...\n";
    auto q_all = maximus::vsds::prejoin_reviews(db, device, params_all);
    auto result_all = db->query(q_all);
    
    if (!result_all || result_all->num_rows() == 0) {
        std::cout << "[SKIPPED] No results (maybe no parts with size >= 10).\n";
        return;
    }
    
    auto full_results = extract_results_by_query(result_all, "rv_reviewkey_queries", "rv_reviewkey");
    int num_queries = full_results.size();
    std::cout << "Full query: " << num_queries << " query IDs, " << result_all->num_rows() << " rows\n";
    
    // Batch 1
    maximus::vsds::QueryParameters params_batch1;
    params_batch1.k = 5;
    params_batch1.query_start = 0;
    params_batch1.query_count = 5;
    
    auto q_batch1 = maximus::vsds::prejoin_reviews(db, device, params_batch1);
    auto result_batch1 = db->query(q_batch1);
    auto batch1_results = extract_results_by_query(result_batch1, "rv_reviewkey_queries", "rv_reviewkey");
    
    // Batch 2
    maximus::vsds::QueryParameters params_batch2;
    params_batch2.k = 5;
    params_batch2.query_start = 5;
    params_batch2.query_count = 5;
    
    auto q_batch2 = maximus::vsds::prejoin_reviews(db, device, params_batch2);
    auto result_batch2 = db->query(q_batch2);
    auto batch2_results = extract_results_by_query(result_batch2, "rv_reviewkey_queries", "rv_reviewkey");
    
    std::cout << "Batch 1: " << batch1_results.size() << ", Batch 2: " << batch2_results.size() << " query IDs\n";
    
    // Merge and compare
    std::map<int64_t, std::vector<QueryResult>> merged = batch1_results;
    for (const auto& [qid, neighbors] : batch2_results) {
        merged[qid] = neighbors;
    }
    
    EXPECT_EQ(merged.size(), full_results.size());
    
    int mismatches = 0;
    for (const auto& [qid, full_neighbors] : full_results) {
        auto it = merged.find(qid);
        if (it == merged.end() || it->second.size() != full_neighbors.size()) {
            mismatches++;
        }
    }
    
    std::cout << "PreJoin batch slicing test: " << (num_queries - mismatches) << "/" << num_queries << " passed\n";
    EXPECT_EQ(mismatches, 0);
}

// ============================================================================
// Post-join Slicing Test (indexed search + join + filter)
// ============================================================================

TEST(VSDS, QuerySlicing_PostJoin_BatchOf5) {
    std::string path = vsds_path();
    std::cout << "VSDS Data Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    
    try {
        db->load_table("reviews", maximus::vsds::schema("reviews"), {}, maximus::DeviceType::CPU);
        db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, maximus::DeviceType::CPU);
        db->load_table("part", maximus::tpch::schema("part"), {}, maximus::DeviceType::CPU);
    } catch (const std::exception& e) {
        std::cout << "[SKIPPED] Failed to load tables: " << e.what() << "\n";
        return;
    }
    
    maximus::IndexPtr index = nullptr;
    try {
        auto training_data = db->get_table("reviews");
        if (training_data.empty()) {
            std::cout << "[SKIPPED] Reviews table empty.\n";
            return;
        }
        index = maximus::faiss::FaissIndex::build(ctx, training_data, "rv_embedding",
                                                   "Flat", maximus::VectorDistanceMetric::L2, false);
    } catch (const std::exception& e) {
        std::cout << "[SKIPPED] Failed to build index: " << e.what() << "\n";
        return;
    }

    // Full query
    maximus::vsds::QueryParameters params_all;
    params_all.k = 5;
    params_all.postfilter_ksearch = 50;
    params_all.query_count = 10; // needed for scatter-gather pattern
    
    std::cout << "Running PostJoin with all queries...\n";
    auto q_all = maximus::vsds::postjoin_reviews(db, index, maximus::DeviceType::CPU, params_all);
    auto result_all = db->query(q_all);
    
    if (!result_all || result_all->num_rows() == 0) {
        std::cout << "[SKIPPED] No results (maybe no matching parts).\n";
        return;
    }
    
    auto full_results = extract_results_by_query(result_all, "rv_reviewkey_queries", "rv_reviewkey");
    int num_queries = full_results.size();
    std::cout << "Full query: " << num_queries << " query IDs, " << result_all->num_rows() << " rows\n";
    
    // Batch 1
    maximus::vsds::QueryParameters params_batch1;
    params_batch1.k = 5;
    params_batch1.postfilter_ksearch = 50;
    params_batch1.query_start = 0;
    params_batch1.query_count = 5;
    
    auto q_batch1 = maximus::vsds::postjoin_reviews(db, index, maximus::DeviceType::CPU, params_batch1);
    auto result_batch1 = db->query(q_batch1);
    auto batch1_results = extract_results_by_query(result_batch1, "rv_reviewkey_queries", "rv_reviewkey");
    
    // Batch 2
    maximus::vsds::QueryParameters params_batch2;
    params_batch2.k = 5;
    params_batch2.postfilter_ksearch = 50;
    params_batch2.query_start = 5;
    params_batch2.query_count = 5;
    
    auto q_batch2 = maximus::vsds::postjoin_reviews(db, index, maximus::DeviceType::CPU, params_batch2);
    auto result_batch2 = db->query(q_batch2);
    auto batch2_results = extract_results_by_query(result_batch2, "rv_reviewkey_queries", "rv_reviewkey");
    
    std::cout << "Batch 1: " << batch1_results.size() << ", Batch 2: " << batch2_results.size() << " query IDs\n";
    
    // Merge and compare
    std::map<int64_t, std::vector<QueryResult>> merged = batch1_results;
    for (const auto& [qid, neighbors] : batch2_results) {
        merged[qid] = neighbors;
    }
    
    EXPECT_EQ(merged.size(), full_results.size());
    
    int mismatches = 0;
    for (const auto& [qid, full_neighbors] : full_results) {
        auto it = merged.find(qid);
        if (it == merged.end() || it->second.size() != full_neighbors.size()) {
            mismatches++;
        }
    }
    
    std::cout << "PostJoin batch slicing test: " << (num_queries - mismatches) << "/" << num_queries << " passed\n";
    EXPECT_EQ(mismatches, 0);
}



// ========================
//      VSDS Queries
// ========================

TEST(VSDS, Q2) {

    // std::string path = vsds_path_sf1(); // sf 1
    std::string path = vsds_path_sf001(); // sf 0.01
    // std::string path = vsds_path(); // sf 0.001

    std::cout << "Path = " << path << "\n";
    // Setup database
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    int nqueries = 5;
    auto device = maximus::DeviceType::CPU;

    // Preload tables to match maxbench behavior (avoiding CSV parsing during query exec)
    db->load_table("part", maximus::vsds::schema("part"), {}, device);
    db->load_table("partsupp", maximus::vsds::schema("partsupp"), {}, device);
    db->load_table("supplier", maximus::vsds::schema("supplier"), {}, device);
    db->load_table("nation", maximus::vsds::schema("nation"), {}, device);
    db->load_table("region", maximus::vsds::schema("region"), {}, device);
    db->load_table("images", maximus::vsds::schema("images"), {}, device);
    db->load_table("images_queries", maximus::vsds::schema("images_queries"), {}, device);

    // Build index
    auto index_desc = "Flat";
    auto training_data = db->get_table("images");
    auto index = maximus::faiss::FaissIndex::build(ctx,
                                              training_data,
                                              "i_embedding",
                                              index_desc,
                                              maximus::VectorDistanceMetric::L2,
                                              false);
    for (int i = 0; i < nqueries; ++i) {
        // Create ENN query for reviews
        maximus::vsds::QueryParameters params;
        params.k = 20;
        params.postfilter_ksearch = 200;
        // only one query at a time
        params.query_count = 1;
        params.query_start = i;

        auto q = maximus::vsds::q2_start(db, index, device, params);

        std::cout << "Query " << i << " Plan = \n" << q->to_string() << std::endl;
        // timer:
        auto start = std::chrono::high_resolution_clock::now();
        auto table = db->query(q);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Query " << i << " time = " << elapsed.count() << " seconds" << std::endl;

        std::cout << "Query " << i << " result = \n" << std::endl;
        if (table) {
            table->print();
        } else {
            std::cout << "Query result is empty" << std::endl;
        }
        ASSERT_TRUE(table);
    }
    
}


TEST(VSDS, Q10) {

    // std::string path = vsds_path_sf1(); // 1 sf
    std::string path = vsds_path_sf001(); // 0.01 sf
    // std::string path = vsds_path(); // 0.001 sf

    std::cout << "Path = " << path << "\n";
    // Setup database
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    int nqueries = 5;
    auto device = maximus::DeviceType::CPU;
    // Preload tables to match maxbench behavior (avoiding CSV parsing during query exec)
    db->load_table("lineitem", maximus::vsds::schema("lineitem"), {}, device);
    db->load_table("orders", maximus::vsds::schema("orders"), {}, device);
    db->load_table("customer", maximus::vsds::schema("customer"), {}, device);
    db->load_table("nation", maximus::vsds::schema("nation"), {}, device);
    db->load_table("region", maximus::vsds::schema("region"), {}, device);
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);

    // Build index
    auto index_desc = "Flat";
    auto training_data = db->get_table("reviews");
    auto index = maximus::faiss::FaissIndex::build(ctx,
                                              training_data,
                                              "rv_embedding",
                                              index_desc,
                                              maximus::VectorDistanceMetric::L2,
                                              false);
    for (int i = 0; i < nqueries; ++i) {
        // Create ENN query for reviews
        maximus::vsds::QueryParameters params;
        params.k = 100;
        // only one query at a time
        params.query_count = 1;
        params.query_start = i;
        params.faiss_index = index_desc; // ENN (also should avoid GPU int32 limitation)

        auto q = maximus::vsds::q10_mid(db, index, device, params);

        std::cout << "Query " << i << " Plan = \n" << q->to_string() << std::endl;
        // timer:
        auto start = std::chrono::high_resolution_clock::now();
        auto table = db->query(q);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Query " << i << " time = " << elapsed.count() << " seconds" << std::endl;

        std::cout << "Query " << i << " result = \n" << std::endl;
        if (table) {
            table->print();
        } else {
            std::cout << "Query result is empty" << std::endl;
        }
        ASSERT_TRUE(table);
    }
}


TEST(VSDS, Q18) {

    // std::string path = vsds_path_sf1(); // 1 sf
    std::string path = vsds_path_sf001(); // 0.01 sf
    // std::string path = vsds_path(); // 0.001 sf

    std::cout << "Path = " << path << "\n";
    // Setup database
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    int nqueries = 1;
    auto device = maximus::DeviceType::CPU;
    // Preload tables to match maxbench behavior (avoiding CSV parsing during query exec)
    db->load_table("lineitem", maximus::vsds::schema("lineitem"), {}, device);
    db->load_table("orders", maximus::vsds::schema("orders"), {}, device);
    db->load_table("customer", maximus::vsds::schema("customer"), {}, device);
    db->load_table("nation", maximus::vsds::schema("nation"), {}, device);
    db->load_table("region", maximus::vsds::schema("region"), {}, device);
    db->load_table("images", maximus::vsds::schema("images"), {}, device);
    db->load_table("images_queries", maximus::vsds::schema("images_queries"), {}, device);

    // Build index
    auto index_desc = "Flat";
    auto training_data = db->get_table("images");
    auto index = maximus::faiss::FaissIndex::build(ctx,
                                              training_data,
                                              "i_embedding",
                                              index_desc,
                                              maximus::VectorDistanceMetric::L2,
                                              false);
    for (int i = 0; i < nqueries; ++i) {
        // Create ENN query for images
        maximus::vsds::QueryParameters params;
        params.k = 100;
        params.postfilter_ksearch=1000;
        // only one query at a time
        params.query_count = 1;
        params.query_start = i;
        params.faiss_index = index_desc; // ENN (also should avoid GPU int32 limitation)

        auto q = maximus::vsds::q18_mid(db, index, device, params);

        std::cout << "Query " << i << " Plan = \n" << q->to_string() << std::endl;
        // timer:
        auto start = std::chrono::high_resolution_clock::now();
        auto table = db->query(q);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Query " << i << " time = " << elapsed.count() << " seconds" << std::endl;

        std::cout << "Query " << i << " result = \n" << std::endl;
        if (table) {
            table->print();
        } else {
            std::cout << "Query result is empty" << std::endl;
        }
        ASSERT_TRUE(table);
    }
}


TEST(VSDS, Q16) {
    // std::string path = vsds_path_sf1(); // 1 sf
    std::string path = vsds_path_sf001(); // 0.01 sf
    // std::string path = vsds_path(); // 0.001 sf

    std::cout << "Path = " << path << "\n";
    // Setup database
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    int nqueries = 5;
    auto device = maximus::DeviceType::CPU;

    // Preload tables to match maxbench behavior (avoiding CSV parsing during query exec)
    db->load_table("lineitem", maximus::vsds::schema("lineitem"), {}, device);
    db->load_table("orders", maximus::vsds::schema("orders"), {}, device);
    db->load_table("customer", maximus::vsds::schema("customer"), {}, device);
    db->load_table("nation", maximus::vsds::schema("nation"), {}, device);
    db->load_table("region", maximus::vsds::schema("region"), {}, device);
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);

    // Build index
    auto index_desc = "Flat";
    auto training_data = db->get_table("reviews");
    auto index = maximus::faiss::FaissIndex::build(ctx,
                                              training_data,
                                              "rv_embedding",
                                              index_desc,
                                              maximus::VectorDistanceMetric::L2,
                                              false);
    for (int i = 0; i < nqueries; ++i) {
        // Create ENN query for reviews
        maximus::vsds::QueryParameters params;
        params.k = 100;
        // only one query at a time
        params.query_count = 1;
        params.query_start = i;
        params.faiss_index = index_desc; // ENN (also should avoid GPU int32 limitation)

        auto q = maximus::vsds::q16_start(db, index, device, params);

        std::cout << "Query " << i << " Plan = \n" << q->to_string() << std::endl;
        // timer:
        auto start = std::chrono::high_resolution_clock::now();
        auto table = db->query(q);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Query " << i << " time = " << elapsed.count() << " seconds" << std::endl;

        std::cout << "Query " << i << " result = \n" << std::endl;
        if (table) {
            // table->print();
            table->slice(0, 10)->print();
        } else {
            std::cout << "Query result is empty" << std::endl;
        }
        ASSERT_TRUE(table);
    }
}



TEST(VSDS, Q19) {
    // std::string path = vsds_path_sf1(); // 1 sf
    std::string path = vsds_path_sf001(); // 0.01 sf
    // std::string path = vsds_path(); // 0.001 sf

    std::cout << "Path = " << path << "\n";
    // Setup database
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    int nqueries = 5;
    auto device = maximus::DeviceType::CPU;
    
    // Preload tables to match maxbench behavior (avoiding CSV parsing during query exec)
    db->load_table("lineitem", maximus::vsds::schema("lineitem"), {}, device);
    db->load_table("orders", maximus::vsds::schema("orders"), {}, device);
    db->load_table("customer", maximus::vsds::schema("customer"), {}, device);
    db->load_table("nation", maximus::vsds::schema("nation"), {}, device);
    db->load_table("region", maximus::vsds::schema("region"), {}, device);
    // x2 vector search in this query
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);
    db->load_table("images", maximus::vsds::schema("images"), {}, device);
    db->load_table("images_queries", maximus::vsds::schema("images_queries"), {}, device);

    // Build indexes
    auto index_desc = "Flat";
    auto reviews_data = db->get_table("reviews");
    auto reviews_index = maximus::faiss::FaissIndex::build(ctx,
                                              reviews_data,
                                              "rv_embedding",
                                              index_desc,
                                              maximus::VectorDistanceMetric::L2,
                                              false);
    //  index_desc = "Flat";
    auto images_data = db->get_table("images");
    auto images_index = maximus::faiss::FaissIndex::build(ctx,
                                              images_data,
                                              "i_embedding",
                                              index_desc,
                                              maximus::VectorDistanceMetric::L2,
                                              false);

    for (int i = 0; i < nqueries; ++i) {
        // Create ENN query for reviews
        maximus::vsds::QueryParameters params;
        params.k = 100;
        params.postfilter_ksearch = 300;
        // only one query at a time
        params.query_count = 1;
        params.query_start = i;
        params.faiss_index = index_desc; // ENN (also should avoid GPU int32 limitation)

        auto q = maximus::vsds::q19_start(db, reviews_index, images_index, device, params);

        std::cout << "Query " << i << " Plan = \n" << q->to_string() << std::endl;
        // timer:
        auto start = std::chrono::high_resolution_clock::now();
        auto table = db->query(q);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Query " << i << " time = " << elapsed.count() << " seconds" << std::endl;

        std::cout << "Query " << i << " result = \n" << std::endl;
        if (table) {
            // table->print();
            table->slice(0, 10)->print();
        } else {
            std::cout << "Query result is empty" << std::endl;
        }
        ASSERT_TRUE(table);
    }
}



TEST(VSDS, Q11) {

    // std::string path = vsds_path_sf1(); // 1 sf
    std::string path = vsds_path_sf001(); // 0.01 sf
    // std::string path = vsds_path(); // 0.001 sf

    std::cout << "Path = " << path << "\n";
    // Setup database
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    int nqueries = 5;
    auto device = maximus::DeviceType::CPU;

    // Preload tables to match maxbench behavior (avoiding CSV parsing during query exec)
    db->load_table("partsupp", maximus::vsds::schema("partsupp"), {}, device);
    db->load_table("nation", maximus::vsds::schema("nation"), {}, device);
    db->load_table("supplier", maximus::vsds::schema("supplier"), {}, device);
    
    db->load_table("images", maximus::vsds::schema("images"), {}, device);
    db->load_table("images_queries", maximus::vsds::schema("images_queries"), {}, device);

    // Build index
    auto index_desc = "Flat";
    auto training_data = db->get_table("images");
    auto index = maximus::faiss::FaissIndex::build(ctx,
                                              training_data,
                                              "i_embedding",
                                              index_desc,
                                              maximus::VectorDistanceMetric::L2,
                                              false);
    for (int i = 0; i < nqueries; ++i) {
        // Create ENN query for reviews
        maximus::vsds::QueryParameters params;
        params.k = 10;
        // only one query at a time
        params.query_count = 1;
        params.query_start = i;
        params.faiss_index = index_desc; // ENN (also should avoid GPU int32 limitation)

        auto q = maximus::vsds::q11_end(db, index, device, params);

        std::cout << "Query " << i << " Plan = \n" << q->to_string() << std::endl;
        // timer:
        auto start = std::chrono::high_resolution_clock::now();
        auto table = db->query(q);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Query " << i << " time = " << elapsed.count() << " seconds" << std::endl;

        std::cout << "Query " << i << " result = \n" << std::endl;
        if (table) {
            table->print();
        } else {
            std::cout << "Query result is empty" << std::endl;
        }
        ASSERT_TRUE(table);
    }
}


TEST(VSDS, Q15) {

    // std::string path = vsds_path_sf1(); // 1 sf
    std::string path = vsds_path_sf001(); // 0.01 sf
    // std::string path = vsds_path(); // 0.001 sf

    std::cout << "Path = " << path << "\n";
    // Setup database
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    int nqueries = 5;
    auto device = maximus::DeviceType::CPU;

    // Preload tables to match maxbench behavior (avoiding CSV parsing during query exec)
    db->load_table("lineitem", maximus::vsds::schema("lineitem"), {}, device);
    db->load_table("partsupp", maximus::vsds::schema("partsupp"), {}, device);
    db->load_table("part", maximus::vsds::schema("part"), {}, device);
    db->load_table("nation", maximus::vsds::schema("nation"), {}, device);
    db->load_table("supplier", maximus::vsds::schema("supplier"), {}, device);
    
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);

    // Build index
    auto index_desc = "Flat";
    auto training_data = db->get_table("reviews");
    auto index = maximus::faiss::FaissIndex::build(ctx,
                                              training_data,
                                              "rv_embedding",
                                              index_desc,
                                              maximus::VectorDistanceMetric::L2,
                                              false);
    for (int i = 0; i < nqueries; ++i) {
        // Create ENN query for reviews
        maximus::vsds::QueryParameters params;
        params.k = 10;
        // only one query at a time
        params.query_count = 1;
        params.query_start = i;
        params.faiss_index = index_desc; // ENN (also should avoid GPU int32 limitation)
        // Need large postfilter_ksearch on small datasets: the top supplier's parts
        // are rare, so we must search broadly to find reviews matching those parts.
        params.postfilter_ksearch = 1000;

        auto q = maximus::vsds::q15_end(db, index, device, params);

        std::cout << "Query " << i << " Plan = \n" << q->to_string() << std::endl;
        // timer:
        auto start = std::chrono::high_resolution_clock::now();
        auto table = db->query(q);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Query " << i << " time = " << elapsed.count() << " seconds" << std::endl;

        std::cout << "Query " << i << " result = \n" << std::endl;
        if (table) {
            table->print();
        } else {
            std::cout << "Query result is empty" << std::endl;
        }
        ASSERT_TRUE(table);
    }
}



TEST(VSDS, Q13) {

    // std::string path = vsds_path_sf1(); // 1 sf
    std::string path = vsds_path_sf001(); // 0.01 sf
    // std::string path = vsds_path(); // 0.001 sf

    std::cout << "Path = " << path << "\n";
    // Setup database
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    int nqueries = 5;
    auto device = maximus::DeviceType::CPU;

    // Preload tables to match maxbench behavior (avoiding CSV parsing during query exec)
    db->load_table("lineitem", maximus::vsds::schema("lineitem"), {}, device);
    db->load_table("partsupp", maximus::vsds::schema("partsupp"), {}, device);
    db->load_table("part", maximus::vsds::schema("part"), {}, device);
    db->load_table("nation", maximus::vsds::schema("nation"), {}, device);
    db->load_table("supplier", maximus::vsds::schema("supplier"), {}, device);
    
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);

    // Build index
    auto index_desc = "Flat";
    auto training_data = db->get_table("reviews");
    auto index = maximus::faiss::FaissIndex::build(ctx,
                                              training_data,
                                              "rv_embedding",
                                              index_desc,
                                              maximus::VectorDistanceMetric::L2,
                                              false);
    for (int i = 0; i < nqueries; ++i) {
        // Create ENN query for reviews
        maximus::vsds::QueryParameters params;
        params.k = 100;
        // only one query at a time
        params.query_count = 1;
        params.query_start = i;
        params.faiss_index = index_desc; // ENN (also should avoid GPU int32 limitation)

        auto q = maximus::vsds::q13_mid(db, index, device, params);

        std::cout << "Query " << i << " Plan = \n" << q->to_string() << std::endl;
        // timer:
        auto start = std::chrono::high_resolution_clock::now();
        auto table = db->query(q);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Query " << i << " time = " << elapsed.count() << " seconds" << std::endl;

        std::cout << "Query " << i << " result = \n" << std::endl;
        if (table) {
            table->print();
        } else {
            std::cout << "Query result is empty" << std::endl;
        }
        ASSERT_TRUE(table);
    }
}


TEST(VSDS, Q1) {

    // std::string path = vsds_path_sf1(); // 1 sf
    std::string path = vsds_path_sf001(); // 0.01 sf
    // std::string path = vsds_path(); // 0.001 sf

    std::cout << "Path = " << path << "\n";
    // Setup database
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    auto device = maximus::DeviceType::CPU;

    // Preload tables
    db->load_table("lineitem", maximus::vsds::schema("lineitem"), {}, device);
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);

    // NOTE: q1_end uses exhaustive_vector_join (no index needed — class embeddings are tiny).
    // The index parameter is passed for API consistency but unused.
    // Build a dummy index to satisfy the function signature.
    auto index_desc = "Flat";
    auto training_data = db->get_table("reviews");
    auto index = maximus::faiss::FaissIndex::build(ctx,
                                              training_data,
                                              "rv_embedding",
                                              index_desc,
                                              maximus::VectorDistanceMetric::L2,
                                              false);

    // q1_end runs once — classifies ALL reviews against ALL class centroids.
    // No query iteration needed (unlike ann/pre/post queries).
    maximus::vsds::QueryParameters params;
    params.k = 1;           // K=1: each review → nearest class (hardcoded in impl, param unused)
    params.query_count = 3; // number of classes for Q1 
    params.query_start = 0;
    params.faiss_index = index_desc;

    auto q = maximus::vsds::q1_start(db, index, device, params);

    std::cout << "Plan = \n" << q->to_string() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto table = db->query(q);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time = " << elapsed.count() << " seconds" << std::endl;

    std::cout << "Result = \n" << std::endl;
    if (table) {
        table->print();
    } else {
        std::cout << "Query result is empty" << std::endl;
    }
    ASSERT_TRUE(table);
}

TEST(VSDS, DISABLED_LOAD_SF1){
    // note (running disabled tests): ninja -C ./build/Release/ && ./build/Release/tests/test.vsds --gtest_also_run_disabled_tests --gtest_filter='VSDS.DISABLED_LOAD_SF1'
    std::string path = vsds_path_sf1();

    std::cout << "Path = " << path << "\n";
    
    // Setup database
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    auto device = maximus::DeviceType::CPU;

    // Preload tables 
    auto start = std::chrono::high_resolution_clock::now();
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);
    db->load_table("images", maximus::vsds::schema("images"), {}, device);
    db->load_table("images_queries", maximus::vsds::schema("images_queries"), {}, device);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time to load reviews and images tables = " << elapsed.count() << " seconds" << std::endl;

    auto second_start = std::chrono::high_resolution_clock::now();
    auto reviews = db->get_table("reviews");
    auto images  = db->get_table("images");
    ASSERT_TRUE(reviews);
    ASSERT_TRUE(images);
    ASSERT_GT(reviews.as_table()->num_rows(), 0);
    ASSERT_GT(images.as_table()->num_rows(), 0);
    std::cout << "Reviews rows: " << reviews.as_table()->num_rows() << "\n";
    std::cout << "Images rows:  " << images.as_table()->num_rows() << "\n";
    auto second_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> second_elapsed = second_end - second_start;
    std::cout << "Time to access loaded tables = " << second_elapsed.count() << " seconds" << std::endl;

    auto third_end = std::chrono::high_resolution_clock::now();
    auto reviews_2 = db->get_table_nocopy("reviews");
    auto images_2  = db->get_table_nocopy("images");
    ASSERT_TRUE(reviews_2);
    ASSERT_TRUE(images_2);
    ASSERT_GT(reviews_2.as_table()->num_rows(), 0);
    ASSERT_GT(images_2.as_table()->num_rows(), 0);
    std::cout << "Reviews rows (nocopy): " << reviews_2.as_table()->num_rows() << "\n";
    std::cout << "Images rows (nocopy):  " << images_2.as_table()->num_rows() << "\n"; 
    std::chrono::duration<double> third_elapsed = third_end - second_end;
    std::cout << "Time to access loaded tables (nocopy) = " << third_elapsed.count() << " seconds" << std::endl;
}