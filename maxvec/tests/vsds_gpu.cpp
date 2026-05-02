#include <gtest/gtest.h>

#include <iostream>
#include <maximus/database.hpp>
#include <maximus/frontend/query_plan_api.hpp>
#include <maximus/indexes/faiss/faiss_index.hpp>
#include <maximus/vsds/vsds_queries.hpp>
#include <maximus/utils/utils.hpp>
#include <chrono>
#include "utils.hpp"

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
// ENN Tests (GPU)
// ============================================================================

// #BUG: Assertion failure in gpu/cudf/parquet.cpp: ctx cannot be nullptr
TEST(VSDS, ENN_Reviews) {
    std::string path = vsds_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    maximus::vsds::QueryParameters params;
    params.k = 5;
    
    auto device = maximus::DeviceType::GPU;
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
    
    auto device = maximus::DeviceType::GPU;
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
// ANN Tests (GPU)
// ============================================================================

TEST(VSDS, ANN_Flat) {
    std::string path = vsds_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    // 1. Build GPU Index (Offline phase)
    std::cout << "Building GPU Index (GPU,Flat)...\n";
    
    auto device = maximus::DeviceType::GPU;
    
    // Load table on CPU first (GPU operators handle conversion internally)
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, maximus::DeviceType::CPU);
    auto training_data = db->get_table("reviews");

    // GPU index uses "GPU," prefix
    auto index = maximus::faiss::FaissIndex::build(ctx,
                                                training_data,
                                                "rv_embedding",
                                                "GPU,Flat",
                                                maximus::VectorDistanceMetric::L2,
                                                false);
    // 2. Execute ANN Query on GPU
    
    //assert device type gpu
    ASSERT_EQ(index->device_type, device);

    maximus::vsds::QueryParameters params;
    params.k = 5;
    params.faiss_index = "GPU,Flat";

    auto q = maximus::vsds::ann_reviews(db, index, device, params);
    
    std::cout << "Query Plan = \n" << q->to_string() << std::endl;

    auto table = db->query(q);
    
    std::cout << "Query ANN " << params.faiss_index << " = \n";
    if (table) {
        std::cout << "Rows: " << table->num_rows() << "\n";
        table->slice(0, 10)->print();
    } else {
        std::cout << "Query result is empty" << std::endl;
    }
    ASSERT_TRUE(table);
}


TEST(VSDS, DISABLED_ANN_Reviews_CAGRA) {
    // std::string path = vsds_path();
    std::string path = vsds_path_sf1();

    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    
    auto device = maximus::DeviceType::GPU;
    // auto index_desc = "GPU,HNSW32,Flat";
    auto index_desc = "GPU,Cagra,64,32,NN_DESCENT";

    // 1. Build Index (Offline phase)
    std::cout << "Building Index (" << index_desc << ")...\n";
    
    // Load on CPU for index build (avoid cuDF LargeList int32 offset limitation on SF1)
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, maximus::DeviceType::CPU);
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

    // Drop large embedding column and move the remaining reviews table to GPU for query execution
    move_tables_to_gpu(db, {"reviews"}, ctx, {{"reviews", "rv_embedding"}});

    // 2. Execute ANN Query
    maximus::vsds::QueryParameters params;
    params.k = 100;
    params.faiss_index = index_desc;
    params.hnsw_efsearch = 64;
    params.cagra_itopksize = 1024;


    // print index device
    std::cout << "Index device type: " << (index->is_on_gpu() ? "GPU" : "CPU") << "\n";

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


TEST(VSDS, DISABLED_ANN_Reviews_CAGRA_MaxbenchPath) {
    // std::string path = vsds_path();
    std::string path = vsds_path_sf1(); // sf 1
    std::cout << "Path = " << path << "\n";

    print_gpu_mem("before make_context");

    auto context     = maximus::make_context();
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue, context);

    const auto device               = maximus::DeviceType::GPU;
    const auto storage_device       = maximus::DeviceType::GPU;
    const auto index_storage_device = maximus::DeviceType::GPU;
    const bool using_large_list     = true;
    const std::string index_desc    = "GPU,Cagra,64,32,NN_DESCENT";
    const std::vector<std::string> queries = {"ann_reviews"};
    const std::string params =
        "k=100,ivf_nprobe=11,query_count=1,query_start=0,use_cuvs=1,metric=IP,use_post=0,"
        "hnsw_efsearch=384,cagra_itopksize=1024,use_limit_per_group=1";

    auto parsed_params = maximus::vsds::parse_query_parameters(params);

    auto tables = get_table_names("vsds");
    auto schemas = get_table_schemas("vsds");
    std::vector<std::string> force_cpu_tables;
    if (using_large_list && storage_device == maximus::DeviceType::GPU) {
        force_cpu_tables = {"reviews"};
    }
    print_gpu_mem("before load_tables");

    load_tables(db, tables, schemas, storage_device, force_cpu_tables);

    print_gpu_mem("after load_tables");

    auto indexes = build_indexes("vsds",
                                 index_desc,
                                 queries,
                                 db,
                                 context,
                                 storage_device,
                                 index_storage_device,
                                 true,
                                 false,
                                 "",
                                 parsed_params.metric,
                                 parsed_params.use_cuvs);

    print_gpu_mem("after build_indexes");

    if (using_large_list && storage_device == maximus::DeviceType::GPU) {
        move_tables_to_gpu(db, {"reviews"}, context, {{"reviews", "rv_embedding"}});
    }

    print_gpu_mem("after move_tables_to_gpu");

    ASSERT_TRUE(indexes.contains("reviews.rv_embedding"));
    ASSERT_TRUE(indexes["reviews.rv_embedding"]->is_on_gpu());

    auto reviews = db->get_table("reviews");
    ASSERT_TRUE(reviews.on_gpu()) << "reviews table should be on GPU before query execution";

    auto query_plan = maximus::vsds::query_plan("ann_reviews", db, device, params, indexes, index_desc);
    ASSERT_TRUE(query_plan);

    std::cout << "Query Plan = \n" << query_plan->to_string() << std::endl;

    print_gpu_mem("before db->query (CAGRA search)");

    auto table = db->query(query_plan);

    print_gpu_mem("after db->query");

    ASSERT_TRUE(table);
    std::cout << "Query result rows: " << table->num_rows() << "\n";
    table->slice(0, 10)->print();
}

// Test: approach 1 — keep buffer alive via get_table_nocopy().
// If the ATS crash is caused by the clone from get_table() being freed,
// then using get_table_nocopy() (which shares buffers with the database)
// should prevent the crash even without the raft/cuVS ATS patch.
TEST(VSDS, DISABLED_ANN_Reviews_CAGRA_NoCopy) {
    std::string path = vsds_path_sf1();
    std::cout << "Path = " << path << "\n";

    auto context     = maximus::make_context();
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue, context);

    const auto device               = maximus::DeviceType::GPU;
    const auto storage_device       = maximus::DeviceType::CPU;
    const auto index_storage_device = maximus::DeviceType::GPU;
    const std::string index_desc    = "GPU,Cagra,64,32,NN_DESCENT";
    const std::string params_str =
        "k=100,query_count=1,query_start=0,use_cuvs=1,metric=IP,cagra_itopksize=1024";
    auto parsed_params = maximus::vsds::parse_query_parameters(params_str);

    // Load tables on CPU
    auto tables = get_table_names("vsds");
    auto schemas = get_table_schemas("vsds");
    load_tables(db, tables, schemas, storage_device, {});

    // Build index using get_table_nocopy — buffer shares with database, stays alive
    auto data = db->get_table_nocopy("reviews");
    ASSERT_FALSE(data.empty());

    auto idx = maximus::faiss::FaissIndex::build(
        context, data, "rv_embedding", index_desc, parsed_params.metric,
        false, "", parsed_params.use_cuvs);
    context->barrier();
    ASSERT_TRUE(idx->is_on_gpu());

    print_gpu_mem("after build");

    // Build query plan and execute — CAGRA search uses the non-owning view
    auto indexes = maximus::IndexMap{{"reviews.rv_embedding", idx}};
    auto query_plan = maximus::vsds::query_plan(
        "ann_reviews", db, device, params_str, indexes, index_desc);
    ASSERT_TRUE(query_plan);

    std::cout << "Query Plan = \n" << query_plan->to_string() << std::endl;

    auto result = db->query(query_plan);
    ASSERT_TRUE(result);
    std::cout << "Query result rows: " << result->num_rows() << "\n";
    result->slice(0, 10)->print();
}


TEST(VSDS, ANN_Images_IVF) {
    std::string path = vsds_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    auto device = maximus::DeviceType::GPU;
    auto index_desc = "GPU,IVF64,Flat";

    // 1. Build Index (Offline phase)
    std::cout << "Building Index (" << index_desc << ")...\n";
    
    // Load table explicitly for index building
    db->load_table("images", maximus::vsds::schema("images"), {}, device);
    auto training_data = db->get_table("images");

    std::cout << "Loaded training data..." << std::endl;

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
    auto device = maximus::DeviceType::GPU;

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
    auto device = maximus::DeviceType::GPU;

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


TEST(VSDS, Post_Reviews_Flat) {
    std::string path = vsds_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    auto device = maximus::DeviceType::GPU;

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
    params.query_count = 10;  // SOS: must be = number of queries for correct results
    params.faiss_index = "Flat";

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

// BUG: Different results than CPU 
// - TODO: Verify correct ones in Postgres...
TEST(VSDS, Post_Images_Flat) {
    std::string path = vsds_path();
    std::cout << "Path = " << path << "\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    auto device = maximus::DeviceType::GPU;

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
    params.query_count = 10;  // SOS: must be = number of queries for correct results
    // params.faiss_index = "Flat";

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


// ============================================================================
// Index Movement Tests
// ============================================================================

// Test: GPU Index -> CPU (to_cpu)
// #BUG: Assertion failure in gpu/cudf/parquet.cpp: ctx cannot be nullptr
TEST(VSDS, DISABLED_IndexMovement_GPU_to_CPU) {
    std::string path = vsds_path();
    std::cout << "=== Test: GPU Index -> CPU ===\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    // Build GPU index
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, maximus::DeviceType::CPU);
    auto training_data = db->get_table("reviews");
    
    auto gpu_index = maximus::faiss::FaissIndex::build(ctx,
                                                        training_data,
                                                        "rv_embedding",
                                                        "GPU,Flat",
                                                        maximus::VectorDistanceMetric::L2,
                                                        false);
    
    // Verify it's on GPU
    ASSERT_TRUE(gpu_index->is_on_gpu());
    ASSERT_FALSE(gpu_index->is_on_cpu());
    std::cout << "GPU index built, device_type = GPU: " << gpu_index->is_on_gpu() << "\n";
    
    // Move to CPU
    auto cpu_index = gpu_index->to_cpu();
    
    // Verify it's now on CPU
    ASSERT_TRUE(cpu_index->is_on_cpu());
    ASSERT_FALSE(cpu_index->is_on_gpu());
    std::cout << "Index moved to CPU, device_type = CPU: " << cpu_index->is_on_cpu() << "\n";
    
    // Original should still be GPU
    ASSERT_TRUE(gpu_index->is_on_gpu());
    std::cout << "Original still on GPU: " << gpu_index->is_on_gpu() << "\n";
}

// Test: CPU Index -> GPU (to_gpu)
// #BUG: Assertion failure in gpu/cudf/parquet.cpp: ctx cannot be nullptr
TEST(VSDS, DISABLED_IndexMovement_CPU_to_GPU) {
    std::string path = vsds_path();
    std::cout << "=== Test: CPU Index -> GPU ===\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    // Build CPU index
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, maximus::DeviceType::CPU);
    auto training_data = db->get_table("reviews");
    
    auto cpu_index = maximus::faiss::FaissIndex::build(ctx,
                                                        training_data,
                                                        "rv_embedding",
                                                        "Flat",  // No GPU, prefix = CPU
                                                        maximus::VectorDistanceMetric::L2,
                                                        false);
    
    // Verify it's on CPU
    ASSERT_TRUE(cpu_index->is_on_cpu());
    ASSERT_FALSE(cpu_index->is_on_gpu());
    std::cout << "CPU index built, device_type = CPU: " << cpu_index->is_on_cpu() << "\n";
    
    // Move to GPU
    auto gpu_index = cpu_index->to_gpu();
    
    // Verify it's now on GPU
    ASSERT_TRUE(gpu_index->is_on_gpu());
    ASSERT_FALSE(gpu_index->is_on_cpu());
    std::cout << "Index moved to GPU, device_type = GPU: " << gpu_index->is_on_gpu() << "\n";
    
    // Original should still be CPU
    ASSERT_TRUE(cpu_index->is_on_cpu());
    std::cout << "Original still on CPU: " << cpu_index->is_on_cpu() << "\n";
}

// Test: GPU Index stays on GPU when to_gpu called (no-op)
// #BUG: Assertion failure in gpu/cudf/parquet.cpp: ctx cannot be nullptr
TEST(VSDS, DISABLED_IndexMovement_GPU_to_GPU_NoOp) {
    std::string path = vsds_path();
    std::cout << "=== Test: GPU Index -> GPU (no-op) ===\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, maximus::DeviceType::CPU);
    auto training_data = db->get_table("reviews");
    
    auto gpu_index = maximus::faiss::FaissIndex::build(ctx,
                                                        training_data,
                                                        "rv_embedding",
                                                        "GPU,Flat",
                                                        maximus::VectorDistanceMetric::L2,
                                                        false);
    
    // to_gpu on GPU index should return self
    auto same_index = gpu_index->to_gpu();
    
    ASSERT_TRUE(same_index->is_on_gpu());
    // Should be same pointer (shared_from_this)
    ASSERT_EQ(gpu_index.get(), same_index.get());
    std::cout << "to_gpu on GPU index returns same instance: " << (gpu_index.get() == same_index.get()) << "\n";
}

// Test: CPU Index stays on CPU when to_cpu called (no-op)
// #BUG: Assertion failure in gpu/cudf/parquet.cpp: ctx cannot be nullptr
TEST(VSDS, DISABLED_IndexMovement_CPU_to_CPU_NoOp) {
    std::string path = vsds_path();
    std::cout << "=== Test: CPU Index -> CPU (no-op) ===\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, maximus::DeviceType::CPU);
    auto training_data = db->get_table("reviews");
    
    auto cpu_index = maximus::faiss::FaissIndex::build(ctx,
                                                        training_data,
                                                        "rv_embedding",
                                                        "Flat",
                                                        maximus::VectorDistanceMetric::L2,
                                                        false);
    
    // to_cpu on CPU index should return self
    auto same_index = cpu_index->to_cpu();
    
    ASSERT_TRUE(same_index->is_on_cpu());
    // Should be same pointer (shared_from_this)
    ASSERT_EQ(cpu_index.get(), same_index.get());
    std::cout << "to_cpu on CPU index returns same instance: " << (cpu_index.get() == same_index.get()) << "\n";
}

// Test: ANN with GPU operator + CPU index -> auto-movement
// NOTE: This test may fail due to cuDF data type issues with list<float>
// #BUG: Assertion failure in gpu/cudf/parquet.cpp: ctx cannot be nullptr
TEST(VSDS, DISABLED_ANN_GPU_Operator_CPU_Index) {
    std::string path = vsds_path();
    std::cout << "=== Test: ANN GPU Operator with CPU Index (auto-movement) ===\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    auto device = maximus::DeviceType::GPU;
    
    // Build CPU index
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, maximus::DeviceType::CPU);
    auto training_data = db->get_table("reviews");
    
    auto cpu_index = maximus::faiss::FaissIndex::build(ctx,
                                                        training_data,
                                                        "rv_embedding",
                                                        "Flat",  // CPU index
                                                        maximus::VectorDistanceMetric::L2,
                                                        false);
    
    ASSERT_TRUE(cpu_index->is_on_cpu());
    std::cout << "Built CPU index\n";
    
    // Query with GPU operator - should auto-move index to GPU
    maximus::vsds::QueryParameters params;
    params.k = 10;
    params.faiss_index = "Flat";
    
    std::cout << "Creating GPU query with CPU index...\n";
    auto q = maximus::vsds::ann_reviews(db, cpu_index, device, params);
    
    std::cout << "Executing query (should auto-move index to GPU)...\n";
    auto table = db->query(q);
    
    ASSERT_TRUE(table != nullptr);
    std::cout << "Query succeeded with " << table->num_rows() << " rows\n";
    table->slice(0, 5)->print();
}

// Test: storage_device vs device separation
// NOTE: This test may fail due to cuDF data type issues
// #BUG: Assertion failure in gpu/cudf/parquet.cpp: ctx cannot be nullptr
TEST(VSDS, DISABLED_StorageDevice_DeviceSeparation) {
    std::string path = vsds_path();
    std::cout << "=== Test: storage_device vs device separation ===\n";

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    // Build GPU index, then move to CPU for storage
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, maximus::DeviceType::CPU);
    auto training_data = db->get_table("reviews");
    
    auto gpu_index = maximus::faiss::FaissIndex::build(ctx,
                                                        training_data,
                                                        "rv_embedding",
                                                        "GPU,Flat",
                                                        maximus::VectorDistanceMetric::L2,
                                                        false);
    
    // Move index to CPU for storage
    auto cpu_stored_index = gpu_index->to_cpu();
    ASSERT_TRUE(cpu_stored_index->is_on_cpu());
    std::cout << "GPU index moved to CPU for storage\n";
    
    // Now run GPU query with CPU-stored index
    // The GPU operator should auto-move the index back to GPU
    maximus::vsds::QueryParameters params;
    params.k = 10;
    params.faiss_index = "GPU,Flat";
    
    auto device = maximus::DeviceType::GPU;
    
    std::cout << "Running GPU query with CPU-stored index...\n";
    auto q = maximus::vsds::ann_reviews(db, cpu_stored_index, device, params);
    
    auto table = db->query(q);
    
    ASSERT_TRUE(table != nullptr);
    std::cout << "Query succeeded: " << table->num_rows() << " rows\n";
    table->slice(0, 5)->print();
}

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
    auto device = maximus::DeviceType::GPU;

    // Preload tables to match maxbench behavior (avoiding CSV parsing during query exec)
    db->load_table("part", maximus::vsds::schema("part"), {}, device);
    db->load_table("partsupp", maximus::vsds::schema("partsupp"), {}, device);
    db->load_table("supplier", maximus::vsds::schema("supplier"), {}, device);
    db->load_table("nation", maximus::vsds::schema("nation"), {}, device);
    db->load_table("region", maximus::vsds::schema("region"), {}, device);
    db->load_table("images", maximus::vsds::schema("images"), {}, device);
    db->load_table("images_queries", maximus::vsds::schema("images_queries"), {}, device);

    // Build index
    auto index_desc = "GPU,Flat";
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

    // std::string path = vsds_path_sf1(); // sf 1
    std::string path = vsds_path_sf001(); // sf 0.01 
    // std::string path = vsds_path(); // sf 0.001

    std::cout << "Path = " << path << "\n";
    // Setup database
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    int nqueries = 5;
    auto device = maximus::DeviceType::GPU;
    // Preload tables to match maxbench behavior (avoiding CSV parsing during query exec)
    db->load_table("lineitem", maximus::vsds::schema("lineitem"), {}, device);
    db->load_table("orders", maximus::vsds::schema("orders"), {}, device);
    db->load_table("customer", maximus::vsds::schema("customer"), {}, device);
    db->load_table("nation", maximus::vsds::schema("nation"), {}, device);
    db->load_table("region", maximus::vsds::schema("region"), {}, device);
    // NOTE: We do not load rv_embedding on the GPU because it will through (>2B elements, int32 limitation)
    db->load_table("reviews", maximus::vsds::schema("reviews"), {"rv_custkey"}, device); // we only need rv_custkey
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);

    // Build index from CPU-loaded embeddings (avoids cuDF large_list size limit)
    auto index_desc = "GPU,Flat";
    // NOTE: Just read the training data on THE CPU directly and pass it to the index
    // - used to avoid the int32 limitation of cuDF large_list 
    auto training_data = maximus::read_table(ctx, 
                                             path + "/reviews.parquet",
                                             maximus::vsds::schema("reviews"), 
                                             {"rv_embedding"},
                                             maximus::DeviceType::CPU);
    // This will be moved to the GPU...
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

    int nqueries = 10;
    auto device = maximus::DeviceType::GPU;

    // Preload tables to match maxbench behavior (avoiding CSV parsing during query exec)
    db->load_table("lineitem", maximus::vsds::schema("lineitem"), {}, device);
    db->load_table("orders", maximus::vsds::schema("orders"), {}, device);
    db->load_table("customer", maximus::vsds::schema("customer"), {}, device);
    db->load_table("nation", maximus::vsds::schema("nation"), {}, device);
    db->load_table("region", maximus::vsds::schema("region"), {}, device);
    db->load_table("images", maximus::vsds::schema("images"), {}, device);
    db->load_table("images_queries", maximus::vsds::schema("images_queries"), {}, device);

    // Build index
    auto index_desc = "GPU,Flat";
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
    auto device = maximus::DeviceType::GPU;
    
    // Preload tables to match maxbench behavior (avoiding CSV parsing during query exec)
    db->load_table("lineitem", maximus::vsds::schema("lineitem"), {}, device);
    db->load_table("orders", maximus::vsds::schema("orders"), {}, device);
    db->load_table("customer", maximus::vsds::schema("customer"), {}, device);
    db->load_table("nation", maximus::vsds::schema("nation"), {}, device);
    db->load_table("region", maximus::vsds::schema("region"), {}, device);
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);

    // Build index
    auto index_desc = "GPU,Flat";
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
    auto device = maximus::DeviceType::GPU;
    // Preload tables to match maxbench behavior (avoiding CSV parsing during query exec)
    db->load_table("lineitem", maximus::vsds::schema("lineitem"), {}, device);
    db->load_table("orders", maximus::vsds::schema("orders"), {}, device);
    db->load_table("customer", maximus::vsds::schema("customer"), {}, device);
    db->load_table("nation", maximus::vsds::schema("nation"), {}, device);
    db->load_table("region", maximus::vsds::schema("region"), {}, device);
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    // x2 vector search in this query
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);
    db->load_table("images", maximus::vsds::schema("images"), {}, device);
    db->load_table("images_queries", maximus::vsds::schema("images_queries"), {}, device);

    // Build indexes
    auto index_desc = "GPU,Flat";
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
    auto device = maximus::DeviceType::GPU;

    // Preload tables to match maxbench behavior (avoiding CSV parsing during query exec)
    db->load_table("partsupp", maximus::vsds::schema("partsupp"), {}, device);
    db->load_table("nation", maximus::vsds::schema("nation"), {}, device);
    db->load_table("supplier", maximus::vsds::schema("supplier"), {}, device);
    
    db->load_table("images", maximus::vsds::schema("images"), {}, device);
    db->load_table("images_queries", maximus::vsds::schema("images_queries"), {}, device);

    // Build index
    auto index_desc = "GPU,Flat";
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
    auto device = maximus::DeviceType::GPU;

    // Preload tables to match maxbench behavior (avoiding CSV parsing during query exec)
    db->load_table("partsupp", maximus::vsds::schema("partsupp"), {}, device);
    db->load_table("nation", maximus::vsds::schema("nation"), {}, device);
    db->load_table("supplier", maximus::vsds::schema("supplier"), {}, device);
    
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);

    // Build index
    auto index_desc = "GPU,Flat";
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
        params.k = 5;
        // only one query at a time
        params.query_count = 1;
        params.query_start = i;
        params.faiss_index = index_desc; // ENN (also should avoid GPU int32 limitation)
        params.postfilter_ksearch = 1000;
        params.use_post=true;

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
    auto device = maximus::DeviceType::GPU;

    // Preload tables to match maxbench behavior (avoiding CSV parsing during query exec)
    db->load_table("lineitem", maximus::vsds::schema("lineitem"), {}, device);
    db->load_table("partsupp", maximus::vsds::schema("partsupp"), {}, device);
    db->load_table("part", maximus::vsds::schema("part"), {}, device);
    db->load_table("nation", maximus::vsds::schema("nation"), {}, device);
    db->load_table("supplier", maximus::vsds::schema("supplier"), {}, device);
    
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);

    // Build index
    auto index_desc = "GPU,Flat";
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
    std::cout << "=== Test: Q1 GPU ENN ===" << std::endl;
    std::cout << "Not currently supported until we get around the cuDF large_list int32 limitation" << std::endl; 
}


TEST(VSDS, DISABLED_LOAD_SF1){
    std::string path = vsds_path_sf1();

    std::cout << "Path = " << path << "\n";
    
    // Setup database
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    auto device = maximus::DeviceType::GPU;

    // Preload tables 
    auto start = std::chrono::high_resolution_clock::now();
    // Don't load embeddings for  reviews on GPU (large list issue) - but we can still check if partitions work
    db->load_table("reviews", maximus::vsds::schema("reviews"),
                   {"rv_rating", "rv_helpful_vote", "rv_title", "rv_text", "rv_partkey", "rv_custkey", "rv_reviewkey"},
                   device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"),
                   {"rv_rating_queries", "rv_helpful_vote_queries", "rv_title_queries", "rv_text_queries", "rv_partkey_queries", "rv_custkey_queries", "rv_reviewkey_queries"},
                   device);

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
    ASSERT_GT(reviews.as_gtable()->get_num_rows(), 0);
    ASSERT_GT(images.as_gtable()->get_num_rows(), 0);
    std::cout << "Reviews rows: " << reviews.as_gtable()->get_num_rows() << "\n";
    std::cout << "Images rows:  " << images.as_gtable()->get_num_rows() << "\n";
    auto second_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> second_elapsed = second_end - second_start;
    std::cout << "Time to access loaded tables = " << second_elapsed.count() << " seconds" << std::endl;

    auto third_end = std::chrono::high_resolution_clock::now();
    auto reviews_2 = db->get_table_nocopy("reviews");
    auto images_2  = db->get_table_nocopy("images");
    ASSERT_TRUE(reviews_2);
    ASSERT_TRUE(images_2);
    ASSERT_GT(reviews_2.as_gtable()->get_num_rows(), 0);
    ASSERT_GT(images_2.as_gtable()->get_num_rows(), 0);
    std::cout << "Reviews rows (nocopy): " << reviews_2.as_gtable()->get_num_rows() << "\n";
    std::cout << "Images rows (nocopy):  " << images_2.as_gtable()->get_num_rows() << "\n"; 
    std::chrono::duration<double> third_elapsed = third_end - second_end;
    std::cout << "Time to access loaded tables (nocopy) = " << third_elapsed.count() << " seconds" << std::endl;
}

// =============================================================================
// Graph cache tests: cagra_cache_graph=1 with index_storage_device=cpu
// Requires APPLY_FAISS_CAGRA_PATCH=true in the container build.
// =============================================================================

// Graph cache + data=0 (non-owning, graph-only movement)
TEST(VSDS, DISABLED_ANN_Reviews_CAGRA_GraphCache_DataOnHost) {
    std::string path = vsds_path_sf1();
    std::cout << "Path = " << path << "\n";

    auto context      = maximus::make_context();
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue, context);

    const auto device               = maximus::DeviceType::GPU;
    const auto storage_device       = maximus::DeviceType::CPU;
    const auto index_storage_device = maximus::DeviceType::CPU;  // index on CPU
    const std::string index_desc    = "GPU,Cagra,64,32,NN_DESCENT";
    const std::vector<std::string> queries = {"ann_reviews"};
    const std::string params =
        "k=100,query_count=1,query_start=0,use_cuvs=1,metric=IP,"
        "cagra_itopksize=1024,index_data_on_gpu=0,cagra_cache_graph=1";

    auto parsed_params = maximus::vsds::parse_query_parameters(params);

    auto tables  = get_table_names("vsds");
    auto schemas = get_table_schemas("vsds");
    load_tables(db, tables, schemas, storage_device, {});

    auto indexes = build_indexes("vsds", index_desc, queries, db, context,
                                 storage_device, index_storage_device,
                                 true, false, "",
                                 parsed_params.metric, parsed_params.use_cuvs,
                                 false,  // pin_index_on_cpu
                                 parsed_params.index_data_on_gpu,
                                 parsed_params.cagra_cache_graph);

    ASSERT_TRUE(indexes.contains("reviews.rv_embedding"));
    ASSERT_TRUE(indexes["reviews.rv_embedding"]->is_on_cpu());

    auto query_plan = maximus::vsds::query_plan("ann_reviews", db, device, params, indexes, index_desc);
    ASSERT_TRUE(query_plan);

    auto table = db->query(query_plan);

    ASSERT_TRUE(table);
    std::cout << "Query result rows: " << table->num_rows() << "\n";
    table->slice(0, 10)->print();
}

// Graph cache + data=1 (owning, dataset copied to GPU — was broken before)
TEST(VSDS, DISABLED_ANN_Reviews_CAGRA_GraphCache_DataOnGPU) {
    std::string path = vsds_path_sf1();
    std::cout << "Path = " << path << "\n";

    auto context      = maximus::make_context();
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue, context);

    const auto device               = maximus::DeviceType::GPU;
    const auto storage_device       = maximus::DeviceType::CPU;
    const auto index_storage_device = maximus::DeviceType::CPU;  // index on CPU
    const std::string index_desc    = "GPU,Cagra,64,32,NN_DESCENT";
    const std::vector<std::string> queries = {"ann_reviews"};
    const std::string params =
        "k=100,query_count=1,query_start=0,use_cuvs=1,metric=IP,"
        "cagra_itopksize=1024,index_data_on_gpu=1,cagra_cache_graph=1";

    auto parsed_params = maximus::vsds::parse_query_parameters(params);

    auto tables  = get_table_names("vsds");
    auto schemas = get_table_schemas("vsds");
    load_tables(db, tables, schemas, storage_device, {});

    auto indexes = build_indexes("vsds", index_desc, queries, db, context,
                                 storage_device, index_storage_device,
                                 true, false, "",
                                 parsed_params.metric, parsed_params.use_cuvs,
                                 false,
                                 parsed_params.index_data_on_gpu,
                                 parsed_params.cagra_cache_graph);

    ASSERT_TRUE(indexes.contains("reviews.rv_embedding"));
    ASSERT_TRUE(indexes["reviews.rv_embedding"]->is_on_cpu());

    auto query_plan = maximus::vsds::query_plan("ann_reviews", db, device, params, indexes, index_desc);
    ASSERT_TRUE(query_plan);

    auto table = db->query(query_plan);

    ASSERT_TRUE(table);
    std::cout << "Query result rows: " << table->num_rows() << "\n";
    table->slice(0, 10)->print();
}

// =============================================================================
// Result Correctness: verify all CAGRA placement configurations return the
// same top-K results. Catches silent data placement errors.
// =============================================================================

// Helper: run a query and extract top-K (reviewkey, distance) pairs
static std::vector<std::pair<int64_t, float>> run_cagra_and_get_results(
    maximus::DeviceType index_storage_device,
    int index_data_on_gpu,
    int cagra_cache_graph)
{
    std::string path = vsds_path_sf001();
    auto context      = maximus::make_context();
    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue, context);

    const auto device         = maximus::DeviceType::GPU;
    const auto storage_device = maximus::DeviceType::CPU;
    const std::string index_desc = "GPU,Cagra,64,32,NN_DESCENT";
    const std::vector<std::string> queries = {"ann_reviews"};
    const std::string params_str =
        "k=10,query_count=1,query_start=0,use_cuvs=1,metric=IP,cagra_itopksize=128";
    auto parsed_params = maximus::vsds::parse_query_parameters(params_str);

    auto tables  = get_table_names("vsds");
    auto schemas = get_table_schemas("vsds");
    load_tables(db, tables, schemas, storage_device, {});

    auto indexes = build_indexes("vsds", index_desc, queries, db, context,
                                 storage_device, index_storage_device,
                                 true, false, "",
                                 parsed_params.metric, parsed_params.use_cuvs,
                                 false, index_data_on_gpu, cagra_cache_graph);

    auto query_plan = maximus::vsds::query_plan("ann_reviews", db, device, params_str,
                                                 indexes, index_desc);
    auto table = db->query(query_plan);
    EXPECT_TRUE(table);
    if (!table) return {};

    // Extract (reviewkey, distance) pairs from result
    std::vector<std::pair<int64_t, float>> results;
    auto arrow_table = table->get_table();
    auto key_col = arrow_table->GetColumnByName("rv_reviewkey");
    auto dist_col = arrow_table->GetColumnByName("vs_distance");
    if (!key_col || !dist_col) return {};

    auto keys = std::static_pointer_cast<arrow::Int64Array>(key_col->chunk(0));
    auto dists = std::static_pointer_cast<arrow::FloatArray>(dist_col->chunk(0));
    for (int64_t i = 0; i < keys->length(); ++i) {
        results.emplace_back(keys->Value(i), dists->Value(i));
    }
    return results;
}

TEST(VSDS, CAGRA_ResultCorrectness_AllConfigs) {
    std::cout << "\n=== CAGRA Result Correctness Test ===" << std::endl;
    std::cout << "Running 4 configurations and comparing top-K results...\n" << std::endl;

    // Config: {index_storage, data_on_gpu, cache_graph, label}
    struct Config {
        maximus::DeviceType storage;
        int data;
        int cache;
        const char* label;
    };
    std::vector<Config> configs = {
        {maximus::DeviceType::CPU, 0, 0, "storage=cpu, data=0, cache=0"},
        {maximus::DeviceType::CPU, 0, 1, "storage=cpu, data=0, cache=1"},
        {maximus::DeviceType::CPU, 1, 1, "storage=cpu, data=1, cache=1"},
        {maximus::DeviceType::GPU, 1, 0, "storage=gpu, data=1, cache=0"},
    };

    std::vector<std::vector<std::pair<int64_t, float>>> all_results;

    for (const auto& cfg : configs) {
        std::cout << "Running: " << cfg.label << std::endl;
        auto results = run_cagra_and_get_results(cfg.storage, cfg.data, cfg.cache);
        ASSERT_FALSE(results.empty()) << "No results for: " << cfg.label;
        std::cout << "  Top-3: ";
        for (int i = 0; i < std::min(3, (int)results.size()); ++i) {
            std::cout << "(" << results[i].first << ", " << results[i].second << ") ";
        }
        std::cout << std::endl;
        all_results.push_back(std::move(results));
    }

    // Compare all configs against the first (baseline)
    const auto& baseline = all_results[0];
    for (size_t c = 1; c < all_results.size(); ++c) {
        const auto& other = all_results[c];
        ASSERT_EQ(baseline.size(), other.size())
            << "Result count mismatch: " << configs[0].label << " vs " << configs[c].label;

        for (size_t i = 0; i < baseline.size(); ++i) {
            EXPECT_EQ(baseline[i].first, other[i].first)
                << "Key mismatch at row " << i << ": " << configs[0].label
                << " key=" << baseline[i].first << " vs " << configs[c].label
                << " key=" << other[i].first;
            EXPECT_NEAR(baseline[i].second, other[i].second, 1e-4)
                << "Distance mismatch at row " << i << ": " << configs[0].label
                << " dist=" << baseline[i].second << " vs " << configs[c].label
                << " dist=" << other[i].second;
        }
        std::cout << "  " << configs[c].label << " matches baseline ✓" << std::endl;
    }

    std::cout << "\n=== All configurations produce identical results ===" << std::endl;
}

// ============================================================================
// IVF cuVS GPU Memory Corruption Reproducer
//
// cuVS IVF search with K>256 uses raft's radix select, which causes GPU memory
// corruption when sharing an RMM pool with cuDF. The radix select's temp
// buffers overlap with cuDF column data, producing garbage indices that crash
// the downstream cudf::gather.
//
// Workaround: use_cuvs=0 for GPU IVF (uses FAISS's own top-k, no radix select).
// See: docs/ivf_gpu_crash_investigation.md (Investigation Log 3)
// ============================================================================

// Helper: run q2_start with given params, return row count or -1 on failure
static int run_q2(std::shared_ptr<maximus::Database>& db,
                  std::shared_ptr<maximus::faiss::FaissIndex>& index,
                  maximus::vsds::QueryParameters& params,
                  maximus::DeviceType device) {
    auto q = maximus::vsds::q2_start(db, index, device, params);
    auto table = db->query(q);
    return table ? table->num_rows() : -1;
}

// DISABLED: Reproduces cuVS IVF crash. Requires sf_1 data + pure GPU mode.
// Fix: set use_cuvs=0 for GPU IVF indexes (re-enable test with use_cuvs=false to verify fix).
TEST(VSDS_IVF, DISABLED_Q2_PostFilter_cuVS_sf1) {
    std::string path = vsds_path_sf1();
    std::cout << "=== q2_start cuVS IVF sf_1 (crash reproducer) ===" << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    db->load_table("images", maximus::vsds::schema("images"), {}, maximus::DeviceType::CPU);
    auto training_data = db->get_table("images");

    auto index = maximus::faiss::FaissIndex::build(
        ctx, training_data, "i_embedding", "GPU,IVF1024,Flat",
        maximus::VectorDistanceMetric::INNER_PRODUCT, true /* use_cuvs — crashes with K>256 */);
    ASSERT_NE(index, nullptr);
    std::cout << "Index: ntotal=" << index->faiss_index->ntotal << std::endl;

    auto device = maximus::DeviceType::GPU;

    maximus::vsds::QueryParameters params;
    params.faiss_index = "GPU,IVF1024,Flat";
    params.metric = maximus::VectorDistanceMetric::INNER_PRODUCT;
    params.ivf_nprobe = 30;
    params.use_cuvs = true;
    params.query_count = 1;
    params.query_start = 0;
    params.use_post = true;

    struct TestCase {
        int64_t k;
        int64_t postfilter_ksearch;
        int n_reps;
        std::string label;
    };
    std::vector<TestCase> cases = {
        {100, 100,  5, "K=100  postK=100  5reps"},
        {100, 500,  5, "K=100  postK=500  5reps"},
        {100, 1000, 3, "K=100  postK=1000 3reps (crash case)"},
        {100, 1000, 5, "K=100  postK=1000 5reps"},
    };

    for (auto& tc : cases) {
        std::cout << "\n--- " << tc.label << " ---" << std::endl;
        params.k = tc.k;
        params.postfilter_ksearch = tc.postfilter_ksearch;

        for (int rep = 0; rep < tc.n_reps; ++rep) {
            int rows = run_q2(db, index, params, device);
            std::cout << "  rep " << rep << ": rows=" << rows << std::endl;
            EXPECT_GE(rows, 0) << "Query failed at rep " << rep;
        }
    }
}