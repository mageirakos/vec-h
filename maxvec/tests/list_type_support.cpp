/**
 * @file list_type_support.cpp
 * @brief Test suite to investigate LIST column type support across CPU/GPU operators
 *
 * This test suite systematically tests which operators work with LIST columns
 * (arrow::ListArray, arrow::LargeListArray) used for embedding vectors.
 *
 * The goal is to identify and document limitations, not fix them.
 * Expected failures are marked as DISABLED_ or use EXPECT_THROW.
 */

#include <gtest/gtest.h>

#include <iostream>
#include <maximus/database.hpp>
#include <maximus/frontend/expressions.hpp>
#include <maximus/frontend/query_plan_api.hpp>
#include <maximus/indexes/faiss/faiss_index.hpp>
#include <maximus/tpch/tpch_queries.hpp>
#include <maximus/types/types.hpp>
#include <maximus/vsds/vsds_queries.hpp>

namespace test {

// =============================================================================
// Test Data Path Helper
// =============================================================================

std::string vsds_test_path() {
    const char* env_p = std::getenv("VSDS_DATA_PATH");
    if (env_p) return env_p;
    
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/vsds/data-industrial_and_scientific-sf_0.001";
    return path;
}

// CPU Tests:

// =============================================================================
// SECTION 1: Loading LIST Types
// Tests that ListArray and LargeListArray can be loaded from parquet
// =============================================================================

/**
 * Test loading embeddings as arrow::ListArray (default schema)
 * 
 * VSDS uses embeddings_list() which creates arrow::list(float32).
 * This test verifies loading works on CPU.
 */
TEST(ListTypeSupport, LoadListArrayFromVSDS_CPU) {
    std::string path = vsds_test_path();
    std::cout << "=== Test: Load ListArray from VSDS (CPU) ===" << std::endl;
    std::cout << "Path = " << path << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    // Load reviews table on CPU with embedding column
    auto schema = maximus::vsds::schema("reviews");
    std::cout << "Schema: " << schema->to_string() << std::endl;

    ASSERT_NO_THROW({
        db->load_table("reviews", schema, {}, maximus::DeviceType::CPU);
    });

    auto table_data = db->get_table("reviews");
    ASSERT_FALSE(table_data.empty());
    
    // Get arrow table from DeviceTablePtr
    auto arrow_table = table_data.as_table()->get_table();
    auto embedding_col = arrow_table->GetColumnByName("rv_embedding");
    ASSERT_NE(embedding_col, nullptr);
    
    // Verify it's a LIST type
    auto type = embedding_col->type();
    std::cout << "Embedding column type: " << type->ToString() << std::endl;
    EXPECT_EQ(type->id(), arrow::Type::LIST);
    
    std::cout << "PASS: Loaded " << arrow_table->num_rows() << " rows with ListArray embedding column" << std::endl;
}

/**
 * Test loading embeddings as arrow::LargeListArray
 * 
 * LargeListArray uses int64 offsets (for datasets > 2^31 elements).
 * Create a schema with large_list and verify loading works.
 */
TEST(ListTypeSupport, LoadLargeListArrayFromVSDS_CPU) {
    std::string path = vsds_test_path();
    std::cout << "=== Test: Load LargeListArray from VSDS (CPU) ===" << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);

    // Create schema with LargeListArray for embedding
    auto fields = {
        arrow::field("rv_rating", arrow::float64(), true),
        arrow::field("rv_helpful_vote", arrow::int64(), false),
        arrow::field("rv_title", arrow::utf8(), true),
        arrow::field("rv_text", arrow::utf8(), true),
        arrow::field("rv_embedding", arrow::large_list(arrow::float32())),  // LargeListArray!
        arrow::field("rv_partkey", arrow::int64(), false),
        arrow::field("rv_custkey", arrow::int64(), false),
        arrow::field("rv_reviewkey", arrow::int64(), false)
    };
    auto schema = std::make_shared<maximus::Schema>(fields);

    std::cout << "Schema with LargeListArray: " << schema->to_string() << std::endl;

    ASSERT_NO_THROW({
        db->load_table("reviews", schema, {}, maximus::DeviceType::CPU);
    });

    auto table_data = db->get_table("reviews");
    ASSERT_FALSE(table_data.empty());
    
    auto arrow_table = table_data.as_table()->get_table();
    auto embedding_col = arrow_table->GetColumnByName("rv_embedding");
    ASSERT_NE(embedding_col, nullptr);
    
    auto type = embedding_col->type();
    std::cout << "Embedding column type: " << type->ToString() << std::endl;
    EXPECT_EQ(type->id(), arrow::Type::LARGE_LIST);
    
    std::cout << "PASS: Loaded " << arrow_table->num_rows() << " rows with LargeListArray embedding column" << std::endl;
}

// =============================================================================
// SECTION 2: CPU Operators (Acero) with LIST columns
// Tests filter, project, hash_join on CPU with LIST columns
// =============================================================================

/**
 * Test CPU filter operator with table containing LIST column.
 * Filter on a non-list column while LIST column is in the table.
 * 
 * Expected: PASS - filter doesn't need to process the LIST values
 */
TEST(ListTypeSupport, CPUFilterWithListColumn) {
    std::string path = vsds_test_path();
    std::cout << "=== Test: CPU Filter with LIST column ===" << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto device       = maximus::DeviceType::CPU;

    // Load reviews with embedding
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    
    //Before start of plan
    std::cout << "Loaded table:\n";
    db->get_table("reviews").as_table()->slice(0, 5)->print();

    // Create query: filter on rv_rating >= 4.0 (keep embedding column)
    namespace cp = ::arrow::compute;
    
    auto source = maximus::table_source(db, "reviews", maximus::vsds::schema("reviews"),
                               {"rv_reviewkey", "rv_embedding", "rv_rating", "rv_custkey"}, device);
    
    auto filter_expr = maximus::expr(maximus::arrow_expr(cp::field_ref("rv_rating"), ">=", maximus::float64_literal(4.0)));
    auto filtered = maximus::filter(source, filter_expr, device);
    
    auto sink = maximus::table_sink(filtered);
    auto qp = maximus::query_plan(sink);
    
    std::cout << "Query Plan:\n" << qp->to_string() << std::endl;
    
    maximus::TablePtr result;
    ASSERT_NO_THROW({
        result = db->query(qp);
    });
    
    ASSERT_TRUE(result != nullptr);
    std::cout << "PASS: Filter returned " << result->num_rows() << " rows" << std::endl;
    
    // Verify embedding column still exists
    auto embedding_col = result->get_table()->GetColumnByName("rv_embedding");
    ASSERT_NE(embedding_col, nullptr);
    EXPECT_EQ(embedding_col->type()->id(), arrow::Type::LIST);
    
    // Print first 5 rows
    std::cout << "Result:\n";
    result->slice(0, 5)->print();
}


/**
 * Test CPU project operator with LIST column renaming and exhaustive vector join
 * 
 * Expected: PASS - simple pass-through projection should work
 */
TEST(ListTypeSupport, CPUProjectRenameExhaustiveVectorJoinWithListColumn) {
    std::string path = vsds_test_path();
    std::cout << "=== Test: CPU Project rename with LIST column ===" << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto device       = maximus::DeviceType::CPU;

    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);

    //Before start of plan
    std::cout << "Loaded tables:\n";
    db->get_table("reviews").as_table()->slice(0, 5)->print();
    db->get_table("reviews_queries").as_table()->slice(0, 1)->print();

    // Simple pass-through projection (no expressions on LIST)
    auto source = maximus::table_source(db, "reviews", maximus::vsds::schema("reviews"),
                               {"rv_reviewkey", "rv_embedding", "rv_rating", "rv_custkey"}, device);
    auto source_queries = maximus::table_source(db, "reviews_queries", maximus::vsds::schema("reviews_queries"),
                               {"rv_reviewkey_queries", "rv_embedding_queries", "rv_rating_queries", "rv_custkey_queries"}, device);
    
    // Rename columns
    auto projected = maximus::rename(source, {"rv_reviewkey", "rv_embedding", "rv_custkey"}, {"data_key", "data_embd", "data_custkey"}, device);
    
    // Do a vector search after we renamed
    auto vs_enn_node = maximus::exhaustive_vector_join(
        projected,
        source_queries,
        "data_embd",
        "rv_embedding_queries",
        maximus::VectorDistanceMetric::L2,
        10,        // K
        std::nullopt,    // no radius
        false,           // don't keep data vector
        false,           // don't keep query vector
        "distance",      // distance column name
        device
    );

    auto qp = maximus::query_plan(maximus::table_sink(vs_enn_node));
    
    std::cout << "Query Plan:\n" << qp->to_string() << std::endl;
    
    maximus::TablePtr result;
    ASSERT_NO_THROW({
        result = db->query(qp);
    });
    
    ASSERT_TRUE(result != nullptr);
    std::cout << "PASS: Project returned " << result->num_rows() << " rows" << std::endl;
    
    auto renamed_key_col = result->get_table()->GetColumnByName("data_key");
    ASSERT_NE(renamed_key_col, nullptr);

    // Print first 5 rows
    std::cout << "Result:\n";
    result->slice(0, 5)->print();
}


/**
 * Test CPU hash join with LIST column as non-key (payload) field.
 * 
 * Expected: FAIL - Acero's hash join doesn't support LIST in non-key fields.
 * Error: "Data type list<item: float> is not supported in join non-key field"
 *
 * POSSIBLE FIX: 
 * 1. Strip LIST columns before join, join on keys only, then re-join to get LIST columns
 * 2. Implement custom join operator that handles LIST columns
 * 3. Use semi-join pattern (as done in prejoin_reviews)
 */
// TEST(ListTypeSupport, DISABLED_CPUHashJoinWithListPayload) {
TEST(ListTypeSupport, CPUHashJoinWithListPayload) {

    std::string path = vsds_test_path();
    std::cout << "=== Test: CPU Hash Join with LIST payload column ===" << std::endl;
    // std::cout << "EXPECTED: FAIL - Acero doesn't support LIST in join non-key fields" << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto device       = maximus::DeviceType::CPU;

    // Load both tables
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("part", maximus::tpch::schema("part"), {}, device);

    // Try to join reviews with part, keeping embedding in output
    auto reviews_source = maximus::table_source(db, "reviews", maximus::vsds::schema("reviews"),
                                        {"rv_reviewkey", "rv_embedding", "rv_partkey"}, device);
    
    auto part_source = maximus::table_source(db, "part", maximus::tpch::schema("part"),
                                     {"p_partkey", "p_size"}, device);
    
    // This join should FAIL because rv_embedding (LIST) is a non-key column
    auto joined = maximus::inner_join(reviews_source, part_source,
                             {"rv_partkey"}, {"p_partkey"},
                             "", "",
                             device);
    
    auto qp = maximus::query_plan(maximus::table_sink(joined));
    
    std::cout << "Query Plan:\n" << qp->to_string() << std::endl;
    
    // Print the exception message
    try {
        auto result = db->query(qp);
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        GTEST_FAIL() << "CPU Hash Join with LIST column failed: " << e.what();
    }

    std::cout << "WORKED!" << std::endl;
}

/**
 * Workaround test: Semi-join + is_in filter to preserve LIST column.
 * 
 * Pattern:
 * 1. Semi-join without LIST column to get filtered keys
 * 2. Use is_in to create boolean mask
 * 3. Filter full table (with LIST) using the mask
 */
TEST(ListTypeSupport, CPUSemiJoinIsInWorkaround_WithListColumn) {
    std::string path = vsds_test_path();
    std::cout << "=== Test: CPU Semi-Join + IsIn workaround - preserve LIST column ===" << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto device       = maximus::DeviceType::CPU;

    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("part", maximus::tpch::schema("part"), {}, device);

    // Get the full reviews table directly for the is_in filter step
    auto reviews_data = db->get_table("reviews");
    auto full_reviews_table = reviews_data.as_table()->get_table();
    
    std::cout << "Full reviews table (with embedding):" << std::endl;
    reviews_data.as_table()->slice(0, 3)->print();

    // ========================================================================
    // STEP 1: Semi-join to get matching reviewkeys (no LIST through join)
    // ========================================================================
    
    // Reviews source - only key columns, NO LIST
    auto reviews_keys_only = maximus::table_source(db, "reviews", maximus::vsds::schema("reviews"),
                                         {"rv_reviewkey", "rv_partkey"}, device);
    
    // Part table source
    auto part_source = maximus::table_source(db, "part", maximus::tpch::schema("part"),
                                     {"p_partkey", "p_size"}, device);
    
    // Filter parts: p_size >= 10
    namespace cp = ::arrow::compute;
    auto part_filter_expr = maximus::expr(maximus::arrow_expr(cp::field_ref("p_size"), ">=", maximus::int32_literal(10)));
    auto filtered_parts = maximus::filter(part_source, part_filter_expr, device);
    auto filtered_parts_proj = maximus::project(filtered_parts, {"p_partkey"}, device);
    
    // Semi-join: get reviews keys that match filtered parts
    auto matched_keys = maximus::left_semi_join(reviews_keys_only, filtered_parts_proj,
                             {"rv_partkey"}, {"p_partkey"},
                             "", "",
                             device);
    
    // Project to just get the reviewkey
    auto matched_reviewkeys = maximus::project(matched_keys, {"rv_reviewkey"}, device);
    
    auto qp_keys = maximus::query_plan(maximus::table_sink(matched_reviewkeys));
    
    std::cout << "Query Plan (semi-join for keys):" << std::endl << qp_keys->to_string() << std::endl;
    
    maximus::TablePtr keys_result;
    ASSERT_NO_THROW({
        keys_result = db->query(qp_keys);
    });
    
    ASSERT_TRUE(keys_result != nullptr);
    std::cout << "Semi-join returned " << keys_result->num_rows() << " matching keys" << std::endl;
    keys_result->slice(0, 5)->print();

    // ========================================================================
    // STEP 2: Use is_in to create boolean mask
    // ========================================================================
    
    // Get the key column from semi-join result as a set
    auto matched_keys_column = keys_result->get_table()->GetColumnByName("rv_reviewkey");
    ASSERT_NE(matched_keys_column, nullptr);
    
    // Convert chunked array to array for set lookup
    auto matched_keys_combined = arrow::Concatenate(matched_keys_column->chunks()).ValueOrDie();
    
    // Create SetLookupOptions with the key set
    cp::SetLookupOptions lookup_opts(matched_keys_combined);
    
    // Get rv_reviewkey from full table for is_in comparison
    auto full_reviewkeys = full_reviews_table->GetColumnByName("rv_reviewkey");
    ASSERT_NE(full_reviewkeys, nullptr);
    
    // Call is_in: for each key in full table, is it in the matched set?
    auto is_in_result = cp::CallFunction("is_in", {full_reviewkeys}, &lookup_opts);
    ASSERT_TRUE(is_in_result.ok()) << "is_in failed: " << is_in_result.status().message();
    
    auto mask = is_in_result.ValueOrDie();
    std::cout << "Created is_in mask with " << mask.length() << " elements" << std::endl;

    // ========================================================================
    // STEP 3: Filter full table (with LIST) using the mask
    // ========================================================================
    
    auto filter_result = cp::Filter(full_reviews_table, mask);
    ASSERT_TRUE(filter_result.ok()) << "Filter failed: " << filter_result.status().message();
    
    auto filtered_table = filter_result.ValueOrDie().table();
    ASSERT_NE(filtered_table, nullptr);
    
    std::cout << "PASS: Filtered table has " << filtered_table->num_rows() << " rows" << std::endl;
    
    // Verify embedding column is present
    auto embedding_col = filtered_table->GetColumnByName("rv_embedding");
    ASSERT_NE(embedding_col, nullptr) << "Embedding column should be present!";
    EXPECT_EQ(embedding_col->type()->id(), arrow::Type::LIST);
    
    std::cout << "SUCCESS: LIST column preserved after join-like operation!" << std::endl;
    
    // Print sample results
    auto result_wrapped = std::make_shared<maximus::Table>(db->get_context(), filtered_table);
    result_wrapped->slice(0, 5)->print();
}

/**
 * TEST: Complete join result with LIST column preserved.
 * 
 * Goal: Produce a join result containing:
 *   - Columns from reviews (including rv_embedding LIST column)
 *   - Columns from part table (e.g., p_size)
 * 
 * Pattern:
 * 1. Inner join reviews (no LIST) with part → get joined result with p_size
 * 2. Use rv_reviewkey from join to filter full reviews (with LIST)
 * 3. Use Arrow to add p_size column to filtered reviews
 */
TEST(ListTypeSupport, CPUCompleteJoinWithListColumn) {
    std::string path = vsds_test_path();
    std::cout << "=== Test: CPU Complete Join with LIST column preserved ===" << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto device       = maximus::DeviceType::CPU;

    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("part", maximus::tpch::schema("part"), {}, device);

    // Get the full reviews table for later
    auto reviews_data = db->get_table("reviews");
    auto full_reviews_table = reviews_data.as_table()->get_table();

    // ========================================================================
    // STEP 1: Inner join without LIST to get columns from both tables
    // ========================================================================
    
    // Reviews: key columns only (no LIST!)
    auto reviews_keys = maximus::table_source(db, "reviews", maximus::vsds::schema("reviews"),
                                     {"rv_reviewkey", "rv_partkey"}, device);
    
    // Part table with filter
    namespace cp = ::arrow::compute;
    auto part_source = maximus::table_source(db, "part", maximus::tpch::schema("part"),
                                     {"p_partkey", "p_size"}, device);
    auto part_filter_expr = maximus::expr(maximus::arrow_expr(cp::field_ref("p_size"), ">=", maximus::int32_literal(15)));
    auto filtered_parts = maximus::filter(part_source, part_filter_expr, device);
    
    // Inner join - works because no LIST columns
    auto joined = maximus::inner_join(reviews_keys, filtered_parts,
                             {"rv_partkey"}, {"p_partkey"},
                             "", "",
                             device);
    
    auto qp_join = maximus::query_plan(maximus::table_sink(joined));
    std::cout << "Query Plan (join without LIST):\n" << qp_join->to_string() << std::endl;
    
    maximus::TablePtr join_result;
    ASSERT_NO_THROW({
        join_result = db->query(qp_join);
    });
    
    ASSERT_TRUE(join_result != nullptr);
    std::cout << "Join result (no LIST): " << join_result->num_rows() << " rows" << std::endl;
    join_result->slice(0, 3)->print();

    // ========================================================================
    // STEP 2: Get matching reviews with LIST column using is_in
    // ========================================================================
    
    // Get rv_reviewkey from join result
    auto joined_reviewkeys = join_result->get_table()->GetColumnByName("rv_reviewkey");
    ASSERT_NE(joined_reviewkeys, nullptr);
    auto joined_keys_combined = arrow::Concatenate(joined_reviewkeys->chunks()).ValueOrDie();
    
    // Create mask for full reviews table
    cp::SetLookupOptions lookup_opts(joined_keys_combined);
    auto full_reviewkeys = full_reviews_table->GetColumnByName("rv_reviewkey");
    auto is_in_result = cp::CallFunction("is_in", {full_reviewkeys}, &lookup_opts);
    ASSERT_TRUE(is_in_result.ok());
    auto mask = is_in_result.ValueOrDie();
    
    // Filter full reviews (with LIST)
    auto filter_result = cp::Filter(full_reviews_table, mask);
    ASSERT_TRUE(filter_result.ok());
    auto filtered_reviews = filter_result.ValueOrDie().table();
    
    std::cout << "Filtered reviews (with LIST): " << filtered_reviews->num_rows() << " rows" << std::endl;

    // ========================================================================
    // STEP 3: Merge into single table with both p_size and embedding
    // ========================================================================
    
    // Sort both tables by rv_reviewkey for alignment
    auto sort_opts = cp::SortOptions({cp::SortKey("rv_reviewkey", cp::SortOrder::Ascending)});
    
    // Sort join result
    auto joined_table = join_result->get_table();
    auto joined_indices = cp::CallFunction("sort_indices", {joined_table}, &sort_opts).ValueOrDie();
    auto sorted_join = cp::Take(joined_table, joined_indices).ValueOrDie().table();
    
    // Sort filtered reviews
    auto filtered_indices = cp::CallFunction("sort_indices", {filtered_reviews}, &sort_opts).ValueOrDie();
    auto sorted_reviews = cp::Take(filtered_reviews, filtered_indices).ValueOrDie().table();
    
    // Verify keys match after sorting
    auto sorted_join_keys = sorted_join->GetColumnByName("rv_reviewkey");
    auto sorted_reviews_keys = sorted_reviews->GetColumnByName("rv_reviewkey");
    ASSERT_EQ(sorted_join_keys->length(), sorted_reviews_keys->length());
    
    // Get p_size column from sorted join result
    auto p_size_sorted = sorted_join->GetColumnByName("p_size");
    ASSERT_NE(p_size_sorted, nullptr);
    
    // Add p_size to sorted reviews (which has embedding)
    auto result_status = sorted_reviews->AddColumn(
        sorted_reviews->num_columns(),  // Add at end
        arrow::field("p_size", arrow::int32()),
        p_size_sorted
    );
    ASSERT_TRUE(result_status.ok()) << result_status.status().message();
    auto complete_result = result_status.ValueOrDie();
    
    // Verify complete result has BOTH p_size and rv_embedding
    auto final_p_size = complete_result->GetColumnByName("p_size");
    auto final_embedding = complete_result->GetColumnByName("rv_embedding");
    
    ASSERT_NE(final_p_size, nullptr) << "Complete result must have p_size";
    ASSERT_NE(final_embedding, nullptr) << "Complete result must have rv_embedding";
    EXPECT_EQ(final_embedding->type()->id(), arrow::Type::LIST);
    
    std::cout << "SUCCESS: Complete merged table with " << complete_result->num_rows() 
              << " rows and " << complete_result->num_columns() << " columns" << std::endl;
    std::cout << "Schema: " << complete_result->schema()->ToString() << std::endl;
    
    // Print sample of complete result
    std::cout << "\nComplete merged result (has BOTH p_size AND embedding):" << std::endl;
    auto complete_wrapped = std::make_shared<maximus::Table>(db->get_context(), complete_result);
    complete_wrapped->slice(0, 3)->print();
}

// =============================================================================
// SECTION 3: GPU Operators (cuDF) with LIST columns
// Tests Arrow→cuDF conversion, filter, project on GPU
// =============================================================================

#ifdef MAXIMUS_WITH_CUDA

/**
 * Test loading data on CPU then converting to GPU (Arrow → cuDF).
 * 
 * Arrow LIST/LARGE_LIST should convert to cudf::LIST successfully.
 * The conversion uses arrow_to_cudf_type which supports LIST.
 */
TEST(ListTypeSupport, ArrowToCuDFListConversion) {
    std::string path = vsds_test_path();
    std::cout << "=== Test: Arrow to cuDF LIST conversion ===" << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    // Load on CPU first
    auto schema = maximus::vsds::schema("reviews");
    db->load_table("reviews", schema, {}, maximus::DeviceType::CPU);
    auto cpu_data = db->get_table("reviews");
    ASSERT_FALSE(cpu_data.empty());
    
    std::cout << "Loaded on CPU: " << cpu_data.as_table()->num_rows() << " rows" << std::endl;
    
    // TODO: we should convert directly not through GTablePtr? DeviceTablePtr -> convet to GPU? 
    // Try to convert to GPU (this triggers Arrow → cuDF conversion)
    ASSERT_NO_THROW({
        cpu_data.convert_to<maximus::GTablePtr>(ctx, schema);
    });
    
    ASSERT_TRUE(cpu_data.on_gpu());
    std::cout << "PASS: Converted to GPU successfully" << std::endl;
}

/**
 * Test GPU filter operator with LIST column.
 * 
 * Expected: May pass if filter just passes LIST through without processing
 */
TEST(ListTypeSupport, GPUFilterWithListColumn) {
    std::string path = vsds_test_path();
    std::cout << "=== Test: GPU Filter with LIST column ===" << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto device       = maximus::DeviceType::GPU;

    // Load on CPU first (operators handle device conversion)
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, maximus::DeviceType::CPU);

    namespace cp = ::arrow::compute;
    
    auto source = maximus::table_source(db, "reviews", maximus::vsds::schema("reviews"),
                               {"rv_reviewkey", "rv_embedding", "rv_rating"}, device);
    
    auto filter_expr = maximus::expr(maximus::arrow_expr(cp::field_ref("rv_rating"), ">=", maximus::float64_literal(4.0)));
    auto filtered = maximus::filter(source, filter_expr, device);
    
    auto qp = maximus::query_plan(maximus::table_sink(filtered));
    
    std::cout << "Query Plan:\n" << qp->to_string() << std::endl;
    
    maximus::TablePtr result = nullptr;
    try {
        result = db->query(qp);
        std::cout << "PASS: GPU Filter returned " << result->num_rows() << " rows" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "FAIL: GPU Filter threw exception: " << e.what() << std::endl;
        FAIL() << "GPU Filter failed: " << e.what();
    }
    
    ASSERT_TRUE(result != nullptr);
}

/**
 * Test GPU project operator with LIST column.
 * 
 * Expected: May FAIL if project tries to use AST on LIST column.
 *
 * POSSIBLE FIX:
 * Add case for cudf::type_id::LIST in cudf_expr.cpp get_expr_type function.
 */
TEST(ListTypeSupport, GPUProjectWithListColumn) {
    std::string path = vsds_test_path();
    std::cout << "=== Test: GPU Project with LIST column ===" << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto device       = maximus::DeviceType::GPU;

    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, maximus::DeviceType::CPU);

    auto source = maximus::table_source(db, "reviews", maximus::vsds::schema("reviews"),
                               {"rv_reviewkey", "rv_embedding", "rv_rating"}, device);
    
    // Simple projection - just select columns
    auto projected = maximus::project(source, {"rv_reviewkey", "rv_embedding"}, device);
    
    auto qp = maximus::query_plan(maximus::table_sink(projected));
    
    std::cout << "Query Plan:\n" << qp->to_string() << std::endl;
    
    maximus::TablePtr result = nullptr;
    try {
        result = db->query(qp);
        std::cout << "PASS: GPU Project returned " << result->num_rows() << " rows" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "FAIL: GPU Project threw exception: " << e.what() << std::endl;
        std::cout << "  This is expected if AST doesn't support LIST type" << std::endl;
        std::cout << "  POSSIBLE FIX: Add LIST case in cudf_expr.cpp get_expr_type()" << std::endl;
        GTEST_FAIL() << "GPU Project failed with LIST column: " << e.what();
        // GTEST_SKIP() << "GPU Project failed with LIST column: " << e.what();
    }
    
    EXPECT_TRUE(result != nullptr);
}

/**
 * Test cuDF to Arrow conversion for LIST columns.
 * This is the CRITICAL blocker - id_to_arrow_type doesn't have LIST case.
 * 
 * Expected: FAIL
 * Error: "Unsupported type_id conversion to arrow type"
 *
 * POSSIBLE FIX:
 * Add case for cudf::type_id::LIST in:
 * 1. cudf/cpp/src/interop/arrow_utilities.cpp: id_to_arrow_type()
 * 2. src/maximus/gpu/cudf/cudf_types.cpp: to_arrow_type()
 * 
 * Note: For LIST types, we need to also convert the child type.
 */
TEST(ListTypeSupport, CuDFToArrowListConversion) {
    std::string path = vsds_test_path();
    std::cout << "=== Test: cuDF to Arrow LIST conversion (CRITICAL) ===" << std::endl;
    std::cout << "EXPECTED: FAIL - id_to_arrow_type missing LIST case" << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();

    // Load on CPU
    auto schema = maximus::vsds::schema("reviews");
    db->load_table("reviews", schema, {}, maximus::DeviceType::CPU);
    auto cpu_data = db->get_table("reviews");
    
    // Convert to GPU
    cpu_data.convert_to<maximus::GTablePtr>(ctx, schema);
    ASSERT_TRUE(cpu_data.on_gpu());
    
    std::cout << "Data is on GPU, attempting to convert back to Arrow..." << std::endl;
    
    // Try to convert back to Arrow (this should fail for LIST columns)
    bool conversion_failed = false;
    try {
        cpu_data.convert_to<maximus::ArrowTablePtr>(ctx, schema);
        auto arrow_table = cpu_data.as_arrow_table();
        std::cout << "UNEXPECTED: Conversion succeeded!" << std::endl;
        std::cout << "Schema: " << arrow_table->schema()->ToString() << std::endl;
    } catch (const std::exception& e) {
        conversion_failed = true;
        std::cout << "CONFIRMED: cuDF→Arrow conversion failed for LIST: " << e.what() << std::endl;
        std::cout << std::endl;
        std::cout << "POSSIBLE FIX: Add LIST case in id_to_arrow_type():" << std::endl;
        std::cout << "  File: cudf/cpp/src/interop/arrow_utilities.cpp" << std::endl;
        std::cout << "  Function: ArrowType id_to_arrow_type(cudf::type_id id)" << std::endl;
        std::cout << "  Add: case cudf::type_id::LIST: return NANOARROW_TYPE_LIST;" << std::endl;
        std::cout << std::endl;
        std::cout << "  Also update Maximus wrapper:" << std::endl;
        std::cout << "  File: src/maximus/gpu/cudf/cudf_types.cpp" << std::endl;
        std::cout << "  Function: to_arrow_type()" << std::endl;
        std::cout << "  Need to also handle child type conversion using cudf::lists_column_view" << std::endl;
    }
    
    // We expect the conversion to fail
    EXPECT_TRUE(conversion_failed) << "Expected cuDF→Arrow conversion to fail for LIST type";
}

/**
 * Test loading parquet directly to GPU.
 * 
 * Expected: May fail due to known ctx assertion bug in gpu/cudf/parquet.cpp
 */
TEST(ListTypeSupport, DISABLED_DirectGPUParquetLoadWithList) {
    std::string path = vsds_test_path();
    std::cout << "=== Test: Direct GPU parquet load with LIST column ===" << std::endl;
    std::cout << "DISABLED: Known ctx assertion bug in gpu/cudf/parquet.cpp" << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto device       = maximus::DeviceType::GPU;

    // Try to load directly to GPU
    try {
        db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
        auto table_data = db->get_table("reviews");
        
        if (!table_data.empty()) {
            std::cout << "PASS: Loaded rows directly to GPU" << std::endl;
        } else {
            std::cout << "FAIL: Table is empty after GPU load" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "FAIL: Direct GPU load failed: " << e.what() << std::endl;
        FAIL() << "Direct GPU load failed: " << e.what();
    }
}

/**
 * Test GPU hash join with LIST payload column.
 * Similar to CPU test - cuDF join may have same limitation.
 */
TEST(ListTypeSupport, DISABLED_GPUHashJoinWithListPayload) {
    std::string path = vsds_test_path();
    std::cout << "=== Test: GPU Hash Join with LIST payload column ===" << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto device       = maximus::DeviceType::GPU;

    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, maximus::DeviceType::CPU);
    db->load_table("part", maximus::tpch::schema("part"), {}, maximus::DeviceType::CPU);

    auto reviews_source = maximus::table_source(db, "reviews", maximus::vsds::schema("reviews"),
                                        {"rv_reviewkey", "rv_embedding", "rv_partkey"}, device);
    
    auto part_source = maximus::table_source(db, "part", maximus::tpch::schema("part"),
                                     {"p_partkey", "p_size"}, device);
    
    auto joined = maximus::inner_join(reviews_source, part_source,
                             {"rv_partkey"}, {"p_partkey"},
                             "", "",
                             device);
    
    auto qp = maximus::query_plan(maximus::table_sink(joined));
    
    std::cout << "Query Plan:\n" << qp->to_string() << std::endl;
    
    try {
        auto result = db->query(qp);
        std::cout << "UNEXPECTED: GPU join with LIST payload succeeded!" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "CONFIRMED: GPU join fails with LIST payload: " << e.what() << std::endl;
    }
}

#endif  // MAXIMUS_WITH_CUDA

// =============================================================================
// SECTION 4: Vector Join Operators with LIST columns
// These are the primary use case - indexed and exhaustive vector joins
// =============================================================================

/**
 * Test exhaustive vector join on CPU (ENN).
 * This is the golden path - should always work.
 */
TEST(ListTypeSupport, ExhaustiveVectorJoin_CPU) {
    std::string path = vsds_test_path();
    std::cout << "=== Test: Exhaustive Vector Join (CPU) ===" << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto device       = maximus::DeviceType::CPU;

    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);

    maximus::vsds::QueryParameters params;
    params.k = 5;
    
    auto qp = maximus::vsds::enn_reviews(db, device, params);
    
    std::cout << "Query Plan:\n" << qp->to_string() << std::endl;
    
    maximus::TablePtr result;
    ASSERT_NO_THROW({
        result = db->query(qp);
    });
    
    ASSERT_TRUE(result != nullptr);
    std::cout << "PASS: Exhaustive vector join returned " << result->num_rows() << " rows" << std::endl;
    
    // Print sample results
    if (result->num_rows() > 0) {
        result->slice(0, 5)->print();
    }
}

/**
 * Test indexed vector join on CPU (ANN).
 */
TEST(ListTypeSupport, IndexedVectorJoin_CPU) {
    std::string path = vsds_test_path();
    std::cout << "=== Test: Indexed Vector Join (CPU) ===" << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto ctx          = db->get_context();
    auto device       = maximus::DeviceType::CPU;

    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, device);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, device);
    
    auto training_data = db->get_table("reviews");
    auto index = maximus::faiss::FaissIndex::build(ctx, training_data, "rv_embedding",
                                          "Flat", maximus::VectorDistanceMetric::L2, false);

    maximus::vsds::QueryParameters params;
    params.k = 5;
    
    auto qp = maximus::vsds::ann_reviews(db, index, device, params);
    
    std::cout << "Query Plan:\n" << qp->to_string() << std::endl;
    
    maximus::TablePtr result;
    ASSERT_NO_THROW({
        result = db->query(qp);
    });
    
    ASSERT_TRUE(result != nullptr);
    std::cout << "PASS: Indexed vector join returned " << result->num_rows() << " rows" << std::endl;
}

#ifdef MAXIMUS_WITH_CUDA

/**
 * Test exhaustive vector join on GPU (ENN).
 * May fail due to cuDF LIST interop issues.
 */
TEST(ListTypeSupport, DISABLED_ExhaustiveVectorJoin_GPU) {
    std::string path = vsds_test_path();
    std::cout << "=== Test: Exhaustive Vector Join (GPU) ===" << std::endl;

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue);
    auto device       = maximus::DeviceType::GPU;

    // Load on CPU first
    db->load_table("reviews", maximus::vsds::schema("reviews"), {}, maximus::DeviceType::CPU);
    db->load_table("reviews_queries", maximus::vsds::schema("reviews_queries"), {}, maximus::DeviceType::CPU);

    maximus::vsds::QueryParameters params;
    params.k = 5;
    
    auto qp = maximus::vsds::enn_reviews(db, device, params);
    
    std::cout << "Query Plan:\n" << qp->to_string() << std::endl;
    
    try {
        auto result = db->query(qp);
        if (result) {
            std::cout << "PASS: GPU Exhaustive vector join returned " << result->num_rows() << " rows" << std::endl;
        } else {
            std::cout << "FAIL: GPU Exhaustive vector join returned null" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "FAIL: GPU Exhaustive vector join threw: " << e.what() << std::endl;
        FAIL() << "GPU ENN failed: " << e.what();
    }
}

#endif  // MAXIMUS_WITH_CUDA

}  // namespace test
