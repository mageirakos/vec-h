#include <arrow/api.h>

#include <algorithm>
#include <maximus/frontend/expressions.hpp>
#include <maximus/frontend/query_plan_api.hpp>
#include <maximus/indexes/faiss/faiss_index.hpp>
#include <maximus/tpch/tpch_queries.hpp>
#include <maximus/types/expression.hpp>
#include <maximus/utils/utils.hpp>
#include <maximus/vsds/vsds_queries.hpp>
#include <sstream>
#include <stdexcept>

#ifdef MAXIMUS_WITH_CUDA
#include <faiss/gpu/GpuIndexCagra.h>
#endif

#include <faiss/IndexHNSW.h>

namespace maximus::vsds {

// NOTE: This file provides VSDS benchmark query implementations.
// - It reuses TPCH schemas for standard TPCH tables
// - It adds `reviews`, `reviews_queries`, `images`, `images_queries` tables

namespace cp = ::arrow::compute;

std::vector<std::string> table_names() {
    auto names = maximus::tpch::table_names();
    names.push_back("reviews");
    names.push_back("reviews_queries");
    names.push_back("images");
    names.push_back("images_queries");
    return names;
}

std::vector<std::pair<std::string, std::string>> available_queries() {
    return {
        // 'basic' VS queries
        // enn
        {"enn_reviews", "Exact Nearest Neighbor Search on Reviews (No Index)"},
        {"enn_reviews_project_distance", "Exact Nearest Neighbor Search on Reviews (No Index) - Project Distance Variant"},
        {"enn_images", "Exact Nearest Neighbor Search on Images (No Index)"},
        {"enn_images_project_distance", "Exact Nearest Neighbor Search on Images (No Index) - Project Distance Variant"},
        // enn & ann
        {"ann_reviews", "Approximate Nearest Neighbor Search on Reviews (Requires Index)"},
        {"ann_images", "Approximate Nearest Neighbor Search on Images (Requires Index)"},
        {"ann_reviews_full", "Approximate Nearest Neighbor Search on Reviews (Requires Index) - Full Columns (title, text on both sides)"},
        {"ann_images_full", "Approximate Nearest Neighbor Search on Images (Requires Index) - Full Columns (variant, image_url on both sides)"},
        // ranged
        {"ann_ranged_reviews", "Approximate Ranged Search on Reviews (Requires Index & Radius Parameter)"},
        {"ann_ranged_images", "Approximate Ranged Search on Images (Requires Index & Radius Parameter)"},
        // filtered
        {"pre_reviews", "Pre-filtering: Filter reviews (rating >= 4.0) then Vector Search"},
        {"pre_reviews_hybrid", "[FAILS until cuDF supports int64]Pre-filtering (Hybrid): Filter reviews (rating >= 4.0) on CPU then Vector Search on GPU"},
        {"pre_images", "Pre-filtering: Filter images (variant='MAIN') then Vector Search"},
        {"pre_images_hybrid", "Pre-filtering (Hybrid): Filter images (variant='MAIN') on CPU then Vector Search on GPU"},
        {"pre_reviews_partitioned", "Pre-filtering (Partitioned): Per-group filter on rv_rating then Vector Search"},
        {"post_reviews", "Post-filtering: Vector Search then Filter reviews (rating >= 4.0)"},
        {"post_reviews_hybrid", "Post-filtering (Hybrid): Vector Search on GPU then Filter reviews (rating >= 4.0) on CPU"},
        {"post_reviews_partitioned", "Post-filtering (Partitioned): Vector Search then Filter reviews (rating >= 4.0) inside each partition"},
        {"post_reviews_filter_partitioned", "Post-filtering (Filter-Partitioned): Single ANN search, scatter by filter group, per-group rv_rating filter"},
        {"post_images", "Post-filtering: Vector Search then Filter images (variant='MAIN')"},
        {"post_images_hybrid", "Post-filtering (Hybrid): Data on CPU, Index on GPU : Vector Search on GPU then Filter images (variant='MAIN') on CPU"},
        {"post_images_partitioned",  "Post-filtering (Partitioned): Vector Search then Filter images (variant='MAIN') inside each partition"},
        {"prejoin_reviews", "Pre-join: Filter Part table -> Join Reviews -> Exhaustive Search"},
        {"postjoin_reviews", "Post-join: Vector Search -> Join Part table -> Filter"},
        {"pre_bitmap_ann_reviews", "Bitmap-Filtering (cur. CPU-only): Project 'filter' -> Indexed Vector Search (Reviews)"},
        {"pre_bitmap_ann_images", "Bitmap-Filtering (cur. CPU-only): Project 'filter' -> Indexed Vector Search (Images)"},
        {"inline_expr_ann_reviews", "Expr-Filtering (cur. CPU-only): Inline Arrow Expression -> Indexed Vector Search (Reviews)"},
        {"inline_expr_ann_images", "Expr-Filtering (cur. CPU-only): Inline Arrow Expression -> Indexed Vector Search (Images)"},
        // VSDS queries enn & ann
        {"q1_start", "VSDS Q1: TPCH Q1 w/ VS@START (semantic clustering & frequency classification)"}, // TODO: figure out Take() speedup
        {"q2_start", "VSDS Q2: TPCH Q2 w/ VS@START (kNN semantic filter)"},
        {"q10_mid", "VSDS Q10: TPCH Q10 w/ VS@MID"},
        {"q11_end", "VSDS Q11: TPCH Q11 w/ VS@END (kNN duplicate detection)"}, // TODO: hypothesis : GPU duplicates is not deterministic vs CPU --> which would affect recall.
        {"q13_mid", "VSDS Q13: TPCH Q13 w/ VS@MID"},
        {"q15_end", "VSDS Q15: TPCH Q15 w/ VS@END"}, // TODO: look into Take() it leads to OOM
        {"q16_start", "VSDS Q16: TPCH Q16 w/ VS@START"},
        {"q18_mid", "VSDS Q18: TPCH Q18 w/ VS@MID"},
        {"q19_start", "VSDS Q19: TPCH Q19 w/ VS@START"},
    };
}

std::shared_ptr<Schema> schema(const std::string& table_name) {
    // Delegate TPCH tables to the TPCH helper
    auto tpch_tables = maximus::tpch::table_names();
    if (std::find(tpch_tables.begin(), tpch_tables.end(), table_name) != tpch_tables.end()) {
        return maximus::tpch::schema(table_name);
    }
    int reviews_dim = 1024;
    int images_dim = 1152;

    if (table_name == "reviews" or table_name == "reviews_queries") {
        auto suffix = (table_name == "reviews") ? std::string("") : std::string("_queries");
        auto embedding_type = (table_name == "reviews") ? arrow::large_list(arrow::float32()) : embeddings_list(arrow::float32(), reviews_dim);

        auto fields = {arrow::field("rv_rating" + suffix, arrow::float32(), true),
                       arrow::field("rv_helpful_vote" + suffix, arrow::int32(), false),
                       arrow::field("rv_title" + suffix, arrow::utf8(), true),
                       arrow::field("rv_text" + suffix, arrow::utf8(), true),
                       arrow::field("rv_embedding" + suffix, embedding_type),
                       arrow::field("rv_partkey" + suffix, arrow::int64(), false),
                       arrow::field("rv_custkey" + suffix, arrow::int64(), false),
                       arrow::field("rv_reviewkey" + suffix, arrow::int64(), false)};
        return std::make_shared<Schema>(fields);
    }

    // NOTE: we could at some point try float16 embeddings & arrow::large_list
    // but can't use large list in GPU when converting from cudf
    if (table_name == "images" or table_name == "images_queries") {
        auto suffix = (table_name == "images") ? std::string("") : std::string("_queries");
        
        // large list if we want > 2B elements on GPU (w/ cuDF tables):
        // auto embedding_type = (table_name == "images") ? arrow::large_list(arrow::float32()) : embeddings_list(arrow::float32(), images_dim);
        // NOTE: keep normal embedding type for images (to be able to do prefilter on GPU)
        auto embedding_type  = embeddings_list(arrow::float32(), images_dim);

        auto fields = {arrow::field("i_image_url" + suffix, arrow::utf8(), true),
                       arrow::field("i_variant" + suffix, arrow::utf8(), true),
                       arrow::field("i_embedding" + suffix, embedding_type),
                       arrow::field("i_imagekey" + suffix, arrow::int64(), false),
                       arrow::field("i_partkey" + suffix, arrow::int64(), false)};
        return std::make_shared<Schema>(fields);
    }

    throw std::runtime_error("The schema for given VSDS table not known.");
}

std::vector<std::shared_ptr<Schema>> schemas() {
    auto s = maximus::tpch::schemas();
    s.push_back(schema("reviews"));
    s.push_back(schema("reviews_queries"));
    s.push_back(schema("images"));
    s.push_back(schema("images_queries"));
    return s;
}

// ============================================================================
// Query Parameter Parsing
// ============================================================================

QueryParameters parse_query_parameters(const std::string& params_str) {
    QueryParameters params;

    if (params_str.empty()) {
        return params;
    }

    std::istringstream stream(params_str);
    std::string token;

    while (std::getline(stream, token, ',')) {
        auto eq_pos = token.find('=');
        if (eq_pos == std::string::npos) {
            continue;  // Skip malformed tokens
        }

        std::string key   = token.substr(0, eq_pos);
        std::string value = token.substr(eq_pos + 1);

        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        // Parse known parameters
        if (key == "faiss_index") {
            params.faiss_index = value;
        } else if (key == "hnsw_efsearch") {
            params.hnsw_efsearch = std::stoi(value);
        } else if (key == "cagra_itopksize") {
            params.cagra_itopksize = std::stoi(value);
        } else if (key == "cagra_searchwidth") {
            params.cagra_searchwidth = std::stoi(value);
        } else if (key == "ivf_nprobe") {
            params.ivf_nprobe = std::stoi(value);
        } else if (key == "postfilter_ksearch") {
            params.postfilter_ksearch = std::stoi(value);
        } else if (key == "k") {
            params.k = std::stoi(value);
        } else if (key == "radius") {
            params.radius = std::stof(value);
        } else if (key == "query_start") {
            params.query_start = std::stoll(value);
        } else if (key == "query_count") {
            params.query_count = std::stoll(value);
        } else if (key == "metric") {
            if (value == "IP") {
                params.metric = VectorDistanceMetric::INNER_PRODUCT;
            } else if (value == "L2") {
                params.metric = VectorDistanceMetric::L2;
            } else {
                throw std::runtime_error("Unknown metric: '" + value + "'. Supported: L2, IP");
            }
        } else if (key == "use_cuvs") {
            params.use_cuvs = (value != "0" && value != "false");
        } else if (key == "use_post") {
            params.use_post = (value != "0" && value != "false");
        } else if (key == "use_limit_per_group") {
            params.use_limit_per_group = (value != "0" && value != "false");
        } else if (key == "incr_step") {
            params.incr_step = std::stoi(value);
        } else if (key == "index_data_on_gpu" || key == "cagra_data_on_gpu") {
            if (value == "auto" || value == "-1") params.index_data_on_gpu = -1;
            else params.index_data_on_gpu = (value != "0" && value != "false") ? 1 : 0;
        } else if (key == "cagra_cache_graph") {
            params.cagra_cache_graph = (value != "0" && value != "false") ? 1 : 0;
        } else if (key == "filter_partkey") {
            params.filter_partkey = std::stoll(value);
        }
        // Unknown parameters are silently ignored
    }

    return params;
}

// ============================================================================
// Query Index Requirements
// ============================================================================

std::vector<std::string> get_query_index_requirements(const std::string& q) {
    // Queries requiring both reviews and images indexes
    if (q == "q19" || q == "q19_start") {
        return {"reviews.rv_embedding", "images.i_embedding"};
    }
    // Queries requiring reviews index
    if (q == "ann_reviews" || q == "ann_reviews_full" || q == "ann" || q == "ann_ranged_reviews" || q == "post_reviews" ||
        q == "post" || q == "postjoin_reviews" || q == "postjoin" ||
        q == "pre_bitmap_ann_reviews" || q == "inline_expr_ann_reviews" ||
        q == "post_reviews_partitioned" || q == "post_reviews_filter_partitioned" || q == "post_reviews_hybrid" ||
        q == "q10" || q == "q10_mid" || q == "q16" || q == "q16_start" ||
        q == "q15" || q == "q15_end" || q == "q13" || q == "q13_mid") {
        return {"reviews.rv_embedding"};
    }
    // Queries requiring images index
    if (q == "ann_images" || q == "ann_images_full" || q == "ann_ranged_images" || q == "post_images" ||
        q == "post_images_hybrid" || q == "post_images_partitioned" ||
        q == "pre_bitmap_ann_images" || q == "inline_expr_ann_images" ||
        q == "q18" || q == "q18_mid" || q == "q11" || q == "q11_end" ||
        q == "q2" || q == "q2_start") {
        return {"images.i_embedding"};
    }
    // Queries explicitly requiring no index (enn, etc.) or unknown
    return {};
}

// ============================================================================
// Search Parameters Factory
// ============================================================================

std::shared_ptr<IndexParameters> make_search_parameters(const std::string& faiss_index,
                                                         const QueryParameters& params,
                                                         DeviceType device) {
#if defined(MAXIMUS_WITH_CUDA) && defined(MAXIMUS_WITH_FAISS_GPUCUVS)
    // Cagra: must check before IVF since "GPU,Cagra,64,32,IVF_PQ" contains "IVF"
    if (maximus::starts_with(faiss_index, "GPU,Cagra")) {
        if (device == DeviceType::CPU) {
            // Index will be moved to CPU as IndexHNSWCagra — must use HNSW params
            auto p      = std::make_unique<::faiss::SearchParametersHNSW>();
            p->efSearch = params.hnsw_efsearch;
            std::cout << "[VSDS] GPU,Cagra on CPU (HNSWCagra): efSearch="
                      << params.hnsw_efsearch << std::endl;
            return std::make_shared<maximus::faiss::FaissSearchParameters>(std::move(p));
        } else {
            auto p          = std::make_unique<::faiss::gpu::SearchParametersCagra>();
            p->itopk_size   = params.cagra_itopksize;
            p->search_width = params.cagra_searchwidth;
            std::cout << "[VSDS] GPU,Cagra GPU: itopk_size=" << params.cagra_itopksize
                      << " search_width=" << params.cagra_searchwidth << std::endl;
            return std::make_shared<maximus::faiss::FaissSearchParameters>(std::move(p));
        }
    }
#endif

    // HNSW: uses efSearch (matches both "HNSW..." and "GPU,HNSW...")
    if (faiss_index.find("HNSW") != std::string::npos) {
        auto p      = std::make_unique<::faiss::SearchParametersHNSW>();
        p->efSearch = params.hnsw_efsearch;
        std::cout << "[VSDS] Using HNSW search_params with efSearch=" << params.hnsw_efsearch
                  << std::endl;
        return std::make_shared<maximus::faiss::FaissSearchParameters>(std::move(p));
    }

    // IVF/IVFPQ (CPU or GPU): uses nprobe (matches both "IVF..." and "GPU,IVF...")
    if (faiss_index.find("IVF") != std::string::npos) {
        auto p    = std::make_unique<::faiss::SearchParametersIVF>();
        p->nprobe = params.ivf_nprobe;
        std::cout << "[VSDS] Using IVF search_params with nprobe=" << params.ivf_nprobe
                  << std::endl;
        return std::make_shared<maximus::faiss::FaissSearchParameters>(std::move(p));
    }

    // Flat or unknown: no params needed
    std::cout << "[VSDS] No index specific search parameters (Flat index or unknown type)." << std::endl;
    return nullptr;
}

// ============================================================================
// Query Range Helpers
// ============================================================================

// Get the number of queries for scatter-gather pattern.
// Simply returns query_count from params if provided (> 0), otherwise -1.
// The caller decides what to do when count is unknown (-1).
int64_t get_partition_count(int64_t query_count) {
    // If explicit count provided, use it
    if (query_count > 0) {
        return query_count;
    }
    // Otherwise unknown - caller should handle (skip partition pattern)
    return -1;
}

// ============================================================================
// Sliced Query Source Helper
// ============================================================================

// Helper: Create a query source with table slicing support for query range execution.
// Uses limit operator with offset for zero-copy slicing.

// TODO : Very easy performance optimization would be to avoid loading entire query table. 
//      - Just have a smaller query table with the ones you're actually using
std::shared_ptr<QueryNode> sliced_query_source(
    std::shared_ptr<Database>& db,
    const std::string& table_name,
    std::shared_ptr<Schema> table_schema,
    const std::vector<std::string>& columns,
    DeviceType device,
    int64_t query_start,
    int64_t query_count) {
    
    // Create base source
    auto source = table_source(db, table_name, table_schema, columns, device);
    
    // No slicing needed - use full table (query_count <= 0 means "all")
    if (query_count <= 0) {
        return source;
    }

    // Apply limit with offset for slicing
    // The limit operator will handle bounds checking at execution time
    // when the table is actually loaded
    std::cout << "[VSDS] Query slice: start=" << query_start
              << ", count=" << query_count << "\n";
    return limit(source, query_count, query_start, device);
}

// ============================================================================
// Per-Query Limit Helper (for post-filtering)
// ============================================================================

// Apply limit K per query after filtering.
// If num_queries == 1, just apply order_by + limit (no partition overhead).
// If num_queries > 1, use scatter -> order_by -> limit -> gather pattern.
// If num_queries <= 0 (unknown), skip the per-query limit and return the node as-is.
// If use_limit_per_group is true, use the streaming LimitPerGroup operator instead
// (assumes data is already grouped by key and sorted by distance within each group).
//
// NOTE: We always order by distance ASC before limiting to ensure we get the K NEAREST neighbors.
std::shared_ptr<QueryNode> apply_per_query_order_by_limit(
    std::shared_ptr<QueryNode>& node,
    const std::string& query_key_column,
    int64_t num_queries,
    int64_t limit_k,
    DeviceType device,
    VectorDistanceMetric metric = VectorDistanceMetric::L2,
    bool use_limit_per_group = false) {

    if (num_queries <= 0) {
        // Unknown query count - can't apply per-query limit at plan build time.
        //
        // WARNING: If the actual data contains multiple query IDs, the results will be WRONG!
        // Without knowing the exact query count, we cannot partition by query key.
        // The post-filter/post-join results will mix results from different queries.
        //
        // To fix this, specify query_count explicitly via params:
        //   e.g., "query_count=10" in the params string
        std::cerr << "[VSDS] WARNING: query_count not specified! Skipping per-query limit.\n"
                  << "       If running multiple queries, results may be INCORRECT.\n"
                  << "       Specify query_count=N in params to enable scatter-gather pattern.\n";
        throw std::runtime_error("Cannot apply per-query limit without known partitions. Please specify query_count in parameters.");
        return node;
    }

    SortOrder order = (metric == VectorDistanceMetric::INNER_PRODUCT) ? SortOrder::DESCENDING : SortOrder::ASCENDING;

    if (use_limit_per_group) {
        // New path: order_by + limit_per_group (avoids scatter + gather)
        // order_by is still needed because GPU operators may not preserve Faiss sort order
        if (num_queries == 1) {
            auto ordered = order_by(node, {SortKey("vs_distance", order)}, device);
            return limit(ordered, limit_k, 0, device);
        }
        // assumes data is already groupd by query_key_column and sorted by distance within each group (e.g., from FAISS)
        return limit_per_group(node, query_key_column, limit_k, device);
    }

    if (num_queries == 1) {
        // Single query: order by distance then limit
        auto ordered = order_by(node, {SortKey("vs_distance", order)}, device); // TODO: only needed because limit complains, but data already sorted from FAISS by distance
        return limit(ordered, limit_k, 0, device);
    }

    // Batch query: use scatter -> order_by -> limit -> gather
    // Order by distance to get K nearest neighbors per partition
    std::vector<SortKey> sort_keys = {SortKey("vs_distance", order)};
    return order_limit_per_partition(node, query_key_column,
                                      static_cast<int>(num_queries), limit_k, device, sort_keys);
}

// ============================================================================
// Helper: scatter -> filter -> order_by -> limit -> gather (per partition)
// Used by post_reviews_partitioned / post_images_partitioned
// ============================================================================

std::shared_ptr<QueryNode> apply_per_query_filter_order_by_limit(
    std::shared_ptr<QueryNode>& node,
    const std::string& query_key_column,
    int64_t num_queries,
    int64_t limit_k,
    std::shared_ptr<Expression> filter_expr,
    DeviceType device,
    VectorDistanceMetric metric = VectorDistanceMetric::L2) {

    if (num_queries <= 0) {
        std::cerr << "[VSDS] WARNING: query_count not specified! Skipping per-query filter+limit.\n"
                  << "       If running multiple queries, results may be INCORRECT.\n"
                  << "       Specify query_count=N in params to enable scatter-gather pattern.\n";
        return node;
    }

    SortOrder order = (metric == VectorDistanceMetric::INNER_PRODUCT) ? SortOrder::DESCENDING : SortOrder::ASCENDING;

    if (num_queries == 1) {
        // Single query: filter -> order_by -> limit (no partition overhead)
        auto filtered = filter(node, filter_expr, device);
        auto ordered  = order_by(filtered, {SortKey("vs_distance", order)}, device);
        return limit(ordered, limit_k, 0, device);
    }

    // Batch: scatter -> (filter -> order_by -> limit) per partition -> gather
    auto scattered = scatter(node, {query_key_column}, static_cast<int>(num_queries), device);

    std::vector<std::shared_ptr<QueryNode>> limited_partitions;
    for (int i = 0; i < num_queries; ++i) {
        auto filtered_i = filter(scattered, filter_expr, device);
        auto ordered_i  = order_by(filtered_i, {SortKey("vs_distance", order)}, device);
        auto limited_i  = limit(ordered_i, limit_k, 0, device);
        limited_partitions.push_back(limited_i);
    }

    return gather(limited_partitions, device);
}

// ============================================================================
//                              QUERIES
// ============================================================================

// ============================================================================
// ENN: Exhaustive Nearest Neighbor (Reviews)
// ============================================================================

std::shared_ptr<QueryPlan> enn_reviews(std::shared_ptr<Database>& db,
                                       DeviceType device,
                                       const QueryParameters& params) {
    // Create table sources with device (aligned with TPC-H pattern)
    auto data_source = table_source(db,
                                     "reviews",
                                     schema("reviews"),
                                     {"rv_reviewkey", "rv_embedding"},
                                     device);
    // Query table with slicing support for batch execution
    auto query_source = sliced_query_source(db,
                                            "reviews_queries",
                                            schema("reviews_queries"),
                                            {"rv_reviewkey_queries", "rv_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    // Create exhaustive vector join (ENN - no index)
    auto vs_node = exhaustive_vector_join(data_source,
                                            query_source,
                                            "rv_embedding",         // data vector column
                                            "rv_embedding_queries", // query vector column
                                            params.metric,
                                            params.k,        // K
                                            std::nullopt,    // no radius
                                            false,           // don't keep data vector
                                            false,           // don't keep query vector
                                            "vs_distance",      // distance column name
                                            params.vs_device);

    return query_plan(table_sink(vs_node));
}

// ============================================================================
// ENN: Exhaustive Nearest Neighbor (Reviews) - Project Distance Variant
// Uses ProjectDistance operator -> OrderBy -> Limit
// This computes the full Cartesian product of distances, then sorts and limits.
// ============================================================================

std::shared_ptr<QueryPlan> enn_reviews_project_distance(std::shared_ptr<Database>& db,
                                                        DeviceType device,
                                                        const QueryParameters& params) {
    // Create table sources
    auto data_source = table_source(db,
                                     "reviews",
                                     schema("reviews"),
                                     {"rv_reviewkey", "rv_embedding"},
                                     device);

    // Query table with slicing support
    auto query_source = sliced_query_source(db,
                                            "reviews_queries",
                                            schema("reviews_queries"),
                                            {"rv_reviewkey_queries", "rv_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    // 1. Project Distance: Compute all-pairs distances
    // Output: rv_reviewkey, rv_embedding, rv_reviewkey_queries, rv_embedding_queries, distance
    auto pd_node = vector_project_distance(data_source,
                                             query_source,
                                             "rv_embedding",
                                             "rv_embedding_queries",
                                             false, // Don't keep data vectors (optimization)
                                             false, // Don't keep query vectors (optimization)
                                             "vs_distance",
                                             params.vs_device);

    // 2. Apply per-query limit K (Order By + Limit)
    // Use query_count from params for partition pattern
    auto is_pre_sorted = true;
    int64_t num_queries = get_partition_count(params.query_count);
    auto limited = apply_per_query_order_by_limit(pd_node, "rv_reviewkey_queries", 
                                          num_queries, params.k, device, params.metric, params.use_limit_per_group);

    return query_plan(table_sink(limited));
}

// ============================================================================
// ENN: Exhaustive Nearest Neighbor (Images)
// ============================================================================

std::shared_ptr<QueryPlan> enn_images(std::shared_ptr<Database>& db,
                                      DeviceType device,
                                      const QueryParameters& params) {
    auto data_source = table_source(db,
                                     "images",
                                     schema("images"),
                                     {"i_imagekey", "i_embedding"},
                                     device);
    // Query table with slicing support for batch execution
    auto query_source = sliced_query_source(db,
                                            "images_queries",
                                            schema("images_queries"),
                                            {"i_imagekey_queries", "i_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    auto vs_node = exhaustive_vector_join(data_source,
                                            query_source,
                                            "i_embedding",
                                            "i_embedding_queries",
                                            params.metric,
                                            params.k,
                                            std::nullopt,
                                            false,
                                            false,
                                            "vs_distance",
                                            params.vs_device);

    return query_plan(table_sink(vs_node));
}

// ============================================================================
// ENN: Exhaustive Nearest Neighbor (Images) - Project Distance Variant
// ============================================================================

std::shared_ptr<QueryPlan> enn_images_project_distance(std::shared_ptr<Database>& db,
                                                       DeviceType device,
                                                       const QueryParameters& params) {
    auto data_source = table_source(db,
                                     "images",
                                     schema("images"),
                                     {"i_imagekey", "i_embedding"},
                                     device);

    // Query table with slicing support
    auto query_source = sliced_query_source(db,
                                            "images_queries",
                                            schema("images_queries"),
                                            {"i_imagekey_queries", "i_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    // 1. Project Distance
    auto pd_node = vector_project_distance(data_source,
                                             query_source,
                                             "i_embedding",
                                             "i_embedding_queries",
                                             false,
                                             false,
                                             "vs_distance",
                                             params.vs_device);
    
    // 2. Apply per-query limit K (Order By + Limit)
    // Use query_count from params for partition pattern
    int64_t num_queries = get_partition_count(params.query_count);
    auto limited = apply_per_query_order_by_limit(pd_node, "i_imagekey_queries", 
                                          num_queries, params.k, device, params.metric, params.use_limit_per_group);

    return query_plan(table_sink(limited));
}

// ============================================================================
// ANN: Approximate Nearest Neighbor (Reviews)
// ============================================================================

std::shared_ptr<QueryPlan> ann_reviews(std::shared_ptr<Database>& db,
                                       IndexPtr index,
                                       DeviceType device,
                                       const QueryParameters& params) {
    if (!index) {
        throw std::runtime_error("VSDS ann_reviews query requires a pre-built index. "
                                 "Build the index before calling ann_reviews().");
    }

    std::cout << "Device = " << device_type_to_string(device) << std::endl;
    
    auto data_source = table_source(db,
                                     "reviews",
                                     schema("reviews"),
                                     {"rv_reviewkey"}, // we already 'built' an index on the embeddings, it owns the data
                                     device,
                                     /*nocopy_variant=*/true);
    // Query table with slicing support for batch execution
    auto qs_base = table_source(db,
                                 "reviews_queries",
                                 schema("reviews_queries"),
                                 {"rv_reviewkey_queries", "rv_embedding_queries"},
                                 device,
                                 /*nocopy_variant=*/true);
    auto query_source = (params.query_count <= 0)
        ? qs_base
        : limit(qs_base, params.query_count, params.query_start, device);

    // Create search parameters based on index type
    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

    // Create indexed vector join (ANN - with pre-built index)
    auto vs_node = indexed_vector_join(data_source,
                                         query_source,
                                         "", // index owns the data
                                         "rv_embedding_queries",
                                         index,
                                         params.k,
                                         std::nullopt,
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         params.vs_device);

    return query_plan(table_sink(vs_node));
}

// ============================================================================
// ANN: Approximate Nearest Neighbor (Images)
// ============================================================================

std::shared_ptr<QueryPlan> ann_images(std::shared_ptr<Database>& db,
                                      IndexPtr index,
                                      DeviceType device,
                                      const QueryParameters& params) {
    if (!index) {
        throw std::runtime_error("VSDS ann_images query requires a pre-built index. "
                                 "Build the index before calling ann_images().");
    }

    std::cout << "Device = " << device_type_to_string(device) << std::endl;
    
    auto data_source = table_source(db,
                                     "images",
                                     schema("images"),
                                     {"i_imagekey"},
                                     device);
    // Query table with slicing support for batch execution
    auto query_source = sliced_query_source(db,
                                            "images_queries",
                                            schema("images_queries"),
                                            {"i_imagekey_queries", "i_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

    auto vs_node = indexed_vector_join(data_source,
                                         query_source,
                                         "",
                                         "i_embedding_queries",
                                         index,
                                         params.k,
                                         std::nullopt,
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         params.vs_device);

    return query_plan(table_sink(vs_node));
}

// ============================================================================
// ANN: Approximate Nearest Neighbor (Reviews) - Full Columns
// Returns rv_title, rv_text from data side and rv_title_queries, rv_text_queries from query side
// ============================================================================

std::shared_ptr<QueryPlan> ann_reviews_full(std::shared_ptr<Database>& db,
                                            IndexPtr index,
                                            DeviceType device,
                                            const QueryParameters& params) {
    if (!index) {
        throw std::runtime_error("VSDS ann_reviews_full query requires a pre-built index. "
                                 "Build the index before calling ann_reviews_full().");
    }

    std::cout << "Device = " << device_type_to_string(device) << std::endl;

    auto data_source = table_source(db,
                                     "reviews",
                                     schema("reviews"),
                                     {"rv_reviewkey", "rv_title", "rv_text"}, // index owns embeddings
                                     device);
    // Query table with slicing support for batch execution
    auto query_source = sliced_query_source(db,
                                            "reviews_queries",
                                            schema("reviews_queries"),
                                            {"rv_reviewkey_queries", "rv_embedding_queries",
                                             "rv_title_queries", "rv_text_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    // Create search parameters based on index type
    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

    // Create indexed vector join (ANN - with pre-built index)
    auto vs_node = indexed_vector_join(data_source,
                                         query_source,
                                         "", // index owns the data
                                         "rv_embedding_queries",
                                         index,
                                         params.k,
                                         std::nullopt,
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         params.vs_device);

    return query_plan(table_sink(vs_node));
}

// ============================================================================
// ANN: Approximate Nearest Neighbor (Images) - Full Columns
// Returns i_variant, i_image_url from data side and i_variant_queries, i_image_url_queries from query side
// ============================================================================

std::shared_ptr<QueryPlan> ann_images_full(std::shared_ptr<Database>& db,
                                           IndexPtr index,
                                           DeviceType device,
                                           const QueryParameters& params) {
    if (!index) {
        throw std::runtime_error("VSDS ann_images_full query requires a pre-built index. "
                                 "Build the index before calling ann_images_full().");
    }

    std::cout << "Device = " << device_type_to_string(device) << std::endl;

    auto data_source = table_source(db,
                                     "images",
                                     schema("images"),
                                     {"i_imagekey", "i_variant", "i_image_url"},
                                     device);
    // Query table with slicing support for batch execution
    auto query_source = sliced_query_source(db,
                                            "images_queries",
                                            schema("images_queries"),
                                            {"i_imagekey_queries", "i_embedding_queries",
                                             "i_variant_queries", "i_image_url_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

    auto vs_node = indexed_vector_join(data_source,
                                         query_source,
                                         "",
                                         "i_embedding_queries",
                                         index,
                                         params.k,
                                         std::nullopt,
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         params.vs_device);

    return query_plan(table_sink(vs_node));
}

// ============================================================================
// ANN: Approximate Ranged Search (Reviews)
// ============================================================================

std::shared_ptr<QueryPlan> ann_ranged_reviews(std::shared_ptr<Database>& db,
                                      IndexPtr index,
                                      DeviceType device,
                                      const QueryParameters& params) {
    if (!index) {
        throw std::runtime_error("VSDS ann_ranged_reviews query requires a pre-built index. "
                                 "Build the index before calling ann_ranged_reviews().");
    }

    std::cout << "Device = " << device_type_to_string(device) << std::endl;
    
    auto data_source = table_source(db,
                                     "reviews",
                                     schema("reviews"),
                                     {"rv_reviewkey"},
                                     device);
    // Query table with slicing support for batch execution
    auto query_source = sliced_query_source(db,
                                            "reviews_queries",
                                            schema("reviews_queries"),
                                            {"rv_reviewkey_queries", "rv_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);


    // NOTE: mapping pgvector to faiss
    //  - L2: if 0.25 radius in pgvector (L2 distance) --> radius=0.501 in FAISS (squared L2 )
    //  - IP: if -0.9 radius in pgvector (inner product) --> radius=-0.9 in FAISS (change sign)
    auto vs_node = indexed_vector_join(data_source,
                                         query_source,
                                         "",
                                         "rv_embedding_queries",
                                         index,
                                         std::nullopt,
                                         params.radius, // radius
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         params.vs_device);


    int64_t num_queries = get_partition_count(params.query_count);
    auto limited = apply_per_query_order_by_limit(vs_node, "rv_reviewkey_queries", 
                                          num_queries, params.k, device, params.metric, params.use_limit_per_group);

    return query_plan(table_sink(limited));
}

std::shared_ptr<QueryPlan> ann_ranged_images(std::shared_ptr<Database>& db,
                                      IndexPtr index,
                                      DeviceType device,
                                      const QueryParameters& params) {
    if (!index) {
        throw std::runtime_error("VSDS ann_ranged_images query requires a pre-built index. "
                                 "Build the index before calling ann_ranged_images().");
    }

    std::cout << "Device = " << device_type_to_string(device) << std::endl;
    
    auto data_source = table_source(db,
                                     "images",
                                     schema("images"),
                                     {"i_imagekey"},
                                     device);
    // Query table with slicing support for batch execution
    auto query_source = sliced_query_source(db,
                                            "images_queries",
                                            schema("images_queries"),
                                            {"i_imagekey_queries", "i_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);


    // NOTE: mapping pgvector to faiss
    //  - L2: if 0.25 radius in pgvector (L2 distance) --> radius=0.501 in FAISS (squared L2 )
    //  - IP: if -0.9 radius in pgvector (inner product) --> radius=-0.9 in FAISS (change sign)
    auto vs_node = indexed_vector_join(data_source,
                                         query_source,
                                         "",
                                         "i_embedding_queries",
                                         index,
                                         std::nullopt,
                                         params.radius, // radius
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         params.vs_device);


    int64_t num_queries = get_partition_count(params.query_count);
    auto limited = apply_per_query_order_by_limit(vs_node, "i_imagekey_queries", 
                                          num_queries, params.k, device, params.metric, params.use_limit_per_group);

    return query_plan(table_sink(limited));
}


// ============================================================================
// Pre-filtering: filter then vector search (Reviews)
// ============================================================================

std::shared_ptr<QueryPlan> pre_reviews(std::shared_ptr<Database>& db,
                                       DeviceType device,
                                       const QueryParameters& params) {
    
    // 1. Load source table and vector search queries
    // Load full reviews table for filtering
    auto data_source = table_source(db,
                                     "reviews",
                                     schema("reviews"),
                                     {"rv_reviewkey", "rv_embedding", "rv_rating"},
                                     device);
    
    // Query table with slicing support for batch execution
    auto vs_query_source = sliced_query_source(db,
                                            "reviews_queries",
                                            schema("reviews_queries"),
                                            {"rv_reviewkey_queries", "rv_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    // 2. Filter: rv_rating >= 4.0 (only good reviews)
    auto filter_expr = expr(arrow_expr(cp::field_ref("rv_rating"), ">=", float32_literal(4.0f)));
    auto filtered_data = filter(data_source, filter_expr, device);
    
    // 3. Perform exhaustive vector join
    auto vs_node = exhaustive_vector_join(filtered_data,
                                            vs_query_source,
                                            "rv_embedding",
                                            "rv_embedding_queries",
                                            params.metric,
                                            params.k,
                                            std::nullopt,
                                            false,
                                            false,
                                            "vs_distance",
                                            params.vs_device);

    return query_plan(table_sink(vs_node));
}

std::shared_ptr<QueryPlan> pre_reviews_hybrid(std::shared_ptr<Database>& db,
                                      const QueryParameters& params) {
    
    auto on_CPU = maximus::DeviceType::CPU;
    auto on_GPU = maximus::DeviceType::GPU;

    // Data on CPU
    auto data_source = table_source(db,
                                     "reviews",
                                     schema("reviews"),
                                     {"rv_reviewkey", "rv_embedding", "rv_rating"},
                                     on_CPU);
    
    auto vs_query_source = sliced_query_source(db,
                                            "reviews_queries",
                                            schema("reviews_queries"),
                                            {"rv_reviewkey_queries", "rv_embedding_queries"},
                                            on_CPU,
                                            params.query_start,
                                            params.query_count);
    
    // Filter on CPU
    auto filter_expr = expr(arrow_expr(cp::field_ref("rv_rating"), ">=", float32_literal(4.0f)));
    auto filtered_data = filter(data_source, filter_expr, on_CPU);
    
    // VS on GPU
    std::cout << "[WARNING] - Expected to crash now unless (a) cuDF adds FixedSizeList or (b) int64 support " << std::endl;
    auto vs_node = exhaustive_vector_join(filtered_data,
                                            vs_query_source,
                                            "rv_embedding",
                                            "rv_embedding_queries",
                                            params.metric,
                                            params.k,
                                            std::nullopt,
                                            false,
                                            false,
                                            "vs_distance",
                                            on_GPU);

return query_plan(table_sink(vs_node));
}


std::shared_ptr<QueryPlan> pre_reviews_partitioned(std::shared_ptr<Database>& db,
                                                    DeviceType device,
                                                    const QueryParameters& params) {
    int64_t num_queries = get_partition_count(params.query_count);
    if (num_queries <= 0) {
        throw std::runtime_error("pre_reviews_partitioned requires explicit query_count > 0");
    }
    if (num_queries == 1) {
        return pre_reviews(db, device, params);
    }

    // Hardcoded unique rv_rating filter thresholds
    std::vector<float> filter_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int num_groups = std::min(static_cast<int>(filter_values.size()),
                              static_cast<int>(num_queries));

    std::vector<std::shared_ptr<QueryNode>> partitions;
    int64_t queries_assigned = 0;

    for (int g = 0; g < num_groups; ++g) {
        // Contiguous block assignment (distribute remainder to earlier groups)
        int64_t block_size = num_queries / num_groups
                           + (g < (num_queries % num_groups) ? 1 : 0);
        int64_t block_start = queries_assigned;
        queries_assigned += block_size;

        // Independent table source (shared underlying Arrow data, own batch reader)
        auto data_source = table_source(db, "reviews", schema("reviews"),
                                         {"rv_reviewkey", "rv_embedding", "rv_rating"}, device);

        // Filter with this group's threshold
        auto filter_expr = expr(arrow_expr(
            cp::field_ref("rv_rating"), ">=", float32_literal(filter_values[g])));
        auto filtered = filter(data_source, filter_expr, device);

        // Query vectors for this contiguous block
        auto query_source = sliced_query_source(
            db, "reviews_queries", schema("reviews_queries"),
            {"rv_reviewkey_queries", "rv_embedding_queries"},
            device, params.query_start + block_start, block_size);

        // Exhaustive vector search for this block
        auto vs = exhaustive_vector_join(filtered, query_source,
                                          "rv_embedding", "rv_embedding_queries",
                                          params.metric, params.k,
                                          std::nullopt, false, false,
                                          "vs_distance", params.vs_device);

        // Per-query order+limit within this block
        auto limited = apply_per_query_order_by_limit(
            vs, "rv_reviewkey_queries", block_size, params.k,
            device, params.metric, params.use_limit_per_group);

        partitions.push_back(limited);
    }

    // Gather all group results
    return query_plan(table_sink(gather(partitions, device)));
}

// ============================================================================
// Image filter helpers — selectivity control via filter_partkey parameter
// ============================================================================

// Build image filter expression: i_partkey < threshold (when set) or i_variant == "MAIN" (default)
static std::shared_ptr<Expression> make_images_filter(const QueryParameters& params) {
    if (params.filter_partkey > 0) {
        return expr(arrow_expr(cp::field_ref("i_partkey"), "<",
                               int64_literal(params.filter_partkey)));
    }
    return expr(arrow_expr(cp::field_ref("i_variant"), "==", string_literal("MAIN")));
}

// Column list for image data source — includes i_partkey when using partkey filter, i_variant otherwise
static std::vector<std::string> images_data_columns(const QueryParameters& params) {
    if (params.filter_partkey > 0) {
        return {"i_imagekey", "i_embedding", "i_partkey"};
    }
    return {"i_imagekey", "i_embedding", "i_variant"};
}

// ============================================================================
// Pre-filtering: filter then vector search (Images)
// ============================================================================

std::shared_ptr<QueryPlan> pre_images(std::shared_ptr<Database>& db,
                                      DeviceType device,
                                      const QueryParameters& params) {
    
    // 1. Load source table and vector search queries
    auto data_source = table_source(db,
                                     "images",
                                     schema("images"),
                                     images_data_columns(params),
                                     device);

    // Query table with slicing support for batch execution
    auto vs_query_source = sliced_query_source(db,
                                            "images_queries",
                                            schema("images_queries"),
                                            {"i_imagekey_queries", "i_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    // 2. Apply Filter: i_partkey < threshold (if set) or i_variant == "MAIN" (default)
    auto filter_expr = make_images_filter(params);
    auto filtered_data = filter(data_source, filter_expr, device);

    // 3. Perform exhaustive vector join
    auto vs_node = exhaustive_vector_join(filtered_data,
                                            vs_query_source,
                                            "i_embedding",
                                            "i_embedding_queries",
                                            params.metric,
                                            params.k,
                                            std::nullopt,
                                            false,
                                            false,
                                            "vs_distance",
                                            params.vs_device);

    return query_plan(table_sink(vs_node));
}


std::shared_ptr<QueryPlan> pre_images_hybrid(std::shared_ptr<Database>& db,
                                      const QueryParameters& params) {

    auto on_CPU = maximus::DeviceType::CPU;
    auto on_GPU = maximus::DeviceType::GPU;

    // Data on CPU
    auto data_source = table_source(db,
                                     "images",
                                     schema("images"),
                                     images_data_columns(params),
                                     on_CPU);

    auto vs_query_source = sliced_query_source(db,
                                            "images_queries",
                                            schema("images_queries"),
                                            {"i_imagekey_queries", "i_embedding_queries"},
                                            on_CPU,
                                            params.query_start,
                                            params.query_count);

    // Filter on CPU
    auto filter_expr = make_images_filter(params);
    auto filtered_data = filter(data_source, filter_expr, on_CPU);
    
    // VS on the GPU
    auto vs_node = exhaustive_vector_join(filtered_data,
                                            vs_query_source,
                                            "i_embedding",
                                            "i_embedding_queries",
                                            params.metric,
                                            params.k,
                                            std::nullopt,
                                            false,
                                            false,
                                            "vs_distance",
                                            on_GPU);

    return query_plan(table_sink(vs_node));
}

// ============================================================================
// Post-filtering: vector search then filter (Reviews)
// ANN search with larger k (postfilter_ksearch) and then filter
// ============================================================================

// NOTE: When using batching through PARTITION BY / UNION ALL:
// - Reliance on query_count:
//     The number of physical partitions (and thus limit nodes) is fixed 
//     at plan creation time using params.query_count
// - Edge Case: If query_count is unknown/zero, 
//     the code skips the partition step entirely (warning printed), 
//     meaning you get a "global" limit or no per-query limit, 
//     leading to incorrect results.
// - Query keys do need to be unique

std::shared_ptr<QueryPlan> post_reviews(std::shared_ptr<Database>& db,
                                        IndexPtr index,
                                        DeviceType device,
                                        const QueryParameters& params) {
    if (!index) {
        throw std::runtime_error("VSDS post_reviews query requires a pre-built index.");
    }
    
    // 1. Load source table and vector search queries
    // Include rating column for post-filtering
    auto data_source = table_source(db,
                                     "reviews",
                                     schema("reviews"),
                                     {"rv_reviewkey", "rv_rating"},
                                     device);
    // Query table with slicing support for batch execution
    auto vs_query_source = sliced_query_source(db,
                                            "reviews_queries",
                                            schema("reviews_queries"),
                                            {"rv_reviewkey_queries", "rv_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

    // 2. Perform indexed vector join
    // Use postfilter_ksearch (larger k) to ensure we get enough results after filtering
    auto vs_node = indexed_vector_join(data_source,
                                         vs_query_source,
                                         "",
                                         "rv_embedding_queries",
                                         index,
                                         params.postfilter_ksearch,
                                         std::nullopt,
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         params.vs_device);

    // 3. Post-filter: rv_rating >= 4.0
    auto filter_expr = expr(arrow_expr(cp::field_ref("rv_rating"), ">=", float32_literal(4.0f)));
    auto filtered = filter(vs_node, filter_expr, device);
    
    // 4. Apply per-query limit K
    // Use query_count from params for partition pattern
    int64_t num_queries = get_partition_count(params.query_count);
    auto limited = apply_per_query_order_by_limit(filtered, "rv_reviewkey_queries", 
                                          num_queries, params.k, device, params.metric, params.use_limit_per_group);

    return query_plan(table_sink(limited));
}


std::shared_ptr<QueryPlan> post_reviews_hybrid(std::shared_ptr<Database>& db,
                                       IndexPtr index,
                                       const QueryParameters& params) {
    if (!index) {
        throw std::runtime_error("VSDS post_reviews_hybrid query requires a pre-built index & --index_storage_device gpu.");
    }

    // Run on GPU then filter on CPU ( start from CPU )
    
    auto on_CPU = maximus::DeviceType::CPU;
    auto on_GPU = maximus::DeviceType::GPU;

    // Start "data" on CPU
    auto data_source = table_source(db,
                                     "reviews",
                                     schema("reviews"),
                                     {"rv_reviewkey", "rv_rating"},
                                     on_CPU);

    auto vs_query_source = sliced_query_source(db,
                                            "reviews_queries",
                                            schema("reviews_queries"),
                                            {"rv_reviewkey_queries", "rv_embedding_queries"},
                                            on_CPU,
                                            params.query_start,
                                            params.query_count);

    // start index on GPU ( --index_storage_device gpu )
    auto search_params = make_search_parameters(params.faiss_index, params, on_GPU);

    auto vs_node = indexed_vector_join(data_source,
                                         vs_query_source,
                                         "",
                                         "rv_embedding_queries",
                                         index,
                                         params.postfilter_ksearch,
                                         std::nullopt,
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         on_GPU);

    // Filtering result on CPU ( after VS )
    auto filter_expr = expr(arrow_expr(cp::field_ref("rv_rating"), ">=", float32_literal(4.0f)));
    auto filtered = filter(vs_node, filter_expr, on_CPU);
    
    // 4. Apply per-query limit K
    // Use query_count from params for partition pattern
    int64_t num_queries = get_partition_count(params.query_count);
    auto limited = apply_per_query_order_by_limit(filtered, "rv_reviewkey_queries", 
                                          num_queries, params.k, on_CPU, params.metric, params.use_limit_per_group);
    
    return query_plan(table_sink(limited));
}

// ============================================================================
// Post-filtering (Partitioned): vector search then filter inside each partition (Reviews)
// ============================================================================

std::shared_ptr<QueryPlan> post_reviews_partitioned(std::shared_ptr<Database>& db,
                                                     IndexPtr index,
                                                     DeviceType device,
                                                     const QueryParameters& params) {
    if (!index) {
        throw std::runtime_error("VSDS post_reviews_partitioned query requires a pre-built index.");
    }

    auto data_source = table_source(db,
                                     "reviews",
                                     schema("reviews"),
                                     {"rv_reviewkey", "rv_rating"},
                                     device);

    auto vs_query_source = sliced_query_source(db,
                                            "reviews_queries",
                                            schema("reviews_queries"),
                                            {"rv_reviewkey_queries", "rv_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

    auto vs_node = indexed_vector_join(data_source,
                                         vs_query_source,
                                         "",
                                         "rv_embedding_queries",
                                         index,
                                         params.postfilter_ksearch,
                                         std::nullopt,
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         params.vs_device);

    auto filter_expr = expr(arrow_expr(cp::field_ref("rv_rating"), ">=", float32_literal(4.0f)));
    int64_t num_queries = get_partition_count(params.query_count);
    auto limited = apply_per_query_filter_order_by_limit(vs_node, "rv_reviewkey_queries",
                                                          num_queries, params.k, filter_expr,
                                                          device, params.metric);

    return query_plan(table_sink(limited));
}

// ============================================================================
// Post-filtering (Filter-Partitioned): Single ANN search, scatter by filter
// group, per-group rv_rating threshold (Reviews)
// ============================================================================

std::shared_ptr<QueryPlan> post_reviews_filter_partitioned(std::shared_ptr<Database>& db,
                                                            IndexPtr index,
                                                            DeviceType device,
                                                            const QueryParameters& params) {
    if (!index) {
        throw std::runtime_error("VSDS post_reviews_filter_partitioned query requires a pre-built index.");
    }

    int64_t num_queries = get_partition_count(params.query_count);
    if (num_queries <= 0) {
        throw std::runtime_error("post_reviews_filter_partitioned requires explicit query_count > 0");
    }
    if (num_queries == 1) {
        return post_reviews(db, index, device, params);
    }

    // Hardcoded unique rv_rating filter thresholds (same as pre_reviews_partitioned)
    std::vector<float> filter_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int num_groups = std::min(static_cast<int>(filter_values.size()),
                              static_cast<int>(num_queries));

    // 1. Build combined query source with filter_group_id column
    std::vector<std::shared_ptr<QueryNode>> query_group_sources;
    int64_t queries_assigned = 0;

    for (int g = 0; g < num_groups; ++g) {
        // Contiguous block assignment (distribute remainder to earlier groups)
        int64_t block_size = num_queries / num_groups
                           + (g < (num_queries % num_groups) ? 1 : 0);
        int64_t block_start = queries_assigned;
        queries_assigned += block_size;

        // Query vectors for this contiguous block
        auto q_source = sliced_query_source(
            db, "reviews_queries", schema("reviews_queries"),
            {"rv_reviewkey_queries", "rv_embedding_queries"},
            device, params.query_start + block_start, block_size);

        // Project filter_group_id = g onto the query source
        std::vector<std::shared_ptr<Expression>> proj_exprs = {
            expr(cp::field_ref("rv_reviewkey_queries")),
            expr(cp::field_ref("rv_embedding_queries")),
            expr(int64_literal(static_cast<int64_t>(g)))
        };
        std::vector<std::string> proj_cols = {"rv_reviewkey_queries", "rv_embedding_queries", "filter_group_id"};
        auto q_with_group = project(q_source, proj_exprs, proj_cols, device);

        query_group_sources.push_back(q_with_group);
    }

    // Gather all query groups into a single combined query source
    auto combined_queries = gather(query_group_sources, device);

    // 2. Data source
    auto data_source = table_source(db,
                                     "reviews",
                                     schema("reviews"),
                                     {"rv_reviewkey", "rv_rating"},
                                     device);

    // 3. Single indexed vector search with all queries
    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

    auto vs_node = indexed_vector_join(data_source,
                                         combined_queries,
                                         "",
                                         "rv_embedding_queries",
                                         index,
                                         params.postfilter_ksearch,
                                         std::nullopt,
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         params.vs_device);

    // 4. Scatter by filter_group_id into num_groups partitions
    auto scattered = scatter(vs_node, {"filter_group_id"}, num_groups, device);

    // 5. Per-group: filter with group-specific threshold, then per-query order+limit
    std::vector<std::shared_ptr<QueryNode>> group_results;
    queries_assigned = 0;

    for (int g = 0; g < num_groups; ++g) {
        int64_t block_size = num_queries / num_groups
                           + (g < (num_queries % num_groups) ? 1 : 0);

        // Filter with this group's rv_rating threshold
        auto filter_expr = expr(arrow_expr(
            cp::field_ref("rv_rating"), ">=", float32_literal(filter_values[g])));
        auto filtered = filter(scattered, filter_expr, device);

        // Per-query order+limit within this group
        auto limited = apply_per_query_order_by_limit(
            filtered, "rv_reviewkey_queries", block_size, params.k,
            device, params.metric, params.use_limit_per_group);

        group_results.push_back(limited);
    }

    // 6. Gather all group results
    return query_plan(table_sink(gather(group_results, device)));
}

// ============================================================================
// Post-filtering: vector search then filter (Images)
// ANN search then filter i_variant = 'MAIN'
// ============================================================================

std::shared_ptr<QueryPlan> post_images(std::shared_ptr<Database>& db,
                                       IndexPtr index,
                                       DeviceType device,
                                       const QueryParameters& params) {
    if (!index) {
        throw std::runtime_error("VSDS post_images query requires a pre-built index.");
    }
    
    // 1. Load source table and vector search queries
    auto data_source = table_source(db,
                                     "images",
                                     schema("images"),
                                     images_data_columns(params),
                                     device);

    // Query table with slicing support for batch execution
    auto vs_query_source = sliced_query_source(db,
                                            "images_queries",
                                            schema("images_queries"),
                                            {"i_imagekey_queries", "i_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    // 2. Perform indexed vector join
    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

    auto vs_node = indexed_vector_join(data_source,
                                         vs_query_source,
                                         "i_embedding",
                                         "i_embedding_queries",
                                         index,
                                         params.postfilter_ksearch,
                                         std::nullopt,
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         params.vs_device);

    // 3. Post-filter: i_partkey < threshold (if set) or i_variant == "MAIN" (default)
    auto filter_expr = make_images_filter(params);
    auto filtered = filter(vs_node, filter_expr, device);
    
    // 4. Apply per-query limit K
    // Use query_count from params for partition pattern
    int64_t num_queries = get_partition_count(params.query_count);
    auto limited = apply_per_query_order_by_limit(filtered, "i_imagekey_queries", 
                                          num_queries, params.k, device, params.metric, params.use_limit_per_group);
    
    return query_plan(table_sink(limited));
}


std::shared_ptr<QueryPlan> post_images_hybrid(std::shared_ptr<Database>& db,
                                       IndexPtr index,
                                       const QueryParameters& params) {
    if (!index) {
        throw std::runtime_error("VSDS post_images_hybrid query requires a pre-built index.");
    }

    // Run on GPU then filter on CPU ( start from CPU )
    
    auto on_CPU = maximus::DeviceType::CPU;
    auto on_GPU = maximus::DeviceType::GPU;

    // Start "data" on CPU
    auto data_source = table_source(db,
                                     "images",
                                     schema("images"),
                                     images_data_columns(params),
                                     on_CPU);

    auto vs_query_source = sliced_query_source(db,
                                            "images_queries",
                                            schema("images_queries"),
                                            {"i_imagekey_queries", "i_embedding_queries"},
                                            on_CPU,
                                            params.query_start,
                                            params.query_count);

    // start index on GPU ( --index_storage_device gpu )
    auto search_params = make_search_parameters(params.faiss_index, params, on_GPU);

    auto vs_node = indexed_vector_join(data_source,
                                         vs_query_source,
                                         "i_embedding",
                                         "i_embedding_queries",
                                         index,
                                         params.postfilter_ksearch,
                                         std::nullopt,
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         on_GPU);

    // Filtering result on CPU ( after VS )
    auto filter_expr = make_images_filter(params);
    auto filtered = filter(vs_node, filter_expr, on_CPU);
    
    // 4. Apply per-query limit K
    // Use query_count from params for partition pattern
    int64_t num_queries = get_partition_count(params.query_count);
    auto limited = apply_per_query_order_by_limit(filtered, "i_imagekey_queries", 
                                          num_queries, params.k, on_CPU, params.metric, params.use_limit_per_group);
    
    return query_plan(table_sink(limited));
}


// ============================================================================
// Post-filtering (Partitioned): vector search then filter inside each partition (Images)
// ============================================================================

std::shared_ptr<QueryPlan> post_images_partitioned(std::shared_ptr<Database>& db,
                                                    IndexPtr index,
                                                    DeviceType device,
                                                    const QueryParameters& params) {
    if (!index) {
        throw std::runtime_error("VSDS post_images_partitioned query requires a pre-built index.");
    }

    auto data_source = table_source(db,
                                     "images",
                                     schema("images"),
                                     {"i_imagekey", "i_embedding", "i_variant"},
                                     device);

    auto vs_query_source = sliced_query_source(db,
                                            "images_queries",
                                            schema("images_queries"),
                                            {"i_imagekey_queries", "i_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

    auto vs_node = indexed_vector_join(data_source,
                                         vs_query_source,
                                         "i_embedding",
                                         "i_embedding_queries",
                                         index,
                                         params.postfilter_ksearch,
                                         std::nullopt,
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         params.vs_device);

    auto filter_expr = expr(arrow_expr(cp::field_ref("i_variant"), "==", string_literal("MAIN")));
    int64_t num_queries = get_partition_count(params.query_count);
    auto limited = apply_per_query_filter_order_by_limit(vs_node, "i_imagekey_queries",
                                                          num_queries, params.k, filter_expr,
                                                          device, params.metric);

    return query_plan(table_sink(limited));
}


// ============================================================================
// Bitmap-filtering: Project 'filter' -> Indexed Vector Search (Reviews)
// ============================================================================

std::shared_ptr<QueryPlan> pre_bitmap_ann_reviews(std::shared_ptr<Database>& db,
                                                  IndexPtr index,
                                                  DeviceType device,
                                                  const QueryParameters& params) {
    if (device != DeviceType::CPU) {
        throw std::runtime_error("pre_bitmap_ann_reviews is currently only supported on CPU.");
    }
    if (!index) {
        throw std::runtime_error("VSDS pre_bitmap_ann_reviews query requires a pre-built index.");
    }
    
    // 1. Load source table and vector search queries
    // Need columns for filter calculation
    auto data_source = table_source(db,
                                     "reviews",
                                     schema("reviews"),
                                     {"rv_reviewkey", "rv_rating"},
                                     device);
                                     
    auto vs_query_source = sliced_query_source(db,
                                            "reviews_queries",
                                            schema("reviews_queries"),
                                            {"rv_reviewkey_queries", "rv_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    // 2. Project 'filter' column (rv_rating >= 4.0)
    auto pred_expr = expr(arrow_expr(cp::field_ref("rv_rating"), ">=", float32_literal(4.0f)));
    
    // Columns to keep + new filter column
    std::vector<std::string> project_cols = {"rv_reviewkey", "rv_rating", "pred_expr"};
    std::vector<std::shared_ptr<Expression>> project_exprs = {
        expr(cp::field_ref("rv_reviewkey")),
        expr(cp::field_ref("rv_rating")),
        pred_expr
    };
    
    auto projected_data = project(data_source, project_exprs, project_cols, device);

    // 3. Indexed Vector Join with Bitmap
    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);
    
    auto vs_node = indexed_vector_join(projected_data,
                                         vs_query_source,
                                         "", // Index owns embeddings
                                         "rv_embedding_queries",
                                         index,
                                         params.k,
                                         std::nullopt,
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         device,
                                         nullptr, // No expr
                                         arrow::FieldRef("pred_expr")); // Use bitmap column

    return query_plan(table_sink(vs_node));
}

// ============================================================================
// Bitmap-filtering: Project 'filter' -> Indexed Vector Search (Images)
// ============================================================================

std::shared_ptr<QueryPlan> pre_bitmap_ann_images(std::shared_ptr<Database>& db,
                                                 IndexPtr index,
                                                 DeviceType device,
                                                 const QueryParameters& params) {
    if (device != DeviceType::CPU) {
        throw std::runtime_error("pre_bitmap_ann_images is currently only supported on CPU.");
    }
    if (!index) {
        throw std::runtime_error("VSDS pre_bitmap_ann_images query requires a pre-built index.");
    }
    
    auto data_source = table_source(db,
                                     "images",
                                     schema("images"),
                                     {"i_imagekey", "i_variant"},
                                     device);
                                     
    auto vs_query_source = sliced_query_source(db,
                                            "images_queries",
                                            schema("images_queries"),
                                            {"i_imagekey_queries", "i_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    // Project 'filter' column (i_variant == "MAIN")
    auto pred_expr = expr(arrow_expr(cp::field_ref("i_variant"), "==", string_literal("MAIN")));
    
    std::vector<std::string> project_cols = {"i_imagekey", "i_variant", "pred_expr"};
    std::vector<std::shared_ptr<Expression>> project_exprs = {
        expr(cp::field_ref("i_imagekey")),
        expr(cp::field_ref("i_variant")),
        pred_expr
    };

    auto projected_data = project(data_source, project_exprs, project_cols, device);

    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);
    
    auto vs_node = indexed_vector_join(projected_data,
                                         vs_query_source,
                                         "", 
                                         "i_embedding_queries",
                                         index,
                                         params.k,
                                         std::nullopt,
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         device,
                                         nullptr,
                                         arrow::FieldRef("pred_expr"));

    return query_plan(table_sink(vs_node));
}

// ============================================================================
// Expr-filtering: Inline Arrow Expression -> Indexed Vector Search (Reviews)
// ============================================================================

std::shared_ptr<QueryPlan> inline_expr_ann_reviews(std::shared_ptr<Database>& db,
                                                   IndexPtr index,
                                                   DeviceType device,
                                                   const QueryParameters& params) {
    if (device != DeviceType::CPU) {
        throw std::runtime_error("inline_expr_ann_reviews is currently only supported on CPU.");
    }
    if (!index) {
        throw std::runtime_error("VSDS inline_expr_ann_reviews query requires a pre-built index.");
    }

    // Load source columns needed for expression evaluation
    auto data_source = table_source(db,
                                     "reviews",
                                     schema("reviews"),
                                     {"rv_reviewkey", "rv_rating"},
                                     device);
                                     
    auto vs_query_source = sliced_query_source(db,
                                            "reviews_queries",
                                            schema("reviews_queries"),
                                            {"rv_reviewkey_queries", "rv_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    // Define filter expression: rv_rating >= 4.0
    auto pred_expr = expr(arrow_expr(cp::field_ref("rv_rating"), ">=", float32_literal(4.0f)));

    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);
    
    auto vs_node = indexed_vector_join(data_source,
                                         vs_query_source,
                                         "", 
                                         "rv_embedding_queries",
                                         index,
                                         params.k,
                                         std::nullopt,
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         device,
                                         pred_expr, // callback expression 
                                         std::nullopt);

    return query_plan(table_sink(vs_node));
}

// ============================================================================
// Expr-filtering: Inline Arrow Expression -> Indexed Vector Search (Images)
// ============================================================================

std::shared_ptr<QueryPlan> inline_expr_ann_images(std::shared_ptr<Database>& db,
                                                  IndexPtr index,
                                                  DeviceType device,
                                                  const QueryParameters& params) {
    if (device != DeviceType::CPU) {
        throw std::runtime_error("inline_expr_ann_images is currently only supported on CPU.");
    }
    if (!index) {
        throw std::runtime_error("VSDS inline_expr_ann_images query requires a pre-built index.");
    }

    auto data_source = table_source(db,
                                     "images",
                                     schema("images"),
                                     {"i_imagekey", "i_variant"},
                                     device);
                                     
    auto vs_query_source = sliced_query_source(db,
                                            "images_queries",
                                            schema("images_queries"),
                                            {"i_imagekey_queries", "i_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    // Define filter expression: i_variant == "MAIN"
    auto pred_expr = expr(arrow_expr(cp::field_ref("i_variant"), "==", string_literal("MAIN")));

    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);
    
    auto vs_node = indexed_vector_join(data_source,
                                         vs_query_source,
                                         "", 
                                         "i_embedding_queries",
                                         index,
                                         params.k,
                                         std::nullopt,
                                         search_params,
                                         false,
                                         false,
                                         "vs_distance",
                                         device,
                                         pred_expr, // expression for inline filtering
                                         std::nullopt);

    return query_plan(table_sink(vs_node));
}

// ============================================================================
// Pre-join: Filter part table, join with reviews, then vector search
// Strategy: part(filter p_size >= 10) → join reviews → EXHAUSTIVE vector search
// 
// This is analogous to pre-filtering but with a join:
// 1. Filter the part table first (e.g., p_size >= 10)
// 2. Join filtered parts with reviews on rv_partkey = p_partkey
// 3. Then do EXHAUSTIVE vector search on the joined (reduced) reviews subset
//
// Uses exhaustive_vector_join because:
// - The data is filtered/joined before search, so we can't use a pre-built index
// - Exhaustive search on reduced dataset is the correct approach
//
// NOTE on batching: Query slicing applies to the query vectors (reviews_queries).
//                   The part filter + join happens on the data side (reviews table),
//                   so batching works correctly - each query batch searches
//                   against the same filtered+joined data subset.
// ============================================================================

std::shared_ptr<QueryPlan> prejoin_reviews(std::shared_ptr<Database>& db,
                                            DeviceType device,
                                            const QueryParameters& params) {
    // Note: index parameter kept for API consistency but not used (exhaustive search)
    
    // 1. Load related tables
    auto part_source = table_source(db,
                                     "part",
                                     maximus::tpch::schema("part"),
                                     {"p_partkey", "p_size"},
                                     device);
    
    auto reviews_source = table_source(db,
                                        "reviews",
                                        schema("reviews"),
                                        {"rv_reviewkey", "rv_partkey", "rv_embedding"},
                                        device);

    auto vs_query_source = sliced_query_source(db,
                                            "reviews_queries",
                                            schema("reviews_queries"),
                                            {"rv_reviewkey_queries", "rv_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    // 2. Apply filter to PART table
    auto part_filter_expr = expr(arrow_expr(cp::field_ref("p_size"), ">=", int32_literal(10)));
    auto filtered_parts = filter(part_source, part_filter_expr, device);
    
    // 3. Join filtered parts with reviews
    auto joined = inner_join(filtered_parts,
                              reviews_source,
                              {"p_partkey"},
                              {"rv_partkey"},
                              "",
                              "",
                              device);
    
    // 4. EXHAUSTIVE vector search on filtered data
    auto vs_node = exhaustive_vector_join(joined,
                                            vs_query_source,
                                            "rv_embedding",
                                            "rv_embedding_queries",
                                            params.metric,
                                            params.k,
                                            std::nullopt,
                                            false,
                                            false,
                                            "vs_distance",
                                            params.vs_device);

    return query_plan(table_sink(vs_node));
}

// ============================================================================
// Post-join: Vector search first, then join with part table and filter
// Strategy: vector search(reviews) → join part → filter p_size >= 10
//
// This is analogous to post-filtering but with a join:
// 1. Do vector search to get top-K (or postfilter_ksearch) neighbors
// 2. Join results with part table on rv_partkey = p_partkey
// 3. Filter on part attributes (e.g., p_size >= 10)
//
// NOTE on batching: Query slicing works correctly here - each query batch
//                   gets its k neighbors, then join+filter happens per batch.
//                   Results maintain correct per-query grouping.
// ============================================================================

std::shared_ptr<QueryPlan> postjoin_reviews(std::shared_ptr<Database>& db,
                                             IndexPtr index,
                                             DeviceType device,
                                             const QueryParameters& params) {
    if (!index) {
        throw std::runtime_error("VSDS postjoin_reviews query requires a pre-built index.");
    }
    
    // 1. Load source tables
    auto data_source = table_source(db,
                                     "reviews",
                                     schema("reviews"),
                                     {"rv_reviewkey", "rv_embedding", "rv_partkey", "rv_rating"},
                                     device);

    auto part_source = table_source(db,
                                     "part",
                                     maximus::tpch::schema("part"),
                                     {"p_partkey", "p_size", "p_brand"},
                                     device);

    auto query_source = sliced_query_source(db,
                                            "reviews_queries",
                                            schema("reviews_queries"),
                                            {"rv_reviewkey_queries", "rv_embedding_queries"},
                                            device,
                                            params.query_start,
                                            params.query_count);

    // 2. Vector search first - use postfilter_ksearch to get more neighbors before filtering
    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);
    
    auto vs_node = indexed_vector_join(data_source,
                                            query_source,
                                            "rv_embedding",
                                            "rv_embedding_queries",
                                            index,
                                            params.postfilter_ksearch,  // Get more to allow for filtering loss
                                            std::nullopt,
                                            search_params,
                                            false,
                                            false,
                                            "vs_distance",
                                            params.vs_device);

    // 3. Join vector search results with part table
    auto joined = inner_join(vs_node,
                              part_source,
                              {"rv_partkey"},
                              {"p_partkey"},
                              "",
                              "",
                              device);

    // 4. Filter on part attribute: p_size >= 10
    auto filter_expr = expr(arrow_expr(cp::field_ref("p_size"), ">=", int32_literal(10)));
    auto filtered = filter(joined, filter_expr, device);
    
    // 5. Apply per-query limit K
    // Use query_count from params for partition pattern
    int64_t num_queries = get_partition_count(params.query_count);
    auto limited = apply_per_query_order_by_limit(filtered, "rv_reviewkey_queries", 
                                          num_queries, params.k, device, params.metric, params.use_limit_per_group);

    return query_plan(table_sink(limited));
}


// ============================================================================
//                             VSDS QUERIES
// ============================================================================ 


std::shared_ptr<QueryPlan> q2_start(std::shared_ptr<Database>& db, IndexPtr index, DeviceType device, const QueryParameters& params) {
    // single query at a time only
    if (params.query_count != 1) {
        throw std::runtime_error("Only one query is supported at a time (query_count must be 1)");
    }

    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    auto part = table_source(
        db, "part", schema("part"), {"p_partkey", "p_type", "p_size", "p_mfgr"}, device);

    auto partsupp = table_source(
        db, "partsupp", schema("partsupp"), {"ps_partkey", "ps_suppkey", "ps_supplycost"}, device);

    auto supplier = table_source(
        db,
        "supplier",
        schema("supplier"),
        {"s_suppkey", "s_nationkey", "s_acctbal", "s_name", "s_address", "s_phone", "s_comment"},
        device);

    auto nation = table_source(db, "nation", schema("nation"), {}, device);

    auto region = table_source(db, "region", schema("region"), {}, device);

    /* ==============================
     * Vector Search Tables
     * ==============================
     */
    auto images = table_source(db, 
        "images", 
        schema("images"), 
        params.use_post ? std::vector<std::string>{"i_partkey", "i_imagekey", "i_variant"} : std::vector<std::string>{"i_partkey", "i_imagekey", "i_embedding", "i_variant"},
        device
    );

    auto vs_query_source = sliced_query_source(
        db,
        "images_queries",
        schema("images_queries"),
        {"i_embedding_queries"},
        device,
        params.query_start,
        params.query_count
    );

    /* ==============================
     *         VECTOR SEARCH
     * ==============================
     */
    std::shared_ptr<QueryNode> vector_search_out_node;

    if (!params.use_post) {
        //  ==========================
        //   VERSION 1: PRE FILTER
        //  ==========================
        
        // 1. Filter for variant = 'Main'
        auto img_filter_expr = expr(arrow_expr(cp::field_ref("i_variant"), "==", string_literal("MAIN")));
        auto img_filter_node = filter(images, img_filter_expr, device);
        
        // 2. Do the vector search

        // NOTE: if dataset is very large and we have to use "large_list" array, then this has to be indexed_vector_join so faiss owns data w/ int64...
        vector_search_out_node = exhaustive_vector_join(
            img_filter_node, // data
            vs_query_source, // query
            "i_embedding",   // data vector column
            "i_embedding_queries", // query vector column
            params.metric,
            params.k,
            std::nullopt,
            false,
            false,
            "vs_distance",
            params.vs_device
        );
    } else {
        //  ==========================
        //   VERSION 2: POST FILTER
        //  ==========================
        
        auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

        auto postfilter_vector_search_node = indexed_vector_join(
            images,
            vs_query_source,
            "", // data is already in the index for `i_embedding` column
            "i_embedding_queries",
            index,
            params.postfilter_ksearch,
            std::nullopt,
            search_params,
            false,
            false,
            "vs_distance",
            params.vs_device
        );

        auto img_filter_expr = expr(arrow_expr(cp::field_ref("i_variant"), "==", string_literal("MAIN")));
        auto img_filter_node = filter(postfilter_vector_search_node, img_filter_expr, device);

        assert(index->metric() == params.metric && "Index metric must match params metric");
        SortOrder order = (index->metric() == VectorDistanceMetric::INNER_PRODUCT) ? SortOrder::DESCENDING : SortOrder::ASCENDING;
        auto postfilter_sort_node = order_by(img_filter_node, {SortKey("vs_distance", order)}, device);
        
        // limit-k since we used postfilter_ksearch 
        vector_search_out_node = limit(postfilter_sort_node, params.k, 0, device);
    }
    
    // /* ==============================
    //  *         END OF VECTOR SEARCH
    //  * ==============================
    //  */

    // Drop variant column no longer needed
    auto vector_search_project_node = project(vector_search_out_node, {"i_partkey", "i_imagekey", "vs_distance"}, device);
    
    /* ==============================
     * CREATING A PROJECT FOR PART
     * ==============================
     */
    auto part_project_node = project(part, {"p_partkey", "p_mfgr"}, device);

    /* ==============================
     *  CREATING IM<->PART JOIN
     * ==============================
     */
    // First join start with vector search results
    auto im_part_join_node = inner_join(
        vector_search_project_node,
        part_project_node,
        {"i_partkey"},
        {"p_partkey"},
        "",
        "",
        device
    );

    /* ==============================
     * CREATING PSP JOIN
     * ==============================
     */
    auto psp_join_node =
        inner_join(im_part_join_node, partsupp, {"p_partkey"}, {"ps_partkey"}, "", "", device);

    /* ==============================
     * CREATING A PSPS JOIN
     * ==============================
     */
    auto psps_join_node =
        inner_join(psp_join_node, supplier, {"ps_suppkey"}, {"s_suppkey"}, "", "", device);
    

    /* =================================
     * CREATING A PROJECT FOR PSPS JOIN
     * =================================
     */
    auto psps_project_node = project(psps_join_node,
                                     {"p_partkey",
                                      "ps_supplycost",
                                      "p_mfgr",
                                      "s_nationkey",
                                      "s_acctbal",
                                      "s_name",
                                      "s_address",
                                      "s_phone",
                                      "s_comment",
                                      "i_imagekey",
                                      "vs_distance"},
                                     device);
    // return query_plan(table_sink(psps_project_node)); // for DEBUG ; rest of code not executed

    /* =================================
     * CREATING A REGION FILTER
     * =================================
     */
    auto region_filter = expr(arrow_expr(cp::field_ref("r_name"), "==", string_literal("EUROPE")));
    auto region_filter_node = filter(region, region_filter, device);
    

    /* =================================
     * CREATING A NR JOIN
     * =================================
     */
    auto nr_join_node =
        inner_join(nation, region_filter_node, {"n_regionkey"}, {"r_regionkey"}, "", "", device);

    /* =================================
     * CREATING A NR PROJECT
     * =================================
     */
    auto nr_project_node = project(nr_join_node, {"n_nationkey", "n_name"}, device); // CORRECT

    /* =================================
     * CREATING A PSPS-NR JOIN
     * =================================
     */

    auto psps_nr_join_node = inner_join(
        psps_project_node, nr_project_node, {"s_nationkey"}, {"n_nationkey"}, "", "", device);
    // return query_plan(table_sink(psps_nr_join_node)); // for DEBUG ; rest of code not executed
    // SOS: By this point something has gone wrong for whatever reason....

    /* ====================================
     * CREATING A PROJECT FOR PSPS-NR JOIN
     * ====================================
     */
    auto psps_nr_project_node = project(psps_nr_join_node,
                                        {"p_partkey",
                                         "ps_supplycost",
                                         "p_mfgr",
                                         "n_name",
                                         "s_acctbal",
                                         "s_name",
                                         "s_address",
                                         "s_phone",
                                         "s_comment",
                                         "i_imagekey",
                                         "vs_distance"},
                                        device);

    /* =================================
     * CREATING A GROUP BY NODE
     * =================================
     */
    auto aggs          = {aggregate("hash_min", "ps_supplycost", "min_supplycost")};
    auto group_by_node = group_by(psps_nr_project_node, {"p_partkey"}, aggs, device);

    /* =================================
     * CREATING A PSPSNR-AGGR JOIN
     * =================================
     */
    // here, psps_nr_project_node has two outputs: it's connected to group_by_node (previous) and pspsnr_aggr_join_node here
    auto pspsnr_aggr_join_node = inner_join(
        psps_nr_project_node,
        group_by_node,
        {"p_partkey", "ps_supplycost"},
        {"p_partkey", "min_supplycost"},
        "_l",
        "_r",
        device);  // here we add a suffix to the right keys, since arrow doesn't coalesce keys with the same name

    /* =================================
     * CREATING A PROJECT FOR PSPSNR-AGGR JOIN
     * =================================
     */
    

    auto pspsnr_aggr_project_node = project(pspsnr_aggr_join_node,
                                            exprs({
                                                   "i_imagekey",
                                                   "vs_distance",
                                                   "s_acctbal",
                                                   "s_name",
                                                   "n_name",
                                                   "p_partkey_l",
                                                   "p_mfgr",
                                                   "s_address",
                                                   "s_phone",
                                                   "s_comment"}),
                                            {
                                             "i_imagekey",
                                             "vs_distance",
                                             "s_acctbal",
                                             "s_name",
                                             "n_name",
                                             "p_partkey",
                                             "p_mfgr",
                                             "s_address",
                                             "s_phone",
                                             "s_comment"},
                                            device);

    /* =================================
     * CREATING A ORDER BY NODE
     * =================================
     */

    SortOrder vs_distance_order = (params.metric == VectorDistanceMetric::INNER_PRODUCT) ? SortOrder::DESCENDING : SortOrder::ASCENDING;
    std::vector<SortKey> sort_keys = {
        {"s_acctbal", SortOrder::DESCENDING}, {"vs_distance", vs_distance_order}, {"n_name"}, {"s_name"}, {"p_partkey"}};
    auto order_by_node = order_by(pspsnr_aggr_project_node, sort_keys, device);

    /* =================================
     * CREATING A LIMIT NODE
     * =================================
     */
    auto limit_node = limit(order_by_node, 100, 0, device);

    return query_plan(table_sink(limit_node));
}



std::shared_ptr<QueryPlan> q10_mid(std::shared_ptr<Database>& db, IndexPtr index, DeviceType device, const QueryParameters& params) {

    // single query at a time only
    if (params.query_count != 1) {
        throw std::runtime_error("Only one query is supported at a time (query_count must be 1)");
    }

    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    auto lineitem = table_source(db,
                                 "lineitem",
                                 schema("lineitem"),
                                 {"l_orderkey", "l_returnflag", "l_extendedprice", "l_discount"},
                                 device);

    auto orders = table_source(
        db, "orders", schema("orders"), {"o_orderkey", "o_custkey", "o_orderdate"}, device);

    auto customer = table_source(
        db,
        "customer",
        schema("customer"),
        {"c_custkey", "c_nationkey", "c_name", "c_acctbal", "c_phone", "c_address", "c_comment"},
        device);

    auto nation = table_source(db, "nation", schema("nation"), {"n_nationkey", "n_name"}, device);

    /* ==============================
     * Vector Search Tables
     * ==============================
     */
    auto reviews = table_source(db, 
        "reviews", 
        schema("reviews"), 
        {"rv_custkey"},
        device
    );

    auto vs_query_source = sliced_query_source(
        db,
        "reviews_queries",
        schema("reviews_queries"),
        {"rv_embedding_queries"},
        device,
        params.query_start,
        params.query_count
    );
    
    /* ============================================================
     *      Branch 1: Vector Search
     * ============================================================
     */

    /* ==============================
     *         VECTOR SEARCH - Independent Branch ("MID")
     * ==============================
     */
    
    // 1. Do the vector search
    // Create search parameters based on index type
    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

    auto vector_search_node = indexed_vector_join(
        reviews,
        vs_query_source,
        "", // data is already in the index for `rv_embedding` column
        "rv_embedding_queries",
        index,
        params.k,
        std::nullopt,
        search_params,
        false,
        false,
        "vs_distance",
        params.vs_device
    );
        
    // We only need the custkey
    auto vector_branch_root_node = distinct(project(vector_search_node, {"rv_custkey"}, device), {"rv_custkey"}, device);
    

    // return query_plan(table_sink(vector_branch_root_node)); // DEBUG 
    
    /* ============================================================
     *         Branch 2: TPCH Q18
     * ============================================================
     */


    /* ==============================
     * CREATING A LINEITEM FILTER
     * ==============================
     */
    auto lineitem_filter_expr =
        expr(arrow_expr(cp::field_ref("l_returnflag"), "==", string_literal("R")));
    auto filter_lineitem_node = filter(lineitem, lineitem_filter_expr, device);

    /* ==============================
     * CREATING A LINEITEM PROJECT
     * ==============================
     */
    auto project_lineitem_node =
        project(filter_lineitem_node, {"l_orderkey", "l_extendedprice", "l_discount"}, device);

    /* ==============================
     * CREATING AN ORDERS FILTER
     * ==============================
     */
    auto order_date_filter  = expr(arrow_in_range(
        cp::field_ref("o_orderdate"), date_literal("1994-01-01"), date_literal("1994-04-01")));
    auto filter_orders_node = filter(orders, order_date_filter, device);

    /* ==============================
     * CREATING AN ORDERS PROJECT
     * ==============================
     */
    auto project_orders_node = project(filter_orders_node, {"o_orderkey", "o_custkey"}, device);

    /* =========================================
     * CREATING A JOIN NODE (LINEITEM, ORDERS)
     * =========================================
     */
    auto lo_join_node = inner_join(
        project_lineitem_node, project_orders_node, {"l_orderkey"}, {"o_orderkey"}, "", "", device);
    
    /* =========================================
     * CREATING A LO-JOIN PROJECTION
     * =========================================
     */
    std::vector<std::string> lo_project_columns = {"l_extendedprice", "l_discount", "o_custkey"};
    auto lo_project_expr                        = exprs(lo_project_columns);
    // add a new column "volume" to the expressions and the column names
    lo_project_expr.push_back(
        expr(arrow_expr(cp::field_ref("l_extendedprice"),
                        "*",
                        arrow_expr(float64_literal(1.00), "-", cp::field_ref("l_discount")))));
    lo_project_columns.emplace_back("volume");
    auto project_lo_node = project(lo_join_node, lo_project_expr, lo_project_columns, device);

    /* =========================================
     * CREATING A GROUP-BY FOR LO-JOIN PROJECT
     * =========================================
     */
    std::vector<std::shared_ptr<Aggregate>> aggregates = {
        aggregate("hash_sum", "volume", "revenue")};
    auto lo_group_by_node = group_by(project_lo_node, {"o_custkey"}, aggregates, device);

    /* =========================================
     * CREATING A LOC-JOIN NODE
     * =========================================
     */
    auto loc_join_node =
        inner_join(customer, lo_group_by_node, {"c_custkey"}, {"o_custkey"}, "", "", device);

    /* =========================================
     * CREATING A LOCN-JOIN
     * =========================================
     */
    auto locn_join_node =
        inner_join(loc_join_node, nation, {"c_nationkey"}, {"n_nationkey"}, "", "", device);

    /* =========================================
     * CREATING A LOCN-PROJECT
     * =========================================
     */
    auto tpch_branch_root = project(locn_join_node,
                                     {"c_custkey",
                                      "c_name",
                                      "revenue",
                                      "c_acctbal",
                                      "n_name",
                                      "c_address",
                                      "c_phone",
                                      "c_comment"},
                                     device);

    /* ============================================================
     *                  MERGE INDEPENDENT BRANCHES
     * ============================================================
     */
    auto left_join_branches_node =
        left_outer_join(tpch_branch_root, vector_branch_root_node, {"c_custkey"}, {"rv_custkey"}, "", "", device);

    // return query_plan(table_sink(left_join_branches_node)); // DEBUG 
    /* =========================================
     * CREATING IS_IN_TOP_K PROJECTION
     * =========================================
     * -- [+] The Correlation Flag
     * -- Checks if the customer exists in the independent "Top-K" subquery (True/False)
     * (top_k_customers.rv_custkey IS NOT NULL) as is_in_top_k
     */
    std::vector<std::string> merged_columns = {"c_custkey", "c_name", "revenue", "c_acctbal", 
                                                "n_name", "c_address", "c_phone", "c_comment"};
    auto merged_project_exprs = exprs(merged_columns);

    // Add is_in_top_k to the FRONT of both vectors
    merged_project_exprs.insert(merged_project_exprs.begin(), expr(arrow_is_not_null(cp::field_ref("rv_custkey"))));
    merged_columns.insert(merged_columns.begin(), "is_in_top_k");

    auto merged_with_flag = project(left_join_branches_node, merged_project_exprs, merged_columns, device);
    /* =========================================
     * CREATING AN ORDER-BY NODE
     * =========================================
     * ORDER BY is_in_top_k ASC, revenue DESC
     * Note: is_in_top_k ASC means customers NOT in top-k (false/0) come first,
     * then customers IN top-k (true/1) come after.
     */
    auto order_by_node = order_by(merged_with_flag, 
                                   {{"is_in_top_k", SortOrder::DESCENDING}, 
                                    {"revenue", SortOrder::DESCENDING}}, 
                                   device);

    /* =========================================
     * CREATING A LIMIT NODE
     * =========================================
     */
    auto limit_node = limit(order_by_node, 20, 0, device);

    return query_plan(table_sink(limit_node));
}


std::shared_ptr<QueryPlan> q18_mid(std::shared_ptr<Database>& db, IndexPtr index, DeviceType device, const QueryParameters& params) {

    // single query at a time only
    if (params.query_count != 1) {
        throw std::runtime_error("Only one query is supported at a time (query_count must be 1)");
    }    

    //  auto ctx = db->get_context();
    /* ==============================
     * CREATING A TABLE SOURCE NODE
     * ==============================
     */
    auto lineitem = table_source(db, "lineitem", schema("lineitem"), {}, device);
    auto orders   = table_source(db, "orders", schema("orders"), {}, device);
    auto customer = table_source(db, "customer", schema("customer"), {}, device);


    /* ==============================
     * Vector Search Tables
     * ==============================
     */
    auto images = table_source(db, 
        "images", 
        schema("images"), 
        params.use_post ? std::vector<std::string>{"i_imagekey", "i_partkey", "i_variant"} : std::vector<std::string>{},
        device
    );

    auto vs_query_source = sliced_query_source(
        db,
        "images_queries",
        schema("images_queries"),
        {"i_embedding_queries"},
        device,
        params.query_start,
        params.query_count
    );


    /* ============================================================
     *      Branch 1: Vector Search
     * ============================================================
     */

    /* ==============================
     *         VECTOR SEARCH
     * ==============================
     */
    std::shared_ptr<QueryNode> vector_branch_root_node;

    if (!params.use_post) {
        //  ==========================
        //   VERSION 1: PRE FILTER
        //  ==========================
        
        // note: exhaustive will fail on GPU if large_list is used. Here on sf=1 ~700k images so we're good but keep in mind if you error in the future

        // 1. Filter for variant = 'Main'
        auto img_filter_expr = expr(arrow_expr(cp::field_ref("i_variant"), "==", string_literal("MAIN")));
        auto img_filter_node = filter(images, img_filter_expr, device);
        
        // 2. Do the vector search
        auto vector_search_node = exhaustive_vector_join(
            img_filter_node, // data
            vs_query_source, // query
            "i_embedding",   // data vector column
            "i_embedding_queries", // query vector column
            params.metric,
            params.k,
            std::nullopt,
            false,
            false,
            "vs_distance",
            params.vs_device
        );

        // We only need the custkey
        vector_branch_root_node = project(vector_search_node, {"i_partkey"}, device);
    } else {
        //  ==========================
        //   VERSION 2: POST FILTER
        //  ==========================
        //      NOTE: postfilter should be faster also works on GPU easily BUT (not GROUND TRUTH -> Recall hit)
        
        // postfilter 'MAIN' variant
        
        // 1. Do the vector searh
        auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

        auto vector_search_node = indexed_vector_join(
            images,
            vs_query_source,
            "", // data is already in the index for `i_embedding` column
            "i_embedding_queries",
            index,
            params.postfilter_ksearch,
            std::nullopt,
            search_params,
            false,
            false,
            "vs_distance",
            params.vs_device
        );
        
        // 2. Filter for variant = 'Main'
        auto img_filter_expr = expr(arrow_expr(cp::field_ref("i_variant"), "==", string_literal("MAIN")));
        auto post_vs_filter_node = filter(vector_search_node, img_filter_expr, device);

        // order by limit k (since we use postfilter_ksearch in postfiltering...)

        assert(index->metric() == params.metric && "Index metric must match params metric");
        SortOrder order = (index->metric() == VectorDistanceMetric::INNER_PRODUCT) ? SortOrder::DESCENDING : SortOrder::ASCENDING;
        auto postfilter_sort_node = order_by(post_vs_filter_node, {SortKey("vs_distance", order)}, device);
        
        // limit-k since we used postfilter_ksearch 
        auto vector_search_out_node = limit(postfilter_sort_node, params.k, 0, device);

        // We only need the custkey
        vector_branch_root_node = project(vector_search_out_node, {"i_partkey"}, device);
    }

    /* ==============================
     *         END OF VECTOR SEARCH
     * ==============================
     */
    
    /* ============================================================
     *         Branch 2: TPCH Q18 (Relational)
     * ============================================================
     */

    /* =================================
     * HAVING SUBQUERY: lineitem → group_by(l_orderkey, sum(l_quantity)) → filter(sum > 300)
     * Produces qualifying l_orderkey values (the "whale" orders).
     * =================================
     */
    std::vector<std::shared_ptr<Aggregate>> having_aggregates = {
        aggregate("hash_sum", "l_quantity", "sum(l_quantity)"),
    };
    auto having_group_by = group_by(lineitem, {"l_orderkey"}, having_aggregates, device);

    auto having_filter_expr =
        expr(arrow_expr(cp::field_ref("sum(l_quantity)"), ">", float64_literal(300.0)));
    auto having_result = filter(having_group_by, having_filter_expr, device);

    /* =================================
     * MAIN JOIN CHAIN (per-row granularity)
     * We need a second lineitem source because the HAVING subquery
     * destroys per-row columns (l_partkey, l_quantity) via group-by.
     * =================================
     */
    auto lineitem2 = table_source(db, "lineitem", schema("lineitem"),
                                  {"l_orderkey", "l_partkey", "l_quantity"}, device);

    /* ==============================
     * customer JOIN orders
     * ==============================
     */
    auto co_join = inner_join(customer, orders, {"c_custkey"}, {"o_custkey"}, "", "", device);

    /* ==============================
     * LEFT SEMI JOIN: filter orders to only "whale" orderkeys
     * Implements: WHERE o_orderkey IN (SELECT l_orderkey FROM ... HAVING sum > 300)
     * Keeps all columns from co_join, but only rows with qualifying o_orderkey.
     * ==============================
     */
    auto co_filtered = left_semi_join(co_join, having_result, {"o_orderkey"}, {"l_orderkey"}, "", "", device);

    /* ==============================
     * JOIN with lineitem2 (per-row data: l_partkey, l_quantity)
     * ==============================
     */
    auto col_join = inner_join(co_filtered, lineitem2, {"o_orderkey"}, {"l_orderkey"}, "", "", device);

    // return query_plan(table_sink(col_join)); // DEBUG

    /* ============================================================
     *                  MERGE INDEPENDENT BRANCHES
     * ============================================================
     * NOTE: Avoids arrow_if_else which segfaults on GPU (cuDF) with
     *       nullable columns from left_outer_join.
     *       Instead, compute total_qty and similar_qty via separate
     *       aggregation paths, then merge the aggregated results.
     *       Also uses left_semi_join (existence check, no row multiplication)
     *       instead of left_outer_join per-row which could inflate
     *       quantities when vector_branch has duplicate i_partkey.
     */

    // return query_plan(table_sink(col_join)); // DEBUG

    /* ==============================
     * PATH A: total_qty — aggregate directly from col_join (no VS branch)
     * ==============================
     */
    auto col_join_total = project(col_join,
        {"c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice", "l_quantity"}, device);

    std::vector<std::shared_ptr<Aggregate>> total_aggregates = {
        aggregate("hash_sum", "l_quantity", "total_qty"),
    };
    auto total_grouped = group_by(col_join_total,
        {"c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice"},
        total_aggregates, device);

    /* ==============================
     * PATH B: similar_qty — only lineitem rows whose part is in VS results
     * left_semi_join keeps col_join rows where l_partkey ∈ vector results,
     * without adding nullable columns or multiplying rows.
     * ==============================
     */
    auto matched_parts = left_semi_join(col_join, vector_branch_root_node,
        {"l_partkey"}, {"i_partkey"}, "", "", device);
    auto matched_projected = project(matched_parts, {"o_orderkey", "l_quantity"}, device);

    std::vector<std::shared_ptr<Aggregate>> similar_aggregates = {
        aggregate("hash_sum", "l_quantity", "similar_qty"),
        aggregate("hash_count", count_all(), "l_quantity", "num_similar_items"),
    };
    auto similar_grouped = group_by(matched_projected,
        {"o_orderkey"}, similar_aggregates, device);

    /* ==============================
     * MERGE: left outer join aggregated total with aggregated similar
     * Orders with no matched parts get NULL similar_qty,
     * which sorts after all non-null values in DESC order (same as 0).
     * ==============================
     */
    auto final_merged = left_outer_join(total_grouped, similar_grouped,
        {"o_orderkey"}, {"o_orderkey"}, "", "_vs", device);

    auto final_projected = project(final_merged,
        {"c_name", "c_custkey", "o_orderkey", "o_orderdate", "o_totalprice", "total_qty", "similar_qty", "num_similar_items"}, device);

    /* ==============================
     * ORDER BY: Re-rank by similar_qty DESC, o_totalprice DESC, o_orderdate ASC
     * NullOrder::LAST ensures NULLs sort after non-null values in DESC order
     * (consistent behavior across CPU/Acero and GPU/cuDF).
     * ==============================
     */
    std::vector<SortKey> sort_keys = {
        {"similar_qty", SortOrder::DESCENDING},
        {"o_totalprice", SortOrder::DESCENDING},
        {"o_orderdate", SortOrder::ASCENDING}};
    
    // Acero and cuDF have inverted NULL ordering semantics for DESC.
    NullOrder null_order = (device == DeviceType::GPU) ? NullOrder::FIRST : NullOrder::LAST;
    auto order_by_node = order_by(final_projected, sort_keys, device, null_order);

    return query_plan(table_sink(limit(order_by_node, 100, 0, device)));
}


std::shared_ptr<QueryPlan> q16_start(std::shared_ptr<Database>& db, IndexPtr index, DeviceType device, const QueryParameters& params) {
    
    // single query at a time only
    if (params.query_count != 1) {
        throw std::runtime_error("Only one query is supported at a time (query_count must be 1)");
    }    

    
    auto part     = table_source(db, "part", schema("part"), {}, device);
    auto partsupp = table_source(db, "partsupp", schema("partsupp"), {}, device);

    /* ==============================
     * Vector Search Tables
     * ==============================
     */


    auto reviews = table_source(db, 
        "reviews", 
        schema("reviews"), 
        {"rv_partkey"},
        device
    );

    auto vs_query_source = sliced_query_source(
        db,
        "reviews_queries",
        schema("reviews_queries"),
        {"rv_embedding_queries"},
        device,
        params.query_start,
        params.query_count
    );

    /* ==============================
     *         VECTOR SEARCH @Start "driving" the query
     * ==============================
     */
    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

    auto vector_search_node = indexed_vector_join(
        reviews,
        vs_query_source,
        "", // data is already in the index for `rv_embedding` column
        "rv_embedding_queries",
        index,
        params.k,
        std::nullopt,
        search_params,
        false,
        false,
        "vs_distance",
        params.vs_device
    );
    
    /* ==============================
     * NOT IN subquery: Identify suppliers of parts with "complaint-like" reviews
     * SELECT DISTINCT ps_suppkey FROM partsupp WHERE ps_partkey IN (vector search results)
     * ==============================
     */
    // 1. Get the partkeys from the vector search results
    auto vs_partkeys = project(vector_search_node, {"rv_partkey"}, device);

    // 2. Join with partsupp to find the suppliers of those parts
    auto complaint_suppkeys = inner_join(partsupp, vs_partkeys, {"ps_partkey"}, {"rv_partkey"}, "", "", device);

    // 3. Get DISTINCT ps_suppkey
    auto excluded_suppliers = distinct(project(complaint_suppkeys, {"ps_suppkey"}, device), {"ps_suppkey"}, device);

    /* ==============================
     * Main query: part filter + partsupp NOT IN excluded suppliers
     * ==============================
     */
    auto part_filter =
        expr(arrow_all({arrow_expr(cp::field_ref("p_brand"), "!=", string_literal("Brand#45")),
                        arrow_not(arrow_field_starts_with("p_type", "MEDIUM POLISHED")),
                        arrow_in(cp::field_ref("p_size"),
                                 {int32_literal(49),
                                  int32_literal(14),
                                  int32_literal(23),
                                  int32_literal(45),
                                  int32_literal(19),
                                  int32_literal(3),
                                  int32_literal(36),
                                  int32_literal(9)})}));

    auto part_filtered = filter(part, part_filter, device);

    // -- [+] ps_suppkey NOT IN (excluded_suppliers)
    // Uses left_anti_join to exclude suppliers associated with complaint-like reviews
    auto partsupp_filtered = left_anti_join(partsupp, excluded_suppliers, {"ps_suppkey"}, {"ps_suppkey"}, "", "", device);

    auto joined =
        inner_join(part_filtered, partsupp_filtered, {"p_partkey"}, {"ps_partkey"}, "", "", device);

    auto grouped = group_by(joined,
                            {"p_brand", "p_type", "p_size"},
                            {aggregate("hash_count_distinct", "ps_suppkey", "supplier_cnt")},
                            device);

    auto selected = project(grouped, {"p_brand", "p_type", "p_size", "supplier_cnt"}, device);

    auto ordered =
        order_by(selected,
                 {{"supplier_cnt", SortOrder::DESCENDING}, {"p_brand"}, {"p_type"}, {"p_size"}},
                 device);

    return query_plan(table_sink(ordered));

}



std::shared_ptr<QueryPlan> q19_start(std::shared_ptr<Database>& db, IndexPtr reviews_index, IndexPtr images_index, DeviceType device, const QueryParameters& params) {
    
    // single query at a time only
    if (params.query_count != 1) {
        throw std::runtime_error("Only one query is supported at a time (query_count must be 1)");
    }    
     /* ==============================
     * Vector Search Tables
     * ==============================
     */

    auto reviews = table_source(db, 
        "reviews", 
        schema("reviews"), 
        {"rv_partkey"},
        device
    );

    auto vs_query_source_reviews = sliced_query_source(
        db,
        "reviews_queries",
        schema("reviews_queries"),
        {"rv_embedding_queries"},
        device,
        params.query_start,
        params.query_count
    );

    auto images = table_source(db, 
        "images", 
        // Need all columns: i_variant for postfilter, i_embedding for prefilter
        schema("images"), 
        params.use_post ? std::vector<std::string>{"i_partkey", "i_variant"} : std::vector<std::string>{},
        device
    );

    auto vs_query_source_images = sliced_query_source(
        db,
        "images_queries",
        schema("images_queries"),
        {"i_embedding_queries"},
        device,
        params.query_start,
        params.query_count
    );
    
    /* ============================================================
     *      Vector Search Branch 1: Reviews
     * ============================================================
     */
    auto reviews_search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

    auto reviews_vs_node = indexed_vector_join(
        reviews,
        vs_query_source_reviews,
        "", // data is already in the index for `rv_embedding` column
        "rv_embedding_queries",
        reviews_index,
        params.k,
        std::nullopt,
        reviews_search_params,
        false,
        false,
        "vs_distance",
        params.vs_device
    );

    auto reviews_partkeys = distinct(project(reviews_vs_node, {"rv_partkey"}, device), {"rv_partkey"}, device);

    /* ============================================================
     *      Vector Search Branch 2: Images
     * ============================================================
     */

    std::shared_ptr<QueryNode> vector_search_out_node;

    if (!params.use_post) {
        //  ==========================
        //   VERSION 1: PRE FILTER
        //  ==========================
        //      NOTE: prefilter is GROUND TRUTH but slower (exhaustive search on filtered data)
        
        // 1. Filter for variant = 'MAIN'
        auto img_prefilter_expr = expr(arrow_expr(cp::field_ref("i_variant"), "==", string_literal("MAIN")));
        auto img_prefilter_node = filter(images, img_prefilter_expr, device);
        
        // 2. Do exhaustive vector search on filtered data
        vector_search_out_node = exhaustive_vector_join(
            img_prefilter_node,
            vs_query_source_images,
            "i_embedding",
            "i_embedding_queries",
            params.metric,
            params.k,
            std::nullopt,
            false,
            false,
            "vs_distance",
            params.vs_device
        );
    } else {
        //  ==========================
        //   VERSION 2: POST FILTER
        //  ==========================
        auto images_search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

        auto images_vs_node = indexed_vector_join(
            images,
            vs_query_source_images,
            "", // data is already in the index for `i_embedding` column
            "i_embedding_queries",
            images_index,
            params.postfilter_ksearch,
            std::nullopt,
            images_search_params,
            false,
            false,
            "vs_distance",
            params.vs_device
        );
        // NOTE: we could also do postfilter_ksearch and then order by limit to k
        // return query_plan(table_sink(images_vs_node)); // DEBUG 

        // Postfilter: filter for i_variant = 'MAIN' after search
        auto img_filter_expr = expr(arrow_expr(cp::field_ref("i_variant"), "==", string_literal("MAIN")));
        auto img_filter_node = filter(images_vs_node, img_filter_expr, device);

        assert(images_index->metric() == params.metric && "Index metric must match params metric");
        SortOrder order = (images_index->metric() == VectorDistanceMetric::INNER_PRODUCT) ? SortOrder::DESCENDING : SortOrder::ASCENDING;
        auto postfilter_sort_node = order_by(img_filter_node, {SortKey("vs_distance", order)}, device);
        vector_search_out_node = limit(postfilter_sort_node, params.k, 0, device);
    }
    
    /* ==============================
     *         END OF VECTOR SEARCH
     * ==============================
     */

    auto images_partkeys = distinct(project(vector_search_out_node, {"i_partkey"}, device), {"i_partkey"}, device);

    /* ============================================================
     *      Main Query: lineitem JOIN part with 5-way OR filter
     * ============================================================
     */
    auto lineitem = table_source(db, "lineitem", schema("lineitem"), {}, device);
    auto part     = table_source(db, "part", schema("part"), {}, device);

    // Pushdown: common AND predicates across all 5 OR branches.
    // l_shipmode IN ('AIR','AIR REG') AND l_shipinstruct='DELIVER IN PERSON'
    // AND l_quantity BETWEEN 1 AND 50 (tight union of per-branch ranges: 1-11,10-20,20-30,30-40,40-50).
    auto lineitem_prefilter_expr = expr(arrow_all(
        {arrow_in(cp::field_ref("l_shipmode"),
                  {string_literal("AIR"), string_literal("AIR REG")}),
         arrow_expr(cp::field_ref("l_shipinstruct"), "==", string_literal("DELIVER IN PERSON")),
         arrow_between(cp::field_ref("l_quantity"), float64_literal(1), float64_literal(50))}));
    auto lineitem_f = filter(lineitem, lineitem_prefilter_expr, device);

    auto joined = inner_join(lineitem_f, part, {"l_partkey"}, {"p_partkey"}, "", "", device);

    // LEFT OUTER JOIN with reviews VS results to check membership
    auto joined_with_reviews = left_outer_join(joined, reviews_partkeys, {"p_partkey"}, {"rv_partkey"}, "", "", device);

    // LEFT OUTER JOIN with images VS results to check membership
    auto joined_with_both = left_outer_join(joined_with_reviews, images_partkeys, {"p_partkey"}, {"i_partkey"}, "", "", device);

    /* ==============================
     * 5-way OR filter
     * ==============================
     */
    auto filter_expr = expr(arrow_any(
        {// Branch 1: Brand#12/ SM containers / qty 1-11 / size 1-5
         arrow_all(
             {arrow_expr(cp::field_ref("p_brand"), "==", string_literal("Brand#12")),
              arrow_in(cp::field_ref("p_container"),
                       {string_literal("SM CASE"),
                        string_literal("SM BOX"),
                        string_literal("SM PACK"),
                        string_literal("SM PKG")}),
              arrow_between(cp::field_ref("l_quantity"), float64_literal(1), float64_literal(11)),
              arrow_between(cp::field_ref("p_size"), int32_literal(1), int32_literal(5)),
              arrow_in(cp::field_ref("l_shipmode"),
                       {string_literal("AIR"), string_literal("AIR REG")}),
              arrow_expr(
                  cp::field_ref("l_shipinstruct"), "==", string_literal("DELIVER IN PERSON"))}),
         // Branch 2: Brand#23 / MED containers / qty 10-20 / size 1-10
         arrow_all(
             {arrow_expr(cp::field_ref("p_brand"), "==", string_literal("Brand#23")),
              arrow_in(cp::field_ref("p_container"),
                       {string_literal("MED BAG"),
                        string_literal("MED BOX"),
                        string_literal("MED PKG"),
                        string_literal("MED PACK")}),
              arrow_between(cp::field_ref("l_quantity"), float64_literal(10), float64_literal(20)),
              arrow_between(cp::field_ref("p_size"), int32_literal(1), int32_literal(10)),
              arrow_in(cp::field_ref("l_shipmode"),
                       {string_literal("AIR"), string_literal("AIR REG")}),
              arrow_expr(
                  cp::field_ref("l_shipinstruct"), "==", string_literal("DELIVER IN PERSON"))}),
         // Branch 3: Brand#34 / LG containers / qty 20-30 / size 1-15
         arrow_all(
             {arrow_expr(cp::field_ref("p_brand"), "==", string_literal("Brand#34")),
              arrow_in(cp::field_ref("p_container"),
                       {string_literal("LG CASE"),
                        string_literal("LG BOX"),
                        string_literal("LG PACK"),
                        string_literal("LG PKG")}),
              arrow_between(cp::field_ref("l_quantity"), float64_literal(20), float64_literal(30)),
              arrow_between(cp::field_ref("p_size"), int32_literal(1), int32_literal(15)),
              arrow_in(cp::field_ref("l_shipmode"),
                       {string_literal("AIR"), string_literal("AIR REG")}),
              arrow_expr(
                  cp::field_ref("l_shipinstruct"), "==", string_literal("DELIVER IN PERSON"))}),
         // [+] Branch 4: Reviews VS - "semantic similarity" / qty 30-40
         arrow_all(
             {arrow_is_not_null(cp::field_ref("rv_partkey")),
              arrow_between(cp::field_ref("l_quantity"), float64_literal(30), float64_literal(40)),
              arrow_in(cp::field_ref("l_shipmode"),
                       {string_literal("AIR"), string_literal("AIR REG")}),
              arrow_expr(
                  cp::field_ref("l_shipinstruct"), "==", string_literal("DELIVER IN PERSON"))}),
         // [+] Branch 5: Images VS - "visual similarity" / qty 40-50
         arrow_all(
             {arrow_is_not_null(cp::field_ref("i_partkey")),
              arrow_between(cp::field_ref("l_quantity"), float64_literal(40), float64_literal(50)),
              arrow_in(cp::field_ref("l_shipmode"),
                       {string_literal("AIR"), string_literal("AIR REG")}),
              arrow_expr(
                  cp::field_ref("l_shipinstruct"), "==", string_literal("DELIVER IN PERSON"))})}));

    auto filter_node = filter(joined_with_both, filter_expr, device);

    auto project_expr =
        expr(arrow_expr(cp::field_ref("l_extendedprice"),
                        "*",
                        arrow_expr(float64_literal(1.0), "-", cp::field_ref("l_discount"))));

    auto project_node = project(filter_node, {std::move(project_expr)}, {"revenue"}, device);

    auto group_by_node =
        group_by(project_node, {}, {aggregate("sum", "revenue", "revenue")}, device);

    return query_plan(table_sink(group_by_node));

}


std::shared_ptr<QueryPlan> q11_end(std::shared_ptr<Database>& db, IndexPtr index, DeviceType device, const QueryParameters& params) {

    /* ==============================
     * CREATING TABLE SOURCE NODES
     * ==============================
     */
    auto nation = table_source(db, "nation", schema("nation"), {}, device);

    auto partsupp = table_source(db, "partsupp", schema("partsupp"), {}, device);

    auto supplier = table_source(db, "supplier", schema("supplier"), {}, device);

    /* ==============================
     * Vector Search Tables
     * ==============================
     */
    // Data side: only i_partkey is used downstream; trained index owns embeddings.
    auto images = table_source(db, "images", schema("images"), {"i_partkey"}, device);
    // Query side: images joined with important stock to get query embeddings. i_embeddings is necessary...
    auto images_for_queries = table_source(db, "images", schema("images"),
        {"i_partkey", "i_imagekey", "i_embedding", "i_variant"},
        device);


    /* ============================================================
     *      PART 1: TPC-H Q11 CTE (Important Stock)
     * ============================================================
     */

    /* ==============================
     * CREATING A FILTER FOR NATION
     * ==============================
     */
    auto nation_filter = expr(arrow_expr(cp::field_ref("n_name"), "==", string_literal("GERMANY")));
    auto filter_nation_node = filter(nation, nation_filter, device);

    /* ===========================================
     * CREATING A JOIN NODE (PARTSUPP, SUPPLIER)
     * ===========================================
     */
    auto supplier_join_node =
        inner_join(partsupp, supplier, {"ps_suppkey"}, {"s_suppkey"}, "", "", device);

    /* ===========================================
     * CREATING A JOIN NODE (SUPPLIER_JOIN, NATION)
     * ===========================================
     */
    auto joined_filtered = inner_join(
        supplier_join_node, filter_nation_node, {"s_nationkey"}, {"n_nationkey"}, "", "", device);

    /* ===========================================
     * COMPUTING SUPPLY_COST * AVAIL_QTY
     * ===========================================
    */
    // in order to compute sum(ps_supplycost * ps_availqty) * 0.0001000000,
    // we first compute value = ps_supplycost * ps_availqty:
    auto cost_times_qty =
        expr(arrow_product({cp::field_ref("ps_supplycost"), cp::field_ref("ps_availqty")}));
    auto project_node = project(joined_filtered,
                                {expr(cp::field_ref("ps_partkey")), std::move(cost_times_qty)},
                                {"ps_partkey", "value"},
                                device);

    /* ===========================================
     * ADDING A GLOBAL AGGR NODE
     * ===========================================
    */
    // now we compute global_value = sum(value) = sum(ps_supplycost * ps_availqty)
    std::vector<std::shared_ptr<Aggregate>> aggregates = {
        aggregate("sum", "value", "global_value")};
    auto global_aggr_group_by = group_by(project_node, {}, aggregates, device);

    // add another projection to remove all columns except the aggregated value
    auto global_value_project_expr =
        expr(arrow_product({cp::field_ref("global_value"), float64_literal(0.0001)}));
    auto global_aggr =
        project(global_aggr_group_by, {global_value_project_expr}, {"global_value"}, device);

    /* ===========================================
     * CREATING A PS_PARTKEY GROUP BY NODE
     * ===========================================
     */
    auto partsupp_aggregate = aggregate("hash_sum", "value", "value");
    auto partkey_aggr       = group_by(project_node, {"ps_partkey"}, {partsupp_aggregate}, device);

    /* ===========================================
     * CREATING A CROSS JOIN NODE
     * ===========================================
     */
    auto cross_join_node = cross_join(
        partkey_aggr, global_aggr, {"value", "ps_partkey"}, {"global_value"}, "", "", device);

    /* ===========================================
     * CREATING A FILTER FOR CROSS JOIN (HAVING)
     * ===========================================
     */
    auto greater_than_global =
        expr(arrow_expr(cp::field_ref("value"), ">", cp::field_ref("global_value")));
    auto filter_node = filter(cross_join_node, greater_than_global, device);

    /* ===========================================
     * ORDER BY value DESC + LIMIT k
     * (CTE result: top-k important stock parts)
     * ===========================================
     */
    auto order_by_node = order_by(filter_node, {{"value", SortOrder::DESCENDING}}, device);
    auto important_stock = project(order_by_node, {"ps_partkey", "value"}, device);
    // NOTE: The limit is optional, just to "control" the size of the queries
    auto important_stock_limited = limit(important_stock, params.k, 0, device);

    // return query_plan(table_sink(important_stock_limited)); // DEBUG: TPC-H Q11 CTE only


    /* ============================================================
     *      PART 2: Vector Search (kNN duplicate detection)
     *      
     *      SQL equivalent of the LEFT JOIN LATERAL:
     *        - Get the MAIN image embedding for each important stock part
     *        - Batch all query embeddings into one indexed vector search
     *        - Exclude self-matches (i_partkey != query_partkey)
     *        - Keep only the best match per important stock part
     *
     *      Two strategies depending on backend:
     *
     *      GPU (cuDF): Direct inner_join with list<float> columns.
     *        cuDF supports list<> in join non-key fields natively.
     *        1. Filter images for MAIN variant (keep i_embedding)
     *        2. Inner join with important stock on i_partkey = ps_partkey
     *        3. Project → VS query batch
     *
     *      CPU (Acero): Index-Take pattern.
     *        Acero does NOT support list<float> in join non-key fields.
     *        1. Project images WITHOUT i_embedding (drop list<float>)
     *        2. Filter for MAIN, inner_join with important stock
     *           → get qualifying i_imagekeys (safe: no list<float>)
     *        3. Use take() to gather i_embedding for matching imagekeys
     *           (bypasses Acero hash join via arrow::compute::Take)
     *        4. Project → VS query batch
     * ============================================================
     */

    // The query batch (query_partkey, i_embedding, value) is built differently
    // depending on the backend, but produces the same schema either way.
    std::shared_ptr<QueryNode> vs_query_batch;

    if (device == DeviceType::GPU) {
        /* ===========================================
         * GPU path: Direct join (cuDF handles list<float>)
         * ===========================================
         */
        // Filter MAIN images — keep i_embedding (cuDF can join with it)

        auto main_images = project(
            filter(images_for_queries,
                   expr(arrow_expr(cp::field_ref("i_variant"), "==", string_literal("MAIN"))),
                   device),
            {"i_partkey", "i_imagekey", "i_embedding"}, device);
        // Inner join with important stock: cuDF supports list<float> in non-key fields
        auto qualifying_with_emb = inner_join(
            main_images, important_stock_limited,
            {"i_partkey"}, {"ps_partkey"}, "", "", device);
        // qualifying_with_emb has: i_partkey, i_imagekey, i_embedding, value

        // Rename i_partkey → query_partkey to avoid collision with data-side i_partkey
        vs_query_batch = project(qualifying_with_emb,
            {expr(cp::field_ref("i_partkey")), expr(cp::field_ref("i_embedding")),
             expr(cp::field_ref("value"))},
            {"query_partkey", "i_embedding", "value"}, device);

    } else {
        /* ===========================================
         * CPU path: Index-Take pattern
         * (Acero cannot join with list<float> non-key fields)
         * ===========================================
         */

        /* -------------------------------------------
         * STEP 1: Get qualifying imagekeys via join
         * (without i_embedding — safe for Acero join)
         * -------------------------------------------
         */
        // Project images WITHOUT the embedding column
        auto main_images_no_emb = project(
            filter(images_for_queries,
                   expr(arrow_expr(cp::field_ref("i_variant"), "==", string_literal("MAIN"))),
                   device),
            {"i_partkey", "i_imagekey"}, device);

        // Inner join with important stock to get only the MAIN images for important parts
        // This is safe because there's no list<float> column in either side
        auto qualifying = inner_join(
            main_images_no_emb, important_stock_limited,
            {"i_partkey"}, {"ps_partkey"}, "", "", device);
        // qualifying has: i_partkey, i_imagekey, value

        /* -------------------------------------------
         * STEP 2: Gather embeddings via take()
         * (bypasses Acero hash join for list<float>)
         * -------------------------------------------
         */
        // Data side: full images table with embeddings, keyed by i_imagekey
        auto images_with_emb = table_source(
            db, "images", schema("images"), {"i_imagekey", "i_partkey", "i_embedding"}, device);

        // Index side: qualifying imagekeys from the join above
        auto qualifying_keys = project(qualifying,
            {expr(cp::field_ref("i_imagekey")), expr(cp::field_ref("value"))},
            {"i_imagekey", "value"}, device);

        // take(): gather rows from images_with_emb where i_imagekey matches qualifying_keys
        // Output: i_imagekey, i_partkey, i_embedding, value
        auto query_images_with_emb = take(
            images_with_emb, qualifying_keys, "i_imagekey", "i_imagekey", DeviceType::CPU);

        /* -------------------------------------------
         * STEP 3: Prepare VS query batch
         * -------------------------------------------
         */
        // Rename i_partkey → query_partkey to avoid collision with data-side i_partkey
        vs_query_batch = project(query_images_with_emb,
            {expr(cp::field_ref("i_partkey")), expr(cp::field_ref("i_embedding")),
             expr(cp::field_ref("value"))},
            {"query_partkey", "i_embedding", "value"}, device);
    }

    // return query_plan(table_sink(vs_query_batch)); // DEBUG: query batch

    /* ===========================================
     * INDEXED VECTOR SEARCH
     * ===========================================
     */
    // For each important part's MAIN image, find the k closest images
    // across the entire images table (via the pre-built index).
    auto images_search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

    auto images_vs_node = indexed_vector_join(
        images,            // data: all images (embeddings owned by index)
        vs_query_batch,    // query: MAIN images of important stock parts
        "",                // data vector column ("" = embeddings already in the index)
        "i_embedding",     // query vector column (from projecting images_for_queries)
        index,
        10, // get the "10" nearest neighbours, and then drop "self" matches in post-filter
        std::nullopt,
        images_search_params,
        false,             // don't keep data vectors
        false,             // don't keep query vectors
        "vs_distance",
        params.vs_device
    );

    // return query_plan(table_sink(images_vs_node)); // DEBUG: raw VS results

    /* ===========================================
     * POST-FILTER: Exclude self-matches
     * ===========================================
     */
    // SQL: data_img.i_partkey != tpch_out.ps_partkey
    // After VS: data-side has `i_partkey`, query-side has `query_partkey`
    auto exclude_self = expr(arrow_expr(
        cp::field_ref("i_partkey"), "!=", cp::field_ref("query_partkey")));
    auto vs_filtered = filter(images_vs_node, exclude_self, device);

    /* ===========================================
     * PER-QUERY LIMIT 1: best match per part
     * ===========================================
     */
    // SQL: ORDER BY distance LIMIT 1 (inside the lateral join)
    // Partition by query_partkey, order by vs_distance ASC, take top-1 per partition


    // NOTE: Since we have STATIC DAG provisioning. We can set "k" number of partitions where "k"
    // where k is a number higher than what we need. Altough we have limit k above to ensure upper bound
    // but even if we did not have the optinoal limit k, it would still work for high k.
    // SOLUTION: Some sort of DYNAMIC DAG creation (defered dag construction), which we don't have rn.
    // There is some performance you lose if k >> actual number of partitions you need
    SortOrder vs_distance_order = (params.metric == VectorDistanceMetric::INNER_PRODUCT) ? SortOrder::DESCENDING : SortOrder::ASCENDING;
    std::shared_ptr<QueryNode> best_per_part;
    if (params.use_limit_per_group) {
        best_per_part = limit_per_group(vs_filtered, "query_partkey", 1, device);
    } else {
        best_per_part = order_limit_per_partition(
            vs_filtered, "query_partkey", params.k, 1, device,
            {SortKey("vs_distance", vs_distance_order)});
    }

    // No need to join back with important_stock_limited — `value` was carried
    // through the query batch (via take → project → VS query side).

    /* ===========================================
     * FINAL PROJECTION (matching SQL output)
     * ===========================================
     */
    auto final_project = project(best_per_part,
        {expr(cp::field_ref("query_partkey")),
         expr(cp::field_ref("value")),
         expr(cp::field_ref("i_partkey")),
         expr(cp::field_ref("vs_distance"))},
        {"ps_partkey", "value", "duplicate_partkey", "visual_distance"},
        device);

    /* ===========================================
     * FINAL ORDER BY
     * ===========================================
     */
    // SQL: ORDER BY dup.dist, tpch_out.value DESC
    auto final_order = order_by(final_project,
        {{"visual_distance", vs_distance_order},
         {"value", SortOrder::DESCENDING}},
        device);

    return query_plan(table_sink(final_order));
}



// TODO: Improve the support for thies query. 
// NOTE:
// - This is a difficult query to properly support at the moment.
// - on CPU it's all good.
// - on GPU, you need to use postfilter (due to large list)
// - but when using postfilter --> you need super large postfilter_ksearch --> so you take the hit on the Recall
// - AND on larger datasets, you can't increase postiflter_ksearch indefinitely you get another error
std::shared_ptr<QueryPlan> q15_end(std::shared_ptr<Database>& db, IndexPtr index, DeviceType device, const QueryParameters& params) {

    // single query at a time only
    if (params.query_count != 1) {
        throw std::runtime_error("Only one query is supported at a time (query_count must be 1)");
    }

    /* ==============================
     * Vector Search Tables
     * ==============================
     */

    // Data side: reviews table (index owns embeddings, load metadata columns for post-filter + final output)
    auto reviews = table_source(db, "reviews", schema("reviews"), {"rv_reviewkey", "rv_partkey", "rv_text"}, device);

    // Query side: query embeddings for semantic search
    auto vs_query_source_reviews = sliced_query_source(
        db,
        "reviews_queries",
        schema("reviews_queries"),
        {"rv_reviewkey_queries", "rv_embedding_queries"},
        device,
        params.query_start,
        params.query_count
    );

    /* ============================================================
     *      PART 1: TPC-H Q15 (Top Revenue Supplier)
     * ============================================================
     */

    auto lineitem = table_source(db, "lineitem", schema("lineitem"), {}, device);
    auto supplier = table_source(db, "supplier", schema("supplier"), {}, device);

    auto lineitem_filter   = expr(arrow_in_range(
        cp::field_ref("l_shipdate"), date_literal("1996-01-01"), date_literal("1996-04-01")));
    auto filtered_lineitem = filter(lineitem, lineitem_filter, device);

    auto lineitem_exprs = exprs({"l_suppkey"});
    lineitem_exprs.push_back(
        expr(arrow_expr(cp::field_ref("l_extendedprice"),
                        "*",
                        arrow_expr(float64_literal(1.0), "-", cp::field_ref("l_discount")))));
    auto lineitem_project =
        project(filtered_lineitem, lineitem_exprs, {"l_suppkey", "revenue"}, device);

    auto revenue_by_supplier = group_by(lineitem_project,
                                        {"l_suppkey"},
                                        {aggregate("hash_sum", "revenue", "total_revenue")},
                                        device);

    auto global_revenue = group_by(
        revenue_by_supplier, {}, {aggregate("max", "total_revenue", "max_total_revenue")}, device);

    auto joined = cross_join(revenue_by_supplier,
                             global_revenue,
                             {"l_suppkey", "total_revenue"},
                             {"max_total_revenue"},
                             "",
                             "",
                             device);

    auto revenue_filter = expr(
        arrow_expr(arrow_abs(arrow_expr(
                       cp::field_ref("total_revenue"), "-", cp::field_ref("max_total_revenue"))),
                   "<",
                   float64_literal(1e-9)));

    auto high_revenue = filter(joined, revenue_filter, device);

    auto revenue_supplier =
        inner_join(high_revenue, supplier, {"l_suppkey"}, {"s_suppkey"}, "", "", device);

    auto tpch_project = project(revenue_supplier,
        exprs({"l_suppkey", "s_name", "s_address", "s_phone", "total_revenue"}),
        {"s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"}, device);

    // return query_plan(table_sink(tpch_project)); // DEBUG: TPC-H Q15 only

    /* ============================================================
     *      PART 2: Find parts supplied by the top supplier
     *      JOIN partsupp ON s_suppkey = ps_suppkey
     *      JOIN part     ON ps_partkey = p_partkey
     * ============================================================
     */

    auto partsupp = table_source(db, "partsupp", schema("partsupp"), {}, device);
    auto part     = table_source(db, "part", schema("part"), {}, device);

    // Get parts supplied by the top supplier
    auto supplier_parts = inner_join(tpch_project, partsupp,
        {"s_suppkey"}, {"ps_suppkey"}, "", "", device);

    auto supplier_parts_with_names = inner_join(supplier_parts, part,
        {"ps_partkey"}, {"p_partkey"}, "", "", device);

    // Project to get partkeys + supplier info (carry through for final output)
    auto top_supplier_partkeys = project(supplier_parts_with_names,
        exprs({"s_suppkey", "s_name", "p_name", "ps_partkey"}),
        {"s_suppkey", "s_name", "p_name", "ps_partkey"}, device);

    // return query_plan(table_sink(top_supplier_partkeys)); // DEBUG: top supplier parts

    /* ============================================================
     *      PART 3: Vector Search on reviews
     * ============================================================
     */


    if (!params.use_post) {
        //  ==========================
        //          PRE FILTER : for large data --> it is CPU ONLY (large_list<float> issue in cuDF). 
        //                        -//-          --> if you want GPU--> use postfilter for huge data
        //  ==========================
        //      NOTE: prefilter is GROUND TRUTH but slower (exhaustive search on filtered data)
        //      CPU-ONLY: reviews uses large_list<float> for rv_embedding, which cuDF cannot process
        
        if (params.vs_device == DeviceType::CPU) {
            // prefilter="Ground Truth" to compare to PG ENN. Since postfilter if postfilter_ksearch not large enough will have recall < 1.
            // Step 1: Get qualifying reviewkeys via join (without rv_embedding — safe for Acero join)
            //         Acero cannot hash-join with list<float> in non-key fields,
            //         so we join without embeddings first, then use take() to gather them.
            auto reviews_no_emb = table_source(db, "reviews", schema("reviews"),
                {"rv_reviewkey", "rv_partkey"}, device);

            auto qualifying_reviews = inner_join(reviews_no_emb, top_supplier_partkeys,
                {"rv_partkey"}, {"ps_partkey"}, "", "", device);
            // qualifying_reviews has: rv_reviewkey, rv_partkey, s_suppkey, s_name, p_name

            auto qualifying_keys = project(qualifying_reviews, {"rv_reviewkey"}, device);

            // Step 2: Gather embeddings via take() (bypasses Acero hash join for list<float>)
            // rv_embedding is large_list<float> → must live on CPU (cuDF rejects large_list at load).
            auto reviews_with_emb = table_source(db, "reviews", schema("reviews"),
                {"rv_reviewkey", "rv_partkey", "rv_embedding", "rv_text"}, DeviceType::CPU);
            // TODO: Fix Take() performance. OOM in sf=1. Either remove Take() or figure out another way to do join on acero w/ list<float>
            auto filtered_reviews_with_emb = take(
                reviews_with_emb, qualifying_keys, "rv_reviewkey", "rv_reviewkey", DeviceType::CPU);
            // filtered_reviews_with_emb has: rv_reviewkey, rv_partkey, rv_embedding, rv_text

            // return query_plan(table_sink(project(filtered_reviews_with_emb, {"rv_reviewkey", "rv_partkey"}, device))); // DEBUG

            // Step 3: Exhaustive vector search on filtered reviews
            auto vs_node = exhaustive_vector_join(
                filtered_reviews_with_emb,
                vs_query_source_reviews,
                "rv_embedding",
                "rv_embedding_queries",
                params.metric,
                params.k,
                std::nullopt,
                false,
                false,
                "distance",
                params.vs_device
            );

            // Step 4: Join with top_supplier_partkeys for s_name, p_name
            auto vs_with_info = inner_join(vs_node, top_supplier_partkeys,
                {"rv_partkey"}, {"ps_partkey"}, "", "", device);

            auto final_project = project(vs_with_info,
                {expr(cp::field_ref("rv_reviewkey")),
                expr(cp::field_ref("s_suppkey")),
                expr(cp::field_ref("s_name")),
                expr(cp::field_ref("p_name")),
                expr(cp::field_ref("distance")),
                expr(cp::field_ref("rv_text"))},
                {"rv_reviewkey", "s_suppkey", "s_name", "part_name", "vs_distance", "rv_text"},
                device);

            return query_plan(table_sink(final_project));
        } else {
            throw std::runtime_error("q15_end prefilter requires vs_device=CPU (rv_embedding is large_list<float>, unsupported by cuDF)");
        }
    } else {
        // ==========================
        //         POST FILTER (alternative)
        // ==========================
        //     NOTE: postfilter is faster (uses index) but NOT ground truth (recall hit)
        //         Works on both CPU and GPU.

        auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

        // Search more broadly to account for post-filter pruning
        auto vs_node = indexed_vector_join(
            reviews,                    // data: all reviews (index owns embeddings)
            vs_query_source_reviews,    // query: query embeddings
            "",                         // data vector column ("" = index owns the data)
            "rv_embedding_queries",     // query vector column
            index,
            params.postfilter_ksearch,  // since we "filter" by partkey after
            std::nullopt,
            search_params,
            false,                      // don't keep data vectors
            false,                      // don't keep query vectors
            "vs_distance",
            params.vs_device
        );

        // return query_plan(table_sink(vs_node)); // DEBUG: raw VS results

        // Post-filter: keep only reviews for the top supplier's parts
        auto vs_filtered = inner_join(vs_node, top_supplier_partkeys,
            {"rv_partkey"}, {"ps_partkey"}, "", "", device);

        auto final_project = project(vs_filtered,
            {expr(cp::field_ref("rv_reviewkey")),
            expr(cp::field_ref("s_suppkey")),
            expr(cp::field_ref("s_name")),
            expr(cp::field_ref("p_name")),
            expr(cp::field_ref("vs_distance")),
            expr(cp::field_ref("rv_text"))},
            {"rv_reviewkey", "s_suppkey", "s_name", "part_name", "vs_distance", "rv_text"},
            device);

        SortOrder vs_distance_order = (params.metric == VectorDistanceMetric::INNER_PRODUCT) ? SortOrder::DESCENDING : SortOrder::ASCENDING;
        auto final_ordered = order_by(final_project, {SortKey("vs_distance", vs_distance_order)}, device);

        // Final LIMIT {k} (since we did postfilter_ksearch)
        auto final_limit = limit(final_ordered, params.k, 0, device);

        return query_plan(table_sink(final_limit));
    }
}


std::shared_ptr<QueryPlan> q13_mid(std::shared_ptr<Database>& db, IndexPtr index, DeviceType device, const QueryParameters& params) {

    // single query at a time only
    if (params.query_count != 1) {
        throw std::runtime_error("Only one query is supported at a time (query_count must be 1)");
    }

    /* ==============================
     *  Tables
     * ==============================
     */

    auto reviews = table_source(db, "reviews", schema("reviews"), {"rv_custkey"}, device);

    auto vs_query_source_reviews = sliced_query_source(
        db,
        "reviews_queries",
        schema("reviews_queries"),
        {"rv_embedding_queries"},
        device,
        params.query_start,
        params.query_count
    );


    auto customer = table_source(db, "customer", schema("customer"), {}, device);
    auto orders   = table_source(db, "orders", schema("orders"), {}, device);

    /* ============================================================
     *      Branch 1: Vector Search (independent)
     *      Find global Top-K reviews most similar to the input vector
     * ============================================================
     */

    auto search_params = make_search_parameters(params.faiss_index, params, params.vs_device);

    auto vector_search_node = indexed_vector_join(
        reviews,
        vs_query_source_reviews,
        "", // data is already in the index for rv_embedding column
        "rv_embedding_queries",
        index,
        params.k,
        std::nullopt,
        search_params,
        false,
        false,
        "vs_distance",
        params.vs_device
    );

    // Pre-aggregate: count how many top-K reviews belong to each customer
    // This avoids cross-product issues if we joined at the row level
    auto vs_custkeys = project(vector_search_node, {"rv_custkey"}, device);
    auto vs_review_counts = group_by(vs_custkeys,
        {"rv_custkey"},
        {aggregate("hash_count", count_all(), "rv_custkey", "review_match_count")},
        device);

    // return query_plan(table_sink(vs_review_counts)); // DEBUG

    /* ============================================================
     *      Branch 2: TPC-H Q13 (Customer Order Distribution)
     * ============================================================
     */
    // auto special_orders_filter = expr(arrow_field_like("o_comment", ".*special.*requests.*"));
    auto special_orders_filter =
        expr(arrow_not(arrow_field_like("o_comment", "%special%requests%")));

    auto special_orders = filter(orders, special_orders_filter, device);

    // here we use outer join (instead of left_anti_join) since we also want to include the columns from orders
    // e.g. o_orderkey is later used in group_by
    auto co_joined =
        left_outer_join(customer, special_orders, {"c_custkey"}, {"o_custkey"}, "", "", device);

    auto c_orders = group_by(co_joined,
                             {"c_custkey"},
                             {aggregate("hash_count", count_valid(), "o_orderkey", "c_count")},
                             device);

    // return query_plan(table_sink(c_orders)); // DEBUG

    /* ============================================================
     *      Merge: LEFT JOIN per-customer orders with VS review counts
     * ============================================================
     */

    auto merged = left_outer_join(c_orders, vs_review_counts,
        {"c_custkey"}, {"rv_custkey"}, "", "", device);

    // return query_plan(table_sink(merged)); // DEBUG

    auto merged_projected = project(merged,
        {expr(cp::field_ref("c_count")), expr(cp::field_ref("review_match_count"))},
        {"c_count", "review_match_count"}, device);

    // return query_plan(table_sink(merged_projected)); // DEBUG
    /* ============================================================
     *      Outer GROUP BY: distribution by c_count
     *      custdist    = COUNT(*)                 — how many customers have this order count
     *      reviewdist  = SUM(review_match_count)  — total top-K reviews in this bucket
     * ============================================================
     */

    auto c_orders_grouped = group_by(merged_projected,
        {"c_count"},
        {aggregate("hash_count", count_all(), "c_count", "custdist"),
         aggregate("hash_sum", "review_match_count", "reviewdist")},
        device);

    auto c_orders_sorted =
        order_by(c_orders_grouped,
                 {{"custdist", SortOrder::DESCENDING}, {"c_count", SortOrder::DESCENDING}},
                 device);

    return query_plan(table_sink(c_orders_sorted));
}


std::shared_ptr<QueryPlan> q1_start(std::shared_ptr<Database>& db, IndexPtr index, DeviceType device, const QueryParameters& params) {
    // NOTE: I say "vs@start" even though we do the lineitem filtering first in this query plan
    // but we could've just as well dont vector search first. Similar to pre/postjoin. 
    // What it does:
    // query_count = number of classes (not number of user queries).
    // rv_reviewkey_queries acts as the class identifier.
    // Each review is classified to its nearest class (K=1 exhaustive search).
    // Then parts are classified by majority vote of their reviews.

    // NOTE: rv_embedding is large_list<float> → CPU only for exhaustive_vector_join
    // (cuDF cannot process large_list<float>)
    if (device != DeviceType::CPU) {
        throw std::runtime_error("q1_start is currently only supported on CPU (reviews use large_list<float>).");
    }

    /* ============================================================
     *      Filter lineitem
     * ============================================================
     */

    auto source_node = table_source(db,
                                    "lineitem",
                                    schema("lineitem"),
                                    {"l_partkey",
                                     "l_shipdate",
                                     "l_returnflag",
                                     "l_linestatus",
                                     "l_quantity",
                                     "l_extendedprice",
                                     "l_discount",
                                     "l_tax"},
                                    device);

    auto sept_2_1998 = date_literal("1998-09-02");
    auto filter_expr = expr(arrow_expr(cp::field_ref("l_shipdate"), "<=", sept_2_1998));
    auto filter_node = filter(source_node, filter_expr, device);

    // Get the set of partkeys that survive the date filter
    auto qualifying_partkeys = distinct(project(filter_node, {"l_partkey"}, device), {"l_partkey"}, device);

    // return query_plan(table_sink(qualifying_partkeys)); // DEBUG

    /* ============================================================
     *      Step 2: Prefilter reviews to only qualifying parts,
     *              then classify via exhaustive vector search (K=1)
     *
     *      Uses Index-Take pattern because rv_embedding is large_list<float>,
     *      which Acero cannot handle as a payload column in joins.
     *      1. Semi-join reviews (WITHOUT rv_embedding) to get qualifying rv_reviewkeys
     *      2. take() to gather rv_embedding for those reviewkeys
     *      3. Exhaustive vector search on the result
     * ============================================================
     */

    // Step 2a: Semi-join WITHOUT embeddings (safe for Acero — no list<float> payload)
    auto reviews_no_emb = table_source(db, "reviews", schema("reviews"),
        {"rv_reviewkey", "rv_partkey"}, device);

    auto qualifying_reviews = left_semi_join(reviews_no_emb, qualifying_partkeys,
        {"rv_partkey"}, {"l_partkey"}, "", "", device);

    auto qualifying_reviewkeys = project(qualifying_reviews, {"rv_reviewkey"}, device);

    // return query_plan(table_sink(qualifying_reviewkeys)); // DEBUG

    // Step 2b: Gather embeddings via take() (bypasses Acero hash join for large_list<float>)
    auto reviews_with_emb = table_source(db, "reviews", schema("reviews"),
        {"rv_reviewkey", "rv_partkey", "rv_embedding"}, device);

    auto relevant_reviews = take(
        reviews_with_emb, qualifying_reviewkeys,
        "rv_reviewkey", "rv_reviewkey", DeviceType::CPU);
    // Output: rv_reviewkey, rv_partkey, rv_embedding (only for qualifying parts)

    // return query_plan(table_sink(project(relevant_reviews, {"rv_reviewkey", "rv_partkey"}, device))); // DEBUG

    // Class embeddings: data side (small, query_count entries)
    // rv_reviewkey_queries acts as the class identifier
    auto class_embeddings = sliced_query_source(db,
        "reviews_queries", schema("reviews_queries"),
        {"rv_reviewkey_queries", "rv_embedding_queries"},
        device,
        params.query_start,
        params.query_count);

    // Exhaustive vector join: K=1, each review → its single nearest class
    // data = class_embeddings (small), query = relevant_reviews (prefiltered via take)
    // NOTE: No index — class embeddings are tiny, brute-force is efficient
    auto vs_node = exhaustive_vector_join(
        class_embeddings,       // data (class embeddings)
        relevant_reviews,       // query (prefiltered reviews)
        "rv_embedding_queries", // data vector column
        "rv_embedding",         // query vector column
        params.metric,
        1,                      // K=1: each review gets exactly one class
        std::nullopt,           // no radius
        false,                  // don't keep data vector column
        false,                  // don't keep query vector column
        "vs_distance",             // distance column name
        params.vs_device);
    // Output: rv_partkey, rv_reviewkey_queries (class ID), distance

    /* ============================================================
     *      Step 3: Classification aggregation — majority vote per part
     * ============================================================
     */

    // Step 3a: Group by (rv_partkey, rv_reviewkey_queries) → frequency, avg_dist
    //          How many reviews of this part voted for this class?
    auto classification_grouped = group_by(vs_node,
        {"rv_partkey", "rv_reviewkey_queries"},
        {aggregate("hash_count", count_all(), "rv_partkey", "frequency"),
         aggregate("hash_mean", sum_defaults(), "vs_distance", "avg_dist")},
        device);

    // Step 3b: Get max frequency per part (which class got the most votes?)
    auto max_freq_per_part = group_by(classification_grouped,
        {"rv_partkey"},
        {aggregate("hash_max", "frequency", "max_freq")},
        device);

    // Step 3c: Self-join to keep only the class(es) with the winning frequency per part.
    //          Same pattern as Q2's min_supplycost self-join.
    auto best_freq_classes = inner_join(
        classification_grouped, max_freq_per_part,
        {"rv_partkey", "frequency"}, {"rv_partkey", "max_freq"},
        "_l", "_r", device);
    // After join: rv_partkey_l, rv_reviewkey_queries, frequency, avg_dist, rv_partkey_r, max_freq

    // Step 3d: Pick one winning class per part.
    //          hash_one: picks an arbitrary value (parallel-safe, unlike hash_first).
    //          If multiple classes tie on frequency, one is chosen arbitrarily — acceptable
    //          for classification since ties are rare and both are equally valid.
    auto winning_class_raw = group_by(best_freq_classes,
        {"rv_partkey_l"},
        {aggregate("hash_one", "rv_reviewkey_queries", "rv_reviewkey_queries"),
         aggregate("hash_one", "avg_dist", "avg_dist")},
        device);

    // Rename rv_partkey_l back to rv_partkey and rv_reviewkey_queries to class_name for downstream joins
    auto winning_class = project(winning_class_raw,
        {expr(cp::field_ref("rv_partkey_l")),
         expr(cp::field_ref("rv_reviewkey_queries")),
         expr(cp::field_ref("avg_dist"))},
        {"rv_partkey", "class_name", "avg_dist"}, device);
    // Output: rv_partkey, class_name (winning class), avg_dist

    // return query_plan(table_sink(winning_class)); // DEBUG: winning class per part

    /* ============================================================
     *      Step 4: LEFT JOIN filtered lineitem with winning class per part
     *      Parts without reviews → NULL class (separate group in final agg)
     * ============================================================
     */

    auto merged = left_outer_join(filter_node, winning_class,
        {"l_partkey"}, {"rv_partkey"}, "", "", device);

    // return query_plan(table_sink(merged)); // DEBUG: merged

    /* ============================================================
     *      PROJECT: Compute derived columns
     *      disc_price = l_extendedprice * (1 - l_discount)
     *      charge     = disc_price * (1 + l_tax)
     * ============================================================
     */

    auto _one                = float64_literal(1.00);
    auto _one_minus_discount = arrow_expr(_one, "-", cp::field_ref("l_discount"));
    auto _one_plus_tax       = arrow_expr(_one, "+", cp::field_ref("l_tax"));
    auto _disc_price = arrow_expr(cp::field_ref("l_extendedprice"), "*", _one_minus_discount);
    auto _charge     = arrow_product({_disc_price, _one_plus_tax});

    auto l_returnflag         = Expression::from_field_ref("l_returnflag");
    auto l_linestatus         = Expression::from_field_ref("l_linestatus");
    auto class_name           = Expression::from_field_ref("class_name");
    auto quantity             = Expression::from_field_ref("l_quantity");
    auto base_price           = Expression::from_field_ref("l_extendedprice");
    auto discount             = Expression::from_field_ref("l_discount");
    auto charge               = expr(_charge);
    auto disc_price           = expr(_disc_price);
    auto avg_dist_expr        = Expression::from_field_ref("avg_dist");

    std::vector<std::shared_ptr<Expression>> projection_list = {
        l_returnflag,
        l_linestatus,
        class_name,
        quantity,
        base_price,
        disc_price,
        charge,
        quantity,
        base_price,
        discount,
        avg_dist_expr};

    std::vector<std::string> project_names = {
        "l_returnflag",
        "l_linestatus",
        "class_name",
        "sum_qty",
        "sum_base_price",
        "sum_disc_price",
        "sum_charge",
        "avg_qty",
        "avg_price",
        "avg_disc",
        "avg_semantic_dist"};

    auto project_node = project(merged, projection_list, project_names, device);

    /* ============================================================
     *      GROUP BY: TPC-H Q1 aggregates + class dimension
     *      Groups by (l_returnflag, l_linestatus, class_name)
     *      class_name is the class ID (or NULL for unclassified parts)
     * ============================================================
     */

    auto sum_opts   = sum_defaults();
    auto count_opts = count_all();
    std::vector<std::shared_ptr<Aggregate>> aggs = {
        aggregate("hash_sum", sum_opts, "sum_qty", "sum_qty"),
        aggregate("hash_sum", sum_opts, "sum_base_price", "sum_base_price"),
        aggregate("hash_sum", sum_opts, "sum_disc_price", "sum_disc_price"),
        aggregate("hash_sum", sum_opts, "sum_charge", "sum_charge"),
        aggregate("hash_mean", sum_opts, "avg_qty", "avg_qty"),
        aggregate("hash_mean", sum_opts, "avg_price", "avg_price"),
        aggregate("hash_mean", sum_opts, "avg_disc", "avg_disc"),
        aggregate("hash_count", count_opts, "l_returnflag", "count_order"),
        aggregate("hash_mean", sum_opts, "avg_semantic_dist", "avg_semantic_dist")};
    auto group_by_node = group_by(project_node,
        {"l_returnflag", "l_linestatus", "class_name"}, aggs, device);

    /* ============================================================
     *      ORDER BY: l_returnflag, l_linestatus, class_name
     * ============================================================
     */

    std::vector<SortKey> sort_keys = {
        {"l_returnflag"}, {"l_linestatus"}, {"class_name"}};
    auto order_by_node = order_by(group_by_node, sort_keys, device);

    return query_plan(table_sink(order_by_node));

}




// ============================================================================
// Query Dispatcher
// ============================================================================

std::shared_ptr<QueryPlan> query_plan(const std::string& q,
                                      std::shared_ptr<Database>& db,
                                      DeviceType device,
                                      const std::string& params_str,
                                      const IndexMap& indexes,
                                      const std::string& index_desc,
                                      DeviceType vs_device) {
    // Parse parameters
    QueryParameters params = parse_query_parameters(params_str);

    // Set faiss_index from index_desc (passed separately to avoid comma parsing issues)
    if (!index_desc.empty()) {
        params.faiss_index = index_desc;
    }

    // Device for vector-search operators (may differ from outer `device` — Case 4).
    params.vs_device = vs_device;

    // Dispatch to specific query implementation
    
    // ENN queries
    if (q == "enn_reviews" || q == "enn") {
        return enn_reviews(db, device, params);
    }
    if (q == "enn_reviews_project_distance") {
        return enn_reviews_project_distance(db, device, params);
    }
    if (q == "enn_images") {
        return enn_images(db, device, params);
    }
    if (q == "enn_images_project_distance") {
        return enn_images_project_distance(db, device, params);
    }
    
    // For index-requiring queries, look up the index(es) from the map
    auto index_keys = get_query_index_requirements(q);
    IndexPtr index = nullptr;
    if (!index_keys.empty()) {
        auto it = indexes.find(index_keys[0]);
        if (it != indexes.end()) {
            index = it->second;
        }
    }
    
    // ANN queries
    if (q == "ann_reviews" || q == "ann") {
        return ann_reviews(db, index, device, params);
    }
    if (q == "ann_images") {
        return ann_images(db, index, device, params);
    }
    if (q == "ann_reviews_full") {
        return ann_reviews_full(db, index, device, params);
    }
    if (q == "ann_images_full") {
        return ann_images_full(db, index, device, params);
    }
    if (q == "ann_ranged_reviews") {
        return ann_ranged_reviews(db, index, device, params);
    }
    if (q == "ann_ranged_images") {
        return ann_ranged_images(db, index, device, params);
    }
    
    // Pre-filtering queries
    if (q == "pre_reviews" || q == "pre") {
        return pre_reviews(db, device, params);
    }
    if (q == "pre_reviews_hybrid") {
        return pre_reviews_hybrid(db, params);
    }
    if (q == "pre_images") {
        return pre_images(db, device, params);
    }
    if (q == "pre_images_hybrid") {
        return pre_images_hybrid(db, params);
    }
    if (q == "pre_reviews_partitioned") {
        return pre_reviews_partitioned(db, device, params);
    }

    // Post-filtering queries
    if (q == "post_reviews" || q == "post") {
        return post_reviews(db, index, device, params);
    }
    if (q == "post_reviews_hybrid") {
        return post_reviews_hybrid(db, index, params);
    }
    if (q == "post_reviews_partitioned") {
        return post_reviews_partitioned(db, index, device, params);
    }
    if (q == "post_reviews_filter_partitioned") {
        return post_reviews_filter_partitioned(db, index, device, params);
    }
    if (q == "post_images") {
        return post_images(db, index, device, params);
    }
    if (q == "post_images_hybrid") {
        return post_images_hybrid(db, index, params);
    }
    if (q == "post_images_partitioned") {
        return post_images_partitioned(db, index, device, params);
    }

    // Join + filter queries (reviews only for now)
    if (q == "prejoin_reviews" || q == "prejoin") {
        return prejoin_reviews(db, device, params);
    }
    if (q == "postjoin_reviews" || q == "postjoin") {
        return postjoin_reviews(db, index, device, params);
    }

    // Bitmap-filtering queries
    if (q == "pre_bitmap_ann_reviews") {
        return pre_bitmap_ann_reviews(db, index, device, params);
    }
    if (q == "pre_bitmap_ann_images") {
        return pre_bitmap_ann_images(db, index, device, params);
    }

    // Expr-filtering queries
    if (q == "inline_expr_ann_reviews") {
        return inline_expr_ann_reviews(db, index, device, params);
    }
    if (q == "inline_expr_ann_images") {
        return inline_expr_ann_images(db, index, device, params);
    }

    // VSDS queries
    if (q == "q1_start" || q == "q1") {
        return q1_start(db, index, device, params);
    }

    if (q == "q2_start" || q == "q2") {
        return q2_start(db, index, device, params);
    }
    
    if (q == "q10_mid" || q == "q10") {
        return q10_mid(db, index, device, params);
    }
    
    if (q == "q11_end" || q == "q11") {
        return q11_end(db, index, device, params);
    }

    if (q == "q13_mid" || q == "q13") {
        return q13_mid(db, index, device, params);
    }

    if (q == "q15_end" || q == "q15") {
        return q15_end(db, index, device, params);
    }

    
    if (q == "q16_start" || q == "q16") {
        return q16_start(db, index, device, params);
    }

    if (q == "q19_start" || q == "q19") {
        // q19 needs two indexes: reviews and images
        IndexPtr reviews_index = nullptr;
        IndexPtr images_index = nullptr;
        for (const auto& key : index_keys) {
            auto it = indexes.find(key);
            if (it != indexes.end()) {
                if (key.find("reviews") != std::string::npos) {
                    reviews_index = it->second;
                } else if (key.find("images") != std::string::npos) {
                    images_index = it->second;
                }
            }
        }
        return q19_start(db, reviews_index, images_index, device, params);
    }

    if (q == "q18_mid" || q == "q18") {
        return q18_mid(db, index, device, params);
    }

    throw std::runtime_error("Non-existing vsds query: " + q);
}

}  // namespace maximus::vsds