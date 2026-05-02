#pragma once

#include <maximus/dag/query_node.hpp>
#include <maximus/dag/query_plan.hpp>
#include <maximus/database.hpp>
#include <maximus/indexes/index.hpp>
#include <maximus/types/device_table_ptr.hpp>
#include <maximus/types/types.hpp>

#include <optional>
#include <unordered_map>

namespace maximus {

// Index registry: maps "table.column" -> IndexPtr
using IndexMap = std::unordered_map<std::string, IndexPtr>;

namespace vsds {

// Query parameters for index configuration
// The faiss_index string determines which params below are actually used
struct QueryParameters {
    // Index description - determines which search params apply
    // Examples: "Flat", "HNSW32", "IVF256,Flat", "GPU,Cagra", "GPU,IVF256,Flat"
    std::string faiss_index = "Flat";

    // ========== HNSW (CPU) ==========
    // Used when: faiss_index starts with "HNSW"
    int hnsw_efsearch = 64;  // Higher = better recall, slower (default: 16-64)

    // ========== IVF / IVFPQ (CPU/GPU) ==========
    // Used when: faiss_index contains "IVF"
    int ivf_nprobe = 10;  // Number of clusters to search (higher = better recall)

    // ========== Cagra (GPU only) ==========
    // Used when: faiss_index starts with "GPU,Cagra"
    int cagra_itopksize   = 128;  // Internal top-k buffer size - must be >= (postfilter_ksearch, k)
    int cagra_searchwidth = 1;   // Graph search width
    // Dataset location: -1=auto (follow index_storage_device), 0=keep on host, 1=copy to GPU.
    // Architecture-dependent behavior:
    //   ATS/unified memory (GH200, DGX-Spark): data=0 keeps dataset on host (non-owning view,
    //     GPU reads via NVLink-C2C / ATS page walks). data=1 forces explicit cudaMalloc+copy
    //     for best search perf (HBM bandwidth). This is the only architecture where the
    //     distinction matters — non-owning host views are only possible with ATS hardware.
    //   Discrete GPU (x86 dGPU): cuVS always copies host data to GPU regardless of this
    //     setting (GPU cannot access host memory directly). data=0 and data=1 are equivalent.
    // Auto (-1): copies to GPU when index_storage_device=gpu.
    // Dataset location: -1=auto (follow index_storage_device), 0=keep on host, 1=copy to GPU.
    // Applies to both CAGRA and IVF indexes on ATS systems.
    //   CAGRA: controls cuVS make_strided_dataset() behavior via raft ATS flag.
    //   IVF: controls referenceFrom() vs copyInvertedListsFrom() in to_gpu().
    //        Requires use_cuvs=0 and APPLY_FAISS_IVF_PATCH.
    // On non-ATS (discrete GPU): 0 and 1 are equivalent (always copies).
    int index_data_on_gpu = -1;
    // Graph caching: 0=normal copyFrom_ex path, 1=cache graph as uint32 for fast to_gpu().
    // CAGRA-specific. Skips HNSW graph extraction + int64->uint32 conversion.
    // Requires APPLY_FAISS_CAGRA_PATCH=true in container build.
    int cagra_cache_graph = 0;

    // ========== Query Configuration ==========
    int k                   = 100;  // Top-K for queries
    int postfilter_ksearch  = 100;  // kSearch for conservative post-filtering
    float radius            = 0.5f; // Radius for range search (if used, overrides k)

    // ========== Distance Metric ==========
    VectorDistanceMetric metric = VectorDistanceMetric::L2;  // L2 or INNER_PRODUCT

    // ========== cuVS Toggle ==========
    bool use_cuvs = true;  // Whether to use cuVS for GPU indexes (default: true)

    // ========== Post Filter ==========
    bool use_post = false; // Whether to use postfilter instead of prefilter (default: false)

    // ========== Streaming LimitPerGroup ==========
    bool use_limit_per_group = false; // Use streaming LimitPerGroup instead of scatter-gather pattern

    // ========== Image Selectivity Control ==========
    int64_t filter_partkey = 0;  // When > 0, replaces i_variant=='MAIN' filter with i_partkey < threshold

    // ========== Query Range Control ==========
    // For batched/individual query execution (table slicing)
    int64_t query_start = 0;   // Starting row in query table (0-indexed)
    int64_t query_count = -1;  // Number of queries to run (-1 = all remaining)
    int incr_step = 0;         // Advance query_start by this amount each hot rep (0 = disabled)

    // Device for VS operators only; rel ops use outer `device`. Set by query_plan().
    DeviceType vs_device = DeviceType::CPU;
};

// Parse parameters from string (e.g., "hnsw_efsearch=100,ivf_nprobe=10")
QueryParameters parse_query_parameters(const std::string& params_str);

// Create index-specific search parameters based on faiss_index prefix
std::shared_ptr<IndexParameters> make_search_parameters(const std::string& faiss_index,
                                                         const QueryParameters& params,
                                                         DeviceType device = DeviceType::CPU);

// ============================================================================
// ENN Queries (Exhaustive Nearest Neighbor, no index)
// ============================================================================

std::shared_ptr<QueryPlan> enn_reviews(std::shared_ptr<Database>& db,
                                       DeviceType device            = DeviceType::CPU,
                                       const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> enn_reviews_project_distance(std::shared_ptr<Database>& db,
                                                        DeviceType device            = DeviceType::CPU,
                                                        const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> enn_images(std::shared_ptr<Database>& db,
                                      DeviceType device            = DeviceType::CPU,
                                      const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> enn_images_project_distance(std::shared_ptr<Database>& db,
                                                       DeviceType device            = DeviceType::CPU,
                                                       const QueryParameters& params = QueryParameters{});

// ============================================================================
// ANN Queries (Approximate Nearest Neighbor, requires pre-built index)
// ============================================================================

std::shared_ptr<QueryPlan> ann_reviews(std::shared_ptr<Database>& db,
                                       IndexPtr index,
                                       DeviceType device            = DeviceType::CPU,
                                       const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> ann_images(std::shared_ptr<Database>& db,
                                      IndexPtr index,
                                      DeviceType device            = DeviceType::CPU,
                                      const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> ann_reviews_full(std::shared_ptr<Database>& db,
                                            IndexPtr index,
                                            DeviceType device            = DeviceType::CPU,
                                            const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> ann_images_full(std::shared_ptr<Database>& db,
                                           IndexPtr index,
                                           DeviceType device            = DeviceType::CPU,
                                           const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> ann_ranged_reviews(std::shared_ptr<Database>& db,
                                       IndexPtr index,
                                       DeviceType device            = DeviceType::CPU,
                                       const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> ann_ranged_images(std::shared_ptr<Database>& db,
                                      IndexPtr index,
                                      DeviceType device            = DeviceType::CPU,
                                      const QueryParameters& params = QueryParameters{});

// ============================================================================
// Pre-filtering Queries (filter then vector search)
// ============================================================================

std::shared_ptr<QueryPlan> pre_reviews(std::shared_ptr<Database>& db,
                                       DeviceType device            = DeviceType::CPU,
                                       const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> pre_reviews_hybrid(std::shared_ptr<Database>& db,
                                       const QueryParameters& params = QueryParameters{});
 
std::shared_ptr<QueryPlan> pre_images(std::shared_ptr<Database>& db,
                                      DeviceType device            = DeviceType::CPU,
                                      const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> pre_images_hybrid(std::shared_ptr<Database>& db,
                                       const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> pre_reviews_partitioned(std::shared_ptr<Database>& db,
                                                    DeviceType device            = DeviceType::CPU,
                                                    const QueryParameters& params = QueryParameters{});

// ============================================================================
// Post-filtering Queries (vector search then filter)
// ============================================================================

std::shared_ptr<QueryPlan> post_reviews(std::shared_ptr<Database>& db,
                                        IndexPtr index,
                                        DeviceType device            = DeviceType::CPU,
                                        const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> post_reviews_hybrid(std::shared_ptr<Database>& db,
                                       IndexPtr index,
                                       const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> post_reviews_partitioned(std::shared_ptr<Database>& db,
                                        IndexPtr index,
                                        DeviceType device            = DeviceType::CPU,
                                        const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> post_reviews_filter_partitioned(std::shared_ptr<Database>& db,
                                        IndexPtr index,
                                        DeviceType device            = DeviceType::CPU,
                                        const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> post_images(std::shared_ptr<Database>& db,
                                       IndexPtr index,
                                       DeviceType device            = DeviceType::CPU,
                                       const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> post_images_hybrid(std::shared_ptr<Database>& db,
                                       IndexPtr index,
                                       const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> post_images_partitioned(std::shared_ptr<Database>& db,
                                        IndexPtr index,
                                        DeviceType device            = DeviceType::CPU,
                                        const QueryParameters& params = QueryParameters{});

// ============================================================================
// Bitmap-filtering Queries (Project 'filter' -> Indexed Vector Search)
// ============================================================================
 
std::shared_ptr<QueryPlan> pre_bitmap_ann_reviews(std::shared_ptr<Database>& db,
                                                IndexPtr index,
                                                DeviceType device            = DeviceType::CPU,
                                                const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> pre_bitmap_ann_images(std::shared_ptr<Database>& db,
                                                IndexPtr index,
                                                DeviceType device            = DeviceType::CPU,
                                                const QueryParameters& params = QueryParameters{});

// ============================================================================
// Expr-filtering Queries (Inline Arrow Expression -> Indexed Vector Search)
// ============================================================================

std::shared_ptr<QueryPlan> inline_expr_ann_reviews(std::shared_ptr<Database>& db,
                                                IndexPtr index,
                                                DeviceType device            = DeviceType::CPU,
                                                const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> inline_expr_ann_images(std::shared_ptr<Database>& db,
                                                IndexPtr index,
                                                DeviceType device            = DeviceType::CPU,
                                                const QueryParameters& params = QueryParameters{});

// ============================================================================
// Join + Filter Queries (combining relational joins with vector search)
// ============================================================================

// Pre-join: filter part → join reviews → exhaustive vector search
std::shared_ptr<QueryPlan> prejoin_reviews(std::shared_ptr<Database>& db,
                                            DeviceType device            = DeviceType::CPU,
                                            const QueryParameters& params = QueryParameters{});

// Post-join: vector search → join part → filter
std::shared_ptr<QueryPlan> postjoin_reviews(std::shared_ptr<Database>& db,
                                             IndexPtr index,
                                             DeviceType device            = DeviceType::CPU,
                                             const QueryParameters& params = QueryParameters{});


// ============================================================================
//                              VSDS Queries
// ============================================================================

std::shared_ptr<QueryPlan> q1_start(std::shared_ptr<Database>& db, IndexPtr index, DeviceType device, const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> q2_start(std::shared_ptr<Database>& db, IndexPtr index, DeviceType device, const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> q10_mid(std::shared_ptr<Database>& db, IndexPtr index, DeviceType device, const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> q11_end(std::shared_ptr<Database>& db, IndexPtr index, DeviceType device, const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> q13_mid(std::shared_ptr<Database>& db, IndexPtr index, DeviceType device, const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> q15_end(std::shared_ptr<Database>& db, IndexPtr index, DeviceType device, const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> q16_start(std::shared_ptr<Database>& db, IndexPtr index, DeviceType device, const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> q18_mid(std::shared_ptr<Database>& db, IndexPtr index, DeviceType device, const QueryParameters& params = QueryParameters{});

std::shared_ptr<QueryPlan> q19_start(std::shared_ptr<Database>& db, IndexPtr reviews_index, IndexPtr images_index, DeviceType device, const QueryParameters& params = QueryParameters{});


// ============================================================================
// Query Dispatcher (aligned with TPC-H/H2O pattern)
// ============================================================================

// Returns the index key(s) ("table.column") required by a query (empty if none needed)
std::vector<std::string> get_query_index_requirements(const std::string& query_name);

std::shared_ptr<QueryPlan> query_plan(const std::string& query,
                                      std::shared_ptr<Database>& db,
                                      DeviceType device         = DeviceType::CPU,
                                      const std::string& params = "",
                                      const IndexMap& indexes   = {},
                                      const std::string& index_desc = "",
                                      DeviceType vs_device      = DeviceType::CPU);

// ============================================================================
// Table metadata
// ============================================================================

std::shared_ptr<Schema> schema(const std::string& table_name);

std::vector<std::string> table_names();

std::vector<std::shared_ptr<Schema>> schemas();

std::vector<std::pair<std::string, std::string>> available_queries();

}  // namespace vsds
}  // namespace maximus
