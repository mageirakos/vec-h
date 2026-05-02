#pragma once

#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <arrow/api.h>
#include <arrow/compute/api.h>

#include <maximus/types/table.hpp>

namespace maximus::vsds {

// Result structure for a single query's k-NN results
struct QueryResult {
    int64_t data_id;
    float distance;
    
    bool operator==(const QueryResult& other) const {
        return data_id == other.data_id && 
               std::abs(distance - other.distance) < 1e-5f;
    }
};


inline std::map<int64_t, std::vector<QueryResult>> 
extract_results_by_query(TablePtr table, 
                         const std::string& query_id_col,
                         const std::string& data_id_col,
                         const std::string& distance_col = "vs_distance") {
    std::map<int64_t, std::vector<QueryResult>> results;
    
    if (!table || table->num_rows() == 0) return results;
    
    auto arrow_table = table->get_table();
    auto query_id_arr = arrow_table->GetColumnByName(query_id_col);
    auto data_id_arr = arrow_table->GetColumnByName(data_id_col);
    auto distance_arr = arrow_table->GetColumnByName(distance_col);
    
    if (!query_id_arr || !data_id_arr || !distance_arr) {
        throw std::runtime_error("extract_results_by_query: Missing required columns. "
                                 "Expected: " + query_id_col + ", " + data_id_col + ", " + distance_col);
    }
    
    // Concatenate chunks for uniform access
    auto qid_concat = arrow::Concatenate(query_id_arr->chunks(), arrow::default_memory_pool()).ValueOrDie();
    auto did_concat = arrow::Concatenate(data_id_arr->chunks(), arrow::default_memory_pool()).ValueOrDie();
    auto dist_concat = arrow::Concatenate(distance_arr->chunks(), arrow::default_memory_pool()).ValueOrDie();
    
    auto qid_typed = std::static_pointer_cast<arrow::Int64Array>(
        arrow::compute::Cast(qid_concat, arrow::int64()).ValueOrDie().make_array());
    auto did_typed = std::static_pointer_cast<arrow::Int64Array>(
        arrow::compute::Cast(did_concat, arrow::int64()).ValueOrDie().make_array());
    auto dist_typed = std::static_pointer_cast<arrow::FloatArray>(
        arrow::compute::Cast(dist_concat, arrow::float32()).ValueOrDie().make_array());
    
    for (int64_t i = 0; i < table->num_rows(); ++i) {
        int64_t qid = qid_typed->Value(i);
        int64_t did = did_typed->Value(i);
        float dist = dist_typed->Value(i);
        results[qid].push_back({did, dist});
    }
    
    return results;
}

// Compute recall: What fraction of ground truth neighbors were found in the results?
// 
// Parameters:
//   results: Results from ANN query (typically from extract_results_by_query)
//   ground_truth: Results from ENN query (ground truth)
//   k: Number of neighbors to consider per query (default: all)
//
// Returns: Recall value in [0.0, 1.0]
inline double compute_recall(
    const std::map<int64_t, std::vector<QueryResult>>& results,
    const std::map<int64_t, std::vector<QueryResult>>& ground_truth,
    int k = -1) {
    
    int64_t total_found = 0;
    int64_t total_expected = 0;
    
    for (const auto& [qid, gt_neighbors] : ground_truth) {
        auto it = results.find(qid);
        if (it == results.end()) continue;
        
        const auto& result_neighbors = it->second;
        
        // Determine how many ground truth neighbors to check
        size_t num_gt = (k > 0) ? std::min(static_cast<size_t>(k), gt_neighbors.size()) 
                                : gt_neighbors.size();
        
        // Count how many ground truth neighbors appear in results
        for (size_t i = 0; i < num_gt; ++i) {
            int64_t gt_id = gt_neighbors[i].data_id;
            for (const auto& r : result_neighbors) {
                if (r.data_id == gt_id) {
                    total_found++;
                    break;
                }
            }
        }
        total_expected += num_gt;
    }
    
    return (total_expected > 0) ? static_cast<double>(total_found) / total_expected : 0.0;
}

}  // namespace maximus::vsds
