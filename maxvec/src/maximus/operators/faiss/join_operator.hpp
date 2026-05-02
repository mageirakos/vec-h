#pragma once

#include <maximus/indexes/faiss/faiss_index.hpp>
#include <maximus/operators/abstract_vector_join_operator.hpp>
#include <maximus/operators/faiss/id_selector.hpp>

namespace maximus::faiss {

using ChunkedArrayPtr      = std::shared_ptr<arrow::ChunkedArray>;
using RangeSearchResultPtr = std::unique_ptr<::faiss::RangeSearchResult>;

class JoinOperator
        : public maximus::AbstractVectorJoinOperator {
protected:
    // Subclass for range_search/knn_search functions' return type
    struct SearchResult {
        std::shared_ptr<arrow::ChunkedArray> left_indices;
        std::shared_ptr<arrow::ChunkedArray> right_indices;
        std::shared_ptr<arrow::ChunkedArray> distances;
    };

    JoinOperator(std::shared_ptr<MaximusContext> &ctx,
                 std::vector<std::shared_ptr<Schema>> input_schemas,
                 std::shared_ptr<VectorJoinProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    virtual DeviceTablePtr run_kernel(std::shared_ptr<MaximusContext> &ctx,
                                      const TablePtr &query_table,
                                      const TablePtr &data_table) = 0;

    static std::shared_ptr<arrow::Table> build_join_side(
        const std::shared_ptr<arrow::Table> &table,
        const std::shared_ptr<arrow::ChunkedArray> &indices,
        const std::shared_ptr<MaximusContext> &ctx,
        const std::vector<std::string> skip_columns);

    static SearchResult parse_range_search_results(
        const std::vector<std::vector<RangeSearchResultPtr>> &all_results, int64_t nq_results);

protected:
    TablePtr _data_table;
    TablePtr _query_table;
    std::unique_ptr<::faiss::IDSelector> _id_filter_selector;
};
}  // namespace maximus::faiss
