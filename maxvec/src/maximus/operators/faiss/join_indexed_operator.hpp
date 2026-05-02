#pragma once

#include <maximus/operators/faiss/join_operator.hpp>

namespace maximus::faiss {

class JoinIndexedOperator : public JoinOperator {
public:
    JoinIndexedOperator(std::shared_ptr<MaximusContext> &ctx,
                        std::vector<std::shared_ptr<Schema>> input_schemas,
                        std::shared_ptr<VectorJoinIndexedProperties> properties);

    void on_no_more_input(int port) override;

    DeviceTablePtr run_kernel(std::shared_ptr<MaximusContext> &ctx,
                              const TablePtr &query_table,
                              const TablePtr &data_table) override;

    SearchResult knn_search(const ChunkedArrayPtr &query_vectors);

    SearchResult range_search(const ChunkedArrayPtr &query_vectors);

protected:
    std::shared_ptr<VectorJoinIndexedProperties> properties;
    std::shared_ptr<::faiss::SearchParameters> _search_parameters;
    FaissIndexPtr _index;
    arrow::compute::Expression bound_filter_expression;
};
}  // namespace maximus::faiss
