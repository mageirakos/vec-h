#pragma once

#include <maximus/operators/faiss/join_operator.hpp>

namespace maximus::faiss {

class JoinExhaustiveOperator : public JoinOperator {
public:
    JoinExhaustiveOperator(std::shared_ptr<MaximusContext> &ctx,
                           std::vector<std::shared_ptr<Schema>> input_schemas,
                           std::shared_ptr<VectorJoinExhaustiveProperties> properties);

    DeviceTablePtr run_kernel(std::shared_ptr<MaximusContext> &ctx,
                              const TablePtr &query_table,
                              const TablePtr &data_table) override;

    SearchResult knn_search(const ChunkedArrayPtr &query_vectors,
                            const ChunkedArrayPtr &data_vectors);

    SearchResult range_search(const ChunkedArrayPtr &query_vectors,
                              const ChunkedArrayPtr &data_vectors);

    static void knn_exhaustive_search(const int64_t D,
                                      arrow::ArrayVector &data_vectors,
                                      const int64_t nq,
                                      const float *query_vectors_ptr,
                                      const int64_t K,
                                      const VectorDistanceMetric metric,
                                      float *distances,
                                      ::faiss::idx_t *labels,
                                      const ::faiss::IDSelector *sel);


public:
    std::shared_ptr<VectorJoinExhaustiveProperties> properties;
};
}  // namespace maximus::faiss
