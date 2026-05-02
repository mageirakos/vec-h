#pragma once

#include <maximus/operators/abstract_vector_project_distance_operator.hpp>

namespace maximus::faiss {

using ChunkedArrayPtr = std::shared_ptr<arrow::ChunkedArray>;
using FloatArrayPtr   = std::shared_ptr<arrow::FloatArray>;

class ProjectDistanceOperator
        : public maximus::AbstractVectorProjectDistanceOperator {
public:
    ProjectDistanceOperator(std::shared_ptr<MaximusContext> &ctx,
                            std::vector<std::shared_ptr<Schema>> input_schemas,
                            std::shared_ptr<VectorProjectDistanceProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;
    void on_no_more_input(int port) override {};

    static TablePtr run_kernel(std::shared_ptr<MaximusContext> &ctx,
                               TablePtr &left_table,
                               TablePtr &right_table,
                               VectorProjectDistanceProperties &properties);

    static FloatArrayPtr compute_pairwise_distances(const ChunkedArrayPtr &query_vectors,
                                                    const ChunkedArrayPtr &data_vectors,
                                                    int64_t D,
                                                    arrow::MemoryPool *pool);

protected:
    TablePtr _buffer_table[2];  // Buffer tables for left and right inputs
};
}  // namespace maximus::faiss
