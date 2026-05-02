#pragma once

#include <maximus/types/device_table_ptr.hpp>
#include <arrow/api.h>
#include <memory>

namespace maximus {

struct QualityMetrics {
    double recall     = NAN;
    double precision  = NAN;
    int num_retrieved = -1;

    void print() {
        if (recall != NAN) std::cout << "Recall: " << recall * 100 << "%" << std::endl;
        if (precision != NAN) std::cout << "Precision: " << precision * 100 << "%" << std::endl;
        if (num_retrieved != -1) std::cout << "#Retrieved: " << num_retrieved << std::endl;
    }
};

// Declare as templates
template <typename ArrayType>
int compute_intersection_size(const std::shared_ptr<ArrayType>& array1,
                              const std::shared_ptr<ArrayType>& array2);

template <typename ArrayType>
int compute_num_tp(const std::shared_ptr<ArrayType>& query_id,
                   const std::shared_ptr<ArrayType>& result_id,
                   const std::shared_ptr<ArrayType>& gt_query_id,
                   const std::shared_ptr<ArrayType>& gt_result_id);

}  // namespace maximus