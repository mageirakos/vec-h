#include <arrow/api.h>

#include <map>
#include <maximus/types/table.hpp>
#include <unordered_set>

namespace maximus {

template <typename ArrayType>
int compute_intersection_size(const std::shared_ptr<ArrayType>& array1,
                              const std::shared_ptr<ArrayType>& array2) {
    using T = typename ArrayType::value_type;

    std::unordered_set<T> id_set;
    for (int64_t i = 0; i < array1->length(); ++i) {
        if (!array1->IsNull(i)) {
            //std::cout << "Inserting value: " << i << ":  " << array1->Value(i) << std::endl;
            id_set.insert(array1->Value(i));
        }
    }

    // Query array 2
    int overlap_count = 0;
    for (int64_t i = 0; i < array2->length(); ++i) {
        //std::cout << "Searching value: " << array2->Value(i) << std::endl;
        if (!array2->IsNull(i) && id_set.find(array2->Value(i)) != id_set.end()) {
            overlap_count++;
        } else {
            //std::cout << "True value not found in retrieved: " << array2->Value(i) << std::endl;
        }
    }

    return overlap_count;
}

template <typename ArrayType>
int compute_num_tp(const std::shared_ptr<ArrayType>& query_ids,
                   const std::shared_ptr<ArrayType>& result_ids,
                   const std::shared_ptr<ArrayType>& gt_query_ids,
                   const std::shared_ptr<ArrayType>& gt_result_ids) {
    using T = typename ArrayType::value_type;

    assert(query_ids->length() == result_ids->length());
    assert(gt_query_ids->length() == gt_result_ids->length());

    std::map<T, std::unordered_set<T>> gt_sets;
    for (int64_t i = 0; i < gt_query_ids->length(); ++i) {
        gt_sets[gt_query_ids->Value(i)].insert(gt_result_ids->Value(i));
    }

    int32_t total_tp = 0;
    for (int64_t i = 0; i < query_ids->length(); ++i) {
        T qid = query_ids->Value(i);
        T rid = result_ids->Value(i);
        
        // Changed cast to size_t to support 64-bit safely
        if (qid < 0 || (size_t)qid >= gt_sets.size()) {
            throw std::runtime_error("Invalid query_id in result");
        }
        if (gt_sets[qid].count(rid)) {
            total_tp++;
        }
    }
    return total_tp;
}

template int compute_intersection_size<arrow::Int32Array>(
    const std::shared_ptr<arrow::Int32Array>&, const std::shared_ptr<arrow::Int32Array>&);

template int compute_num_tp<arrow::Int32Array>(
    const std::shared_ptr<arrow::Int32Array>&, const std::shared_ptr<arrow::Int32Array>&,
    const std::shared_ptr<arrow::Int32Array>&, const std::shared_ptr<arrow::Int32Array>&);

template int compute_intersection_size<arrow::Int64Array>(
    const std::shared_ptr<arrow::Int64Array>&, const std::shared_ptr<arrow::Int64Array>&);

template int compute_num_tp<arrow::Int64Array>(
    const std::shared_ptr<arrow::Int64Array>&, const std::shared_ptr<arrow::Int64Array>&,
    const std::shared_ptr<arrow::Int64Array>&, const std::shared_ptr<arrow::Int64Array>&);

}  // namespace maximus
