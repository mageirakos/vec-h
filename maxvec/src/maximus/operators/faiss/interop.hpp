#pragma once

#include <faiss/MetricType.h>

#include <maximus/operators/properties.hpp>

namespace maximus {

::faiss::MetricType to_faiss_metric(const VectorDistanceMetric maximus_metric);

VectorDistanceMetric from_faiss_metric(const ::faiss::MetricType metric_type);


/************************************************************************************/
// Arrow Helper functions.                                                           /
/************************************************************************************/

float *raw_ptr_from_array(const arrow::FixedSizeListArray &array);
float *raw_ptr_from_array(const arrow::ListArray &array);
float *raw_ptr_from_array(const arrow::LargeListArray &array);
const float *get_embedding_raw_ptr(const std::shared_ptr<arrow::Array> &array);
// Short human readable metric name (e.g. "IP", "L2")
std::string metric_short_name(const ::faiss::MetricType metric);
std::string compute_arrow_data_hash(const std::shared_ptr<arrow::Array> &vector_array,
                         int dimension);

}  // namespace maximus
