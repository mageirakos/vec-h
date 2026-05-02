#pragma once
#include <arrow/type.h>

#include <maximus/error_handling.hpp>

#if defined(MAXIMUS_WITH_CUDA) && defined(MAXIMUS_WITH_VS)
#include <cudf/column/column_view.hpp>
#endif

namespace maximus {
// we want to define maximus::DataType to be the same (alias) as the arrow:DataType
using DataType              = arrow::DataType;
using Type                  = arrow::Type;
using DataTypeLayout        = arrow::DataTypeLayout;
using TimeUnit              = arrow::TimeUnit;
using TimestampType         = arrow::TimestampType;
using DurationType          = arrow::DurationType;
using DecimalType           = arrow::DecimalType;
using Decimal128Type        = arrow::Decimal128Type;
using FixedSizeBinaryType   = arrow::FixedSizeBinaryType;
using FixedSizeListType     = arrow::FixedSizeListType;
using ListType              = arrow::ListType;

// ********************************************************
// CHANGING THIS CHANGES THE WAY HOW EMBEDDINGS ARE STORED
// ********************************************************
using EmbeddingsListType    = arrow::ListType;
using EmbeddingsArray       = arrow::ListArray;
enum VectorDistanceMetric { INNER_PRODUCT, L2 };

constexpr arrow::Type::type EmbeddingsListTypeId  = arrow::Type::LIST;

std::shared_ptr<arrow::DataType> embeddings_list(std::shared_ptr<arrow::DataType> precision, int dimension);

int embedding_dimension(const std::shared_ptr<arrow::Array>& vector_array);

#if defined(MAXIMUS_WITH_CUDA) && defined(MAXIMUS_WITH_VS)
int embedding_dimension(cudf::column_view const& vector_col);
#endif

std::shared_ptr<arrow::FloatArray> embeddings_values(const std::shared_ptr<arrow::Array>& column);

Status are_types_supported(const std::shared_ptr<arrow::Schema> &schema);

std::shared_ptr<DataType> to_maximus_type(std::shared_ptr<arrow::DataType> type);
std::shared_ptr<arrow::DataType> to_arrow_type(std::shared_ptr<DataType> type);

}  // namespace maximus
