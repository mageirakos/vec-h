#include <cassert>
#include <unordered_set>

#if defined(MAXIMUS_WITH_CUDA) && defined(MAXIMUS_WITH_VS)
#include <cuda_runtime_api.h>

#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>
#endif

#include <maximus/types/types.hpp>

namespace maximus {
// Define a set of supported types for quick lookup
const std::unordered_set<arrow::Type::type> supported_types = {
    arrow::Type::BOOL,       arrow::Type::UINT8,
    arrow::Type::INT8,       arrow::Type::UINT16,
    arrow::Type::INT16,      arrow::Type::UINT32,
    arrow::Type::INT32,      arrow::Type::UINT64,
    arrow::Type::INT64,      arrow::Type::HALF_FLOAT,
    arrow::Type::FLOAT,      arrow::Type::DOUBLE,
    arrow::Type::DECIMAL128, arrow::Type::FIXED_SIZE_BINARY,
    arrow::Type::BINARY,     arrow::Type::STRING,
    arrow::Type::DATE32,     arrow::Type::DATE64,
    arrow::Type::TIMESTAMP,  arrow::Type::TIME32,
    arrow::Type::TIME64,     arrow::Type::FIXED_SIZE_LIST,
    arrow::Type::LIST,       arrow::Type::LARGE_LIST};

Status are_types_supported(const std::shared_ptr<arrow::Schema>& schema) {
    for (const auto& field : schema->fields()) {
        auto type_id = field->type()->id();

        if (supported_types.count(type_id)) {
            continue;
        }

        return {ErrorCode::MaximusError,
                "Unsupported type found in the schema: " + field->type()->ToString()};
    }
    return Status::OK();
}

std::shared_ptr<DataType> to_maximus_type(std::shared_ptr<arrow::DataType> type) {
    return type;
}

std::shared_ptr<arrow::DataType> to_arrow_type(std::shared_ptr<DataType> type) {
    return type;
}

std::shared_ptr<arrow::DataType> embeddings_list(std::shared_ptr<arrow::DataType> precision,
                                                 int dimension) {
    return arrow::list(precision);
}

int embedding_dimension(const std::shared_ptr<arrow::Array>& vector_array) {
    if (!vector_array) throw std::invalid_argument("Null vector_array");

    auto type = vector_array->type();

    switch (type->id()) {
        case arrow::Type::FIXED_SIZE_LIST: {
            auto fixed_type = std::static_pointer_cast<arrow::FixedSizeListType>(type);
            return fixed_type->list_size();
        }

        case arrow::Type::LIST: {
            auto list_array = std::static_pointer_cast<arrow::ListArray>(vector_array);

            if (list_array->length() == 0)
                throw std::runtime_error("Empty ListArray: cannot determine embedding dimension");
            if (list_array->IsNull(0))
                throw std::runtime_error(
                    "First element is null: cannot determine embedding dimension");

            return list_array->value_length(0);
        }

        case arrow::Type::LARGE_LIST: {
            auto large_list_array = std::static_pointer_cast<arrow::LargeListArray>(vector_array);

            if (large_list_array->length() == 0)
                throw std::runtime_error("Empty LargeListArray: cannot determine embedding dimension");
            if (large_list_array->IsNull(0))
                throw std::runtime_error(
                    "First element is null: cannot determine embedding dimension");

            return large_list_array->value_length(0);
        }

        default:
            throw std::runtime_error("Expected a ListType, LargeListType or FixedSizeListType for embeddings");
    }
}

#if defined(MAXIMUS_WITH_CUDA) && defined(MAXIMUS_WITH_VS)

int embedding_dimension(cudf::column_view const& vector_col) {
    assert(!vector_col.is_empty());

    auto type_id = vector_col.type().id();
    assert(type_id == cudf::type_id::LIST);

    cudf::lists_column_view lists_col(vector_col);

    // offsets_begin() returns a device pointer - must copy to host to read
    auto offsets_device_ptr = lists_col.offsets_begin();
    cudf::size_type host_offsets[2];
    cudaMemcpy(
        host_offsets, offsets_device_ptr, 2 * sizeof(cudf::size_type), cudaMemcpyDeviceToHost);

    return static_cast<int>(host_offsets[1] - host_offsets[0]);
}

#endif

std::shared_ptr<arrow::FloatArray> embeddings_values(const std::shared_ptr<arrow::Array>& column) {
    if (!column) throw std::invalid_argument("column is null");

    auto type = column->type();

    switch (type->id()) {
        case arrow::Type::FIXED_SIZE_LIST: {
            auto fixed = std::static_pointer_cast<arrow::FixedSizeListArray>(column);
            return std::static_pointer_cast<arrow::FloatArray>(fixed->values());
        }
        case arrow::Type::LIST: {
            auto list = std::static_pointer_cast<arrow::ListArray>(column);
            return std::static_pointer_cast<arrow::FloatArray>(list->values());
        }
        case arrow::Type::LARGE_LIST: {
            auto large_list = std::static_pointer_cast<arrow::LargeListArray>(column);
            return std::static_pointer_cast<arrow::FloatArray>(large_list->values());
        }
        default:
            throw std::runtime_error("Column is neither ListArray, LargeListArray nor FixedSizeListArray");
    }
}

}  // namespace maximus
