#include <maximus/gpu/cudf/table_reader.hpp>

namespace maximus {

namespace gpu {

std::string type_id_to_string(cudf::type_id t) {
    switch (t) {
        case cudf::type_id::EMPTY:
            return "EMPTY";
        case cudf::type_id::INT8:
            return "INT8";
        case cudf::type_id::INT16:
            return "INT16";
        case cudf::type_id::INT32:
            return "INT32";
        case cudf::type_id::INT64:
            return "INT64";
        case cudf::type_id::UINT8:
            return "UINT8";
        case cudf::type_id::UINT16:
            return "UINT16";
        case cudf::type_id::UINT32:
            return "UINT32";
        case cudf::type_id::UINT64:
            return "UINT64";
        case cudf::type_id::FLOAT32:
            return "FLOAT32";
        case cudf::type_id::FLOAT64:
            return "FLOAT64";
        case cudf::type_id::BOOL8:
            return "BOOL8";
        case cudf::type_id::TIMESTAMP_DAYS:
            return "TIMESTAMP_DAYS";
        case cudf::type_id::TIMESTAMP_SECONDS:
            return "TIMESTAMP_SECONDS";
        case cudf::type_id::TIMESTAMP_MILLISECONDS:
            return "TIMESTAMP_MILLISECONDS";
        case cudf::type_id::TIMESTAMP_MICROSECONDS:
            return "TIMESTAMP_MICROSECONDS";
        case cudf::type_id::TIMESTAMP_NANOSECONDS:
            return "TIMESTAMP_NANOSECONDS";
        case cudf::type_id::DURATION_DAYS:
            return "DURATION_DAYS";
        case cudf::type_id::DURATION_SECONDS:
            return "DURATION_SECONDS";
        case cudf::type_id::DURATION_MILLISECONDS:
            return "DURATION_MILLISECONDS";
        case cudf::type_id::DURATION_MICROSECONDS:
            return "DURATION_MICROSECONDS";
        case cudf::type_id::DURATION_NANOSECONDS:
            return "DURATION_NANOSECONDS";
        case cudf::type_id::DICTIONARY32:
            return "DICTIONARY32";
        case cudf::type_id::STRING:
            return "STRING";
        case cudf::type_id::LIST:
            return "LIST";
        case cudf::type_id::DECIMAL32:
            return "DECIMAL32";
        case cudf::type_id::DECIMAL64:
            return "DECIMAL64";
        case cudf::type_id::DECIMAL128:
            return "DECIMAL128";
        case cudf::type_id::STRUCT:
            return "STRUCT";
        case cudf::type_id::NUM_TYPE_IDS:
            return "NUM_TYPE_IDS";
        default:
            return "UNKNOWN_TYPE_ID";
    }
}

std::shared_ptr<cudf::table> gtable_to_cudf(std::shared_ptr<GTable> gtable) {
    return gtable->get_table();
}

std::shared_ptr<cudf::table_view> gtable_to_cudf_view(std::shared_ptr<GTable> gtable) {
    return std::make_shared<cudf::table_view>(gtable->get_table()->view());
}

}  // namespace gpu
}  // namespace maximus
