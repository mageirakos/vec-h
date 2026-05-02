#include <maximus/gpu/cudf/cudf_types.hpp>

#include <cudf/column/column_factories.hpp>
#include <maximus/types/schema.hpp>

namespace maximus {

namespace gpu {

std::shared_ptr<DataType> to_maximus_type(const cudf::data_type &type) {
    switch (type.id()) {
        case cudf::type_id::BOOL8:
            return arrow::boolean();
        case cudf::type_id::INT8:
            return arrow::int8();
        case cudf::type_id::INT16:
            return arrow::int16();
        case cudf::type_id::INT32:
            return arrow::int32();
        case cudf::type_id::INT64:
            return arrow::int64();
        case cudf::type_id::UINT8:
            return arrow::uint8();
        case cudf::type_id::UINT16:
            return arrow::uint16();
        case cudf::type_id::UINT32:
            return arrow::uint32();
        case cudf::type_id::UINT64:
            return arrow::uint64();
        case cudf::type_id::FLOAT32:
            return arrow::float32();
        case cudf::type_id::FLOAT64:
            return arrow::float64();
        case cudf::type_id::TIMESTAMP_DAYS:
            return arrow::date32();
        case cudf::type_id::TIMESTAMP_SECONDS:
            return arrow::timestamp(arrow::TimeUnit::SECOND);
        case cudf::type_id::TIMESTAMP_MILLISECONDS:
            return arrow::timestamp(arrow::TimeUnit::MILLI);
        case cudf::type_id::TIMESTAMP_MICROSECONDS:
            return arrow::timestamp(arrow::TimeUnit::MICRO);
        case cudf::type_id::TIMESTAMP_NANOSECONDS:
            return arrow::timestamp(arrow::TimeUnit::NANO);
        case cudf::type_id::DURATION_DAYS:  // Maximus has no DURATION_DAYS type
            break;
        case cudf::type_id::DURATION_SECONDS:  // Maximus has no DURATION_SECONDS
                                               // type
            break;
        case cudf::type_id::DURATION_MILLISECONDS:  // Maximus has no
            // DURATION_MILLISECONDS type
            break;
        case cudf::type_id::DURATION_MICROSECONDS:  // Maximus has no
            // DURATION_MICROSECONDS type
            break;
        case cudf::type_id::DURATION_NANOSECONDS:  // not supported currently
            break;
        case cudf::type_id::STRING:
            return arrow::utf8();
        case cudf::type_id::LIST:  // not supported currently
            break;
        case cudf::type_id::DECIMAL32:  // Maximus has no DECIMAL32 type
            break;
        case cudf::type_id::DECIMAL64:  // Maximus has no DECIMAL64 type
            break;
        case cudf::type_id::DECIMAL128:  // Maximus has no DECIMAL128 type
            break;
        case cudf::type_id::STRUCT:  // not supported currently
            break;
        case cudf::type_id::DICTIONARY32:  // not supported currently
            break;
    }
    return nullptr;
}

cudf::data_type to_cudf_type(const std::shared_ptr<DataType> &type) {
    switch (type->id()) {
        case arrow::Type::BOOL:
            return cudf::data_type(cudf::type_id::BOOL8);
        case arrow::Type::INT8:
            return cudf::data_type(cudf::type_id::INT8);
        case arrow::Type::INT16:
            return cudf::data_type(cudf::type_id::INT16);
        case arrow::Type::INT32:
            return cudf::data_type(cudf::type_id::INT32);
        case arrow::Type::INT64:
            return cudf::data_type(cudf::type_id::INT64);
        case arrow::Type::UINT8:
            return cudf::data_type(cudf::type_id::UINT8);
        case arrow::Type::UINT16:
            return cudf::data_type(cudf::type_id::UINT16);
        case arrow::Type::UINT32:
            return cudf::data_type(cudf::type_id::UINT32);
        case arrow::Type::UINT64:
            return cudf::data_type(cudf::type_id::UINT64);
        case arrow::Type::FLOAT:
            return cudf::data_type(cudf::type_id::FLOAT32);
        case arrow::Type::DOUBLE:
            return cudf::data_type(cudf::type_id::FLOAT64);
        case arrow::Type::DATE32:
            return cudf::data_type(cudf::type_id::TIMESTAMP_DAYS);
        case arrow::Type::TIMESTAMP:
            switch (std::static_pointer_cast<arrow::TimestampType>(type)->unit()) {
                case arrow::TimeUnit::SECOND:
                    return cudf::data_type(cudf::type_id::TIMESTAMP_SECONDS);
                case arrow::TimeUnit::MILLI:
                    return cudf::data_type(cudf::type_id::TIMESTAMP_MILLISECONDS);
                case arrow::TimeUnit::MICRO:
                    return cudf::data_type(cudf::type_id::TIMESTAMP_MICROSECONDS);
                case arrow::TimeUnit::NANO:
                    return cudf::data_type(cudf::type_id::TIMESTAMP_NANOSECONDS);
            }
        case arrow::Type::STRING:
            return cudf::data_type(cudf::type_id::STRING);
        // WIP 3 NOTE @Vasilis: unsure if this is related to debugging the GPU operator, spend some time on it yourself
        case arrow::Type::LIST:
        case arrow::Type::LARGE_LIST:
        // case arrow::Type::FIXED_SIZE_LIST:
            // cuDF uses LIST type for all Arrow list variants
            return cudf::data_type(cudf::type_id::LIST);
        default:
            throw std::runtime_error("cuDF does not support the data type: " + type->ToString());
            break;
    }
    return cudf::data_type(cudf::type_id::EMPTY);
}

/**
 * To convert an arrow data type to a cudf data type
 */
std::shared_ptr<arrow::DataType> to_arrow_type(const cudf::data_type &type) {
    switch (type.id()) {
        case cudf::type_id::BOOL8:
            return arrow::boolean();
        case cudf::type_id::INT8:
            return arrow::int8();
        case cudf::type_id::INT16:
            return arrow::int16();
        case cudf::type_id::INT32:
            return arrow::int32();
        case cudf::type_id::INT64:
            return arrow::int64();
        case cudf::type_id::UINT8:
            return arrow::uint8();
        case cudf::type_id::UINT16:
            return arrow::uint16();
        case cudf::type_id::UINT32:
            return arrow::uint32();
        case cudf::type_id::UINT64:
            return arrow::uint64();
        case cudf::type_id::FLOAT32:
            return arrow::float32();
        case cudf::type_id::FLOAT64:
            return arrow::float64();
        case cudf::type_id::TIMESTAMP_DAYS:
            return arrow::date32();
        case cudf::type_id::TIMESTAMP_SECONDS:
            return arrow::timestamp(arrow::TimeUnit::SECOND);
        case cudf::type_id::TIMESTAMP_MILLISECONDS:
            return arrow::timestamp(arrow::TimeUnit::MILLI);
        case cudf::type_id::TIMESTAMP_MICROSECONDS:
            return arrow::timestamp(arrow::TimeUnit::MICRO);
        case cudf::type_id::TIMESTAMP_NANOSECONDS:
            return arrow::timestamp(arrow::TimeUnit::NANO);
        case cudf::type_id::STRING:
            return arrow::utf8();
        case cudf::type_id::LIST:
            break;
        case cudf::type_id::DECIMAL32:
            break;
        case cudf::type_id::DECIMAL64:
            break;
        case cudf::type_id::DECIMAL128:
            break;
        case cudf::type_id::STRUCT:
            break;
        case cudf::type_id::DICTIONARY32:
            break;
    }
    return nullptr;
}

std::vector<cudf::column_metadata> to_cudf_column_metadata(
    const std::shared_ptr<arrow::Schema> &schema) {
    std::vector<cudf::column_metadata> metadata;
    metadata.reserve(schema->num_fields());
    for (auto &field : schema->fields()) {
        metadata.emplace_back(field->name());
    }
    return metadata;
}

std::shared_ptr<cudf::table> make_empty_cudf_table(const std::shared_ptr<Schema> &schema) {
    auto fields = schema->get_schema()->fields();
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(fields.size());
    for (const auto &field : fields) {
        columns.push_back(cudf::make_empty_column(to_cudf_type(field->type())));
    }
    return std::make_shared<cudf::table>(std::move(columns));
}

}  // namespace gpu

}  // namespace maximus
