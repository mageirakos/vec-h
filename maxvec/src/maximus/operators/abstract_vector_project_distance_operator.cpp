#include <maximus/operators/abstract_vector_project_distance_operator.hpp>

namespace maximus {

maximus::AbstractVectorProjectDistanceOperator::AbstractVectorProjectDistanceOperator(
    std::shared_ptr<MaximusContext>& ctx,
    std::vector<std::shared_ptr<Schema>> input_schemas,
    std::shared_ptr<VectorProjectDistanceProperties> properties)
        : AbstractOperator(PhysicalOperatorType::VECTOR_PROJECT_DISTANCE, ctx, input_schemas)
        , properties(properties) {

    auto left_data_port = 0;
    auto right_query_port = 1;

    set_streaming_port(left_data_port);
    set_streaming_port(right_query_port);

    assert(input_schemas.size() == 2);  // Two ports

    auto right_schema       = input_schemas[1]->get_schema();
    auto right_field_result = properties->right_vector_column.GetOne(*right_schema);
    CHECK_STATUS(right_field_result.status());  // right schema has the column
    auto right_vector_type = right_field_result.ValueOrDie()->type();

    auto left_schema       = input_schemas[0]->get_schema();
    auto left_field_result = properties->left_vector_column.GetOne(*left_schema);
    CHECK_STATUS(left_field_result.status());  // right schema has the column
    auto left_vector_type = left_field_result.ValueOrDie()->type();

    // Output Schema
    std::vector<std::shared_ptr<arrow::Field>> output_fields;
    // 1. Query Columns (Right Table - Queries) first
    for (const auto& field : right_schema->fields()) {
        if (!properties->keep_right_vector_column &&
            field->name() == right_field_result.ValueOrDie()->name()) {
            continue;
        }
        output_fields.push_back(field);
    }
    // 2. Data Columns (Left Table) second
    for (const auto& field : left_schema->fields()) {
        if (!properties->keep_left_vector_column &&
            field->name() == left_field_result.ValueOrDie()->name()) {
            continue;
        }
        output_fields.push_back(field);
    }
    auto distance_field = arrow::field(properties->distance_column_name, arrow::float32());
    output_fields.push_back(distance_field);
    auto joined_schema = arrow::schema(output_fields);
    assign_output_schema(std::make_shared<Schema>(joined_schema));

    // D can only be known once we receive the first batch of data
    // D = std::static_pointer_cast<arrow::FixedSizeListType>(right_vector_type)->list_size();
}

}  // namespace maximus