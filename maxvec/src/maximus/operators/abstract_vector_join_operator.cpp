#include <maximus/operators/abstract_vector_join_operator.hpp>

namespace maximus {

maximus::AbstractVectorJoinOperator::AbstractVectorJoinOperator(
    std::shared_ptr<MaximusContext>& ctx,
    std::vector<std::shared_ptr<Schema>> input_schemas,
    std::shared_ptr<VectorJoinProperties> properties)
        : AbstractOperator(PhysicalOperatorType::VECTOR_JOIN, ctx, input_schemas)
        , abstract_properties(properties) {
    auto data_port = get_data_port();
    auto query_port = get_query_port();

    set_blocking_port(data_port);
    set_streaming_port(query_port);  // Query vectors can be streamed (on the CPU)

    // Two ports expected
    assert(input_schemas.size() == 2);

    // Output schema
    auto data_schema        = input_schemas[data_port]->get_schema();
    auto query_schema       = input_schemas[query_port]->get_schema();

    auto maybe_query_field = properties->query_vector_column.GetOne(*query_schema);
    CHECK_STATUS(maybe_query_field.status());
    auto query_field = maybe_query_field.ValueOrDie();
    // assert(query_field->type()->id() == EmbeddingsListTypeId);  // must be an embedding type

    // Check if we're using an indexed join with a trained index
    // If so, the data vector column is not required in the input schema (the index owns the data)
    std::shared_ptr<arrow::Field> data_field = nullptr;
    auto indexed_props = std::dynamic_pointer_cast<VectorJoinIndexedProperties>(properties);
    bool has_trained_index = indexed_props && indexed_props->index && indexed_props->index->is_trained();
    
    if (has_trained_index) {
        // With a trained index, the data vector column is optional in input schema
        auto maybe_data_field = properties->data_vector_column.GetOne(*data_schema);
        if (maybe_data_field.ok()) {
            data_field = maybe_data_field.ValueOrDie();
        }
        // If not found, data_field remains nullptr - this is fine for trained indexes
    } else {
        // Without a trained index, the data vector column must exist in the input schema
        auto maybe_data_field = properties->data_vector_column.GetOne(*data_schema);
        CHECK_STATUS(maybe_data_field.status());
        data_field = maybe_data_field.ValueOrDie();
    }

    std::vector<std::shared_ptr<arrow::Field>> output_fields;
    for (const auto &field : query_schema->fields()) {
        // if the properties specify that we should not keep the embedding column, skip it
        if (!properties->keep_query_vector_column && field->name() == query_field->name()) {
            continue;
        }
        output_fields.push_back(field);
    }
    for (const auto &field : data_schema->fields()) {
        // if the properties specify that we should not keep the embedding column, skip it
        // (only if data_field exists - it won't exist for trained index without data column in schema)
        if (data_field && !properties->keep_data_vector_column && field->name() == data_field->name()) {
            continue;
        }
        // if a filter_bitmap is provided, skip that column as it's temporarily used by the vectorjoin
        if (properties->filter_bitmap.has_value() && 
            field->name() == *properties->filter_bitmap.value().name()) {
            continue;
        }
        output_fields.push_back(field);
    }
    if (properties->distance_column) {
        auto distance_field = arrow::field(properties->distance_column.value(), arrow::float32());
        output_fields.push_back(distance_field);
    }
    auto joined_schema = arrow::schema(output_fields);
    assign_output_schema(std::make_shared<Schema>(joined_schema));

    // dimensionality can only be set once we receive the data
}
}  // namespace maximus