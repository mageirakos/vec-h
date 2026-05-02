#include <maximus/operators/abstract_take_operator.hpp>

namespace maximus {

AbstractTakeOperator::AbstractTakeOperator(
    std::shared_ptr<MaximusContext>& ctx,
    std::shared_ptr<Schema> data_schema,
    std::shared_ptr<Schema> index_schema,
    std::shared_ptr<TakeProperties> properties)
        : AbstractOperator(PhysicalOperatorType::TAKE, ctx)
        , properties(std::move(properties)) {
    std::vector<std::shared_ptr<Schema>> inputs;
    inputs.reserve(2);
    inputs.emplace_back(data_schema);
    inputs.emplace_back(index_schema);
    assign_input_schemas(inputs);

    // Compute output schema: all data columns + index non-key columns
    auto data_arrow = data_schema->get_schema();
    auto index_arrow = index_schema->get_schema();
    const auto& index_key_name = this->properties->index_key;

    arrow::FieldVector output_fields = data_arrow->fields();
    for (const auto& field : index_arrow->fields()) {
        if (field->name() != index_key_name) {
            output_fields.push_back(field);
        }
    }
    output_schema = std::make_shared<Schema>(arrow::schema(output_fields));

    // Port 0 (data) is blocking: we need all data to build the key→row map
    set_blocking_port(0);
}

}  // namespace maximus
