#include <maximus/operators/abstract_gather_operator.hpp>

namespace maximus {

AbstractGatherOperator::AbstractGatherOperator(
    std::shared_ptr<MaximusContext>& ctx,
    std::shared_ptr<Schema> input_schema,
    std::shared_ptr<GatherProperties> properties)
        : AbstractOperator(PhysicalOperatorType::GATHER, ctx,
            // Create input schemas for all ports (all same schema)
            std::vector<std::shared_ptr<Schema>>(properties->num_inputs, input_schema),
            input_schema)  // Output schema is the same as input
        , properties(std::move(properties)) {
    input_tables_.resize(this->properties->num_inputs);
}

}  // namespace maximus
