#include <maximus/operators/abstract_fused_operator.hpp>

namespace maximus {

AbstractFusedOperator::AbstractFusedOperator(std::shared_ptr<MaximusContext> &ctx,
                                             std::shared_ptr<Schema> input_schema,
                                             std::shared_ptr<FusedProperties> properties)
        : AbstractOperator(PhysicalOperatorType::FUSED, ctx, std::move(input_schema))
        , properties(std::move(properties)) {
    for (const auto &node : this->properties->node_types) {
        physical_types.push_back(node_type_to_operator_type(node));
    }
}

[[nodiscard]] std::string AbstractFusedOperator::to_string_extra() {
    std::stringstream ss;
    for (std::size_t i = 0; i < physical_types.size(); ++i) {
        if (i < physical_types.size() - 1) {
            ss << physical_operator_to_string(physical_types[i]) << " + ";
        } else {
            ss << physical_operator_to_string(physical_types[i]);
        }
    }
    return ss.str();
}

}  // namespace maximus