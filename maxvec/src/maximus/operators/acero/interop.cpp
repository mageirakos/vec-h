#include <maximus/operators/abstract_fused_operator.hpp>
#include <maximus/operators/acero/interop.hpp>

namespace maximus {
NodeType from_acero_node_name(const std::string& node_name) {
    // rewrite the code below using switch statement
    if (node_name == "table_source" || node_name == "record_batch_reader_source") {
        return NodeType::TABLE_SOURCE;
    } else if (node_name == "filter") {
        return NodeType::FILTER;
    } else if (node_name == "project") {
        return NodeType::PROJECT;
    } else if (node_name == "hashjoin") {
        return NodeType::HASH_JOIN;
    } else if (node_name == "aggregate") {
        return NodeType::GROUP_BY;
    } else if (node_name == "order_by") {
        return NodeType::ORDER_BY;
    } else if (node_name == "fetch") {
        return NodeType::LIMIT;
    } else {
        throw std::runtime_error("Unsupported Acero node type: " + node_name);
    }
}

std::string to_acero_node_name(const NodeType& node_type,
                               const std::shared_ptr<NodeProperties>& properties) {
    switch (node_type) {
        case NodeType::TABLE_SOURCE: {
            auto source_properties = std::dynamic_pointer_cast<TableSourceProperties>(properties);
            if (source_properties->table) {
                return "table_source";
            } else {
                return "record_batch_reader_source";
            }
        }
        case NodeType::FILTER:
            return "filter";
        case NodeType::DISTINCT:
            return "aggregate";  // distinct is a group-by with empty keys
        case NodeType::PROJECT:
            return "project";
        case NodeType::HASH_JOIN:
            return "hashjoin";
        case NodeType::GROUP_BY:
            return "aggregate";
        case NodeType::ORDER_BY:
            return "order_by";
        case NodeType::LIMIT:
            return "fetch";
        default:
            throw std::runtime_error("Unsupported Maximus node type");
    }
}

std::string to_string(arrow::acero::Declaration decl) {
    auto maybe_str = arrow::acero::DeclarationToString(decl);
    if (!maybe_str.ok()) {
        CHECK_STATUS(maybe_str.status());
    }
    return maybe_str.ValueOrDie();
}

arrow::acero::Declaration to_acero_declaration(const std::shared_ptr<QueryNode>& node) {
    auto& inputs = node->get_inputs();
    if (inputs.empty()) {
        assert(node->logical_type == NodeType::TABLE_SOURCE);
    }

    // the plan must be unfused before converting to Acero
    assert(node->logical_type != NodeType::FUSED);

    if (node->logical_type == NodeType::LOCAL_BROADCAST) {
        throw std::runtime_error(
            "LocalBroadcastOperator is not supported in Acero. There is a plan "
            "to do it, but it's not there yet.");
    }

    auto node_type = to_acero_node_name(node->logical_type, node->properties);

    // std::cout << "Converting Maximus node: " << node_type_to_string(node->logical_type) << " ---> Acero node: " << node_type << std::endl;

    auto options = to_acero_options(node->get_context(), node->properties);

    // auto temp_maximus_properties = from_acero_options(options, ctx);
    // std::cout << "Maximus properties: " << temp_maximus_properties->to_string() << std::endl;

    std::vector<arrow::acero::Declaration::Input> input_decl;
    input_decl.reserve(inputs.size());
    for (auto& input : inputs) {
        input_decl.emplace_back(to_acero_declaration(input));
    }
    auto decl = arrow::acero::Declaration(node_type, std::move(input_decl), std::move(options));
    // std::cout << "Converted " << node_type << " as Acero Declaration = " << to_string(decl) << std::endl;
    return decl;
}

arrow::acero::Declaration to_acero_declaration(const std::shared_ptr<QueryPlan>& plan) {
    assert(plan);
    assert(plan->is_query_plan_root());
    assert(plan->get_inputs().size() == 1);
    auto table_sink = plan->get_inputs()[0];
    assert(table_sink);
    assert(table_sink->logical_type == NodeType::TABLE_SINK);

    assert(table_sink->get_inputs().size() == 1);

    table_sink->unfuse_deep();

    /*
    std::cout << "=================================" << std::endl;
    std::cout << "Unfused table sink:\n " << table_sink->to_string() << std::endl;
    std::cout << "=================================" << std::endl;
    */

    auto inner_qp = table_sink->get_inputs()[0];

    assert(inner_qp);

    return to_acero_declaration(inner_qp);
}
}  // namespace maximus
