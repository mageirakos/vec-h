#pragma once
#include <arrow/acero/api.h>

#include <maximus/dag/query_plan.hpp>
#include <maximus/operators/acero/properties.hpp>

namespace maximus {

NodeType from_acero_node_name(const std::string& node_name);

std::string to_acero_node_name(const NodeType& node_type,
                               const std::shared_ptr<NodeProperties>& properties);

std::string to_string(arrow::acero::Declaration decl);

arrow::acero::Declaration to_acero_declaration(const std::shared_ptr<QueryNode>& node);

arrow::acero::Declaration to_acero_declaration(const std::shared_ptr<QueryPlan>& plan);

}  // namespace maximus