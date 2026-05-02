#pragma once
#include <maximus/dag/query_node.hpp>

namespace maximus {

class QueryPlan : public QueryNode {
public:
    QueryPlan(std::shared_ptr<MaximusContext>& ctx);

    void convert_to_physical() override;

    void infer_types() override;

    void fuse() override;

    void unfuse() override;

    std::vector<std::shared_ptr<Pipeline>> to_pipelines();

    void add_pipeline(std::shared_ptr<Pipeline> pipeline);

    std::vector<std::shared_ptr<Pipeline>> pipelines;

    TablePtr result();

    std::string physical_plan() const;
};
}  // namespace maximus
