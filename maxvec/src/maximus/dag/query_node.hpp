#pragma once

#include <maximus/context.hpp>
#include <maximus/dag/edge.hpp>
#include <maximus/exec/pipeline.hpp>
#include <maximus/operators/abstract_operator.hpp>
#include <maximus/operators/engine.hpp>
#include <maximus/operators/properties.hpp>
#include <maximus/types/node_type.hpp>
#include <maximus/types/schema.hpp>
#include <vector>
#include <unordered_set>

namespace maximus {

// forward-declaration
class QueryPlan;

class QueryNode : public std::enable_shared_from_this<QueryNode> {
public:
    // with a vector of input and output schemas
    QueryNode();

    QueryNode(EngineType engine,
              DeviceType device,
              NodeType logical_type,
              std::shared_ptr<NodeProperties> properties,
              std::shared_ptr<MaximusContext>& ctx);

    ~QueryNode();

    virtual void convert_to_physical();

    void convert_to_physical_deep();

    // Get a reference to the internal operator
    std::shared_ptr<AbstractOperator>& get_operator();

    std::size_t in_degree() const;
    std::size_t out_degree() const;

    void add_input(std::shared_ptr<QueryNode> source);
    void add_output(std::shared_ptr<QueryNode> target);

    static void rewire(std::shared_ptr<QueryNode> source,
                       std::shared_ptr<QueryNode> target,
                       int source_out_port,
                       int target_in_port);

    std::vector<std::shared_ptr<QueryNode>> get_outputs() const;
    std::vector<std::shared_ptr<QueryNode>> get_inputs() const;
    std::vector<std::shared_ptr<QueryNode>>& get_inputs();

    std::vector<std::shared_ptr<Schema>> get_input_schemas() const;
    std::shared_ptr<Schema> get_output_schema() const;

    bool is_source() const;
    bool is_sink() const;
    bool last_before_root() const;

    bool is_query_plan_root() const;

    virtual void infer_types();
    void infer_types_deep();

    virtual void fuse();
    void fuse_deep();

    virtual void unfuse();
    void unfuse_deep();

    void collect_source_nodes(std::vector<std::shared_ptr<QueryNode>>& source_nodes);

    // handle nodes with multiple outputs by inserting a local broadcast node
    virtual void break_cycles();
    void break_cycles_deep();

    std::shared_ptr<Pipeline> create_pipelines(
        std::shared_ptr<QueryPlan>& root,
        int parent_op_id,
        std::unordered_map<int, std::shared_ptr<Pipeline>>& bcast_op_id_to_inner_pipeline);

    std::string to_string() const;
    std::string to_string_recursive(const std::string& prefix, bool isTail,
                                    std::unordered_set<const QueryNode*>& visited) const;

    std::shared_ptr<MaximusContext>& get_context();

    std::vector<std::shared_ptr<QueryNode>> inputs;
    std::vector<std::weak_ptr<QueryNode>> outputs;

    std::vector<Edge> in_edges;
    std::vector<Edge> out_edges;

    EngineType engine_type = EngineType::UNDEFINED;

    DeviceType device_type = DeviceType::UNDEFINED;

    NodeType logical_type = NodeType::UNDEFINED;

    std::shared_ptr<NodeProperties> properties;

protected:
    std::shared_ptr<AbstractOperator> operator_;

    std::shared_ptr<MaximusContext> ctx_;

private:
    void validate_inputs(std::size_t expected_in_degree, std::size_t expected_out_degree) const;
    void validate_outputs(std::size_t expected_in_degree, std::size_t expected_out_degree) const;

    void make_filter_operator();
    void make_distinct_operator();
    void make_project_operator();
    void make_hash_join_operator();
    void make_order_by_operator();
    void make_group_by_operator();
    void make_random_table_source_operator();
    void make_table_source_operator();
#ifdef MAXIMUS_WITH_DATASET_API
    void make_table_source_filter_project_operator();
#endif
    void make_table_sink_operator();
    void make_limit_operator();
    void make_fused_operator();
    void make_vector_join_operator();
    void make_project_vector_distance_operator();
    void make_local_broadcast_operator();
    void make_scatter_operator();
    void make_gather_operator();
    void make_limit_per_group_operator();
    void make_take_operator();

    template<typename DerivedProperties>
    DerivedProperties cast_properties(NodeProperties&& base) {
        // Attempt dynamic cast
        auto* derived_ptr = dynamic_cast<DerivedProperties*>(&base);

        // Check if the cast was successful
        if (!derived_ptr) {
            throw std::invalid_argument("Failed to cast to the specified type.");
        }

        // Return the casted object using std::move
        return std::move(*derived_ptr);
    }
};
}  // namespace maximus
