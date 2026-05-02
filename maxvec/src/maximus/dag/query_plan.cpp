#include <iostream>
#include <maximus/dag/query_plan.hpp>
#include <sstream>

namespace maximus {

QueryPlan::QueryPlan(std::shared_ptr<MaximusContext>& ctx) {
    ctx_         = ctx;
    logical_type = NodeType::QUERY_PLAN_ROOT;
    engine_type  = EngineType::NATIVE;
}

void QueryPlan::convert_to_physical() {
    assert(out_degree() == 0);
    assert(!operator_);
}

void QueryPlan::infer_types() {
    // std::cout << "QueryPlan:: inferring types." << std::endl;
    assert(!operator_);
}

void QueryPlan::fuse() {
    assert(!operator_);
}

void QueryPlan::unfuse() {
    assert(!operator_);
}

std::vector<std::shared_ptr<Pipeline>> QueryPlan::to_pipelines() {
    pipelines.clear();

    assert(!operator_);

    // std::cout << "Starting the inference!" << std::endl;
    // ====================================
    //    FUSING THE OPERATORS
    // ====================================
    /*
    std::cout << "====================================" << std::endl;
    std::cout << "FUSING THE PLAN OPERATORS" << std::endl;
    std::cout << "====================================" << std::endl;

     */

    // handle the cases when nodes have multiple outputs
    this->break_cycles_deep();

    if (ctx_->fusing_enabled) {
        this->fuse_deep();
    }

    /*
    std::cout << "====================================" << std::endl;
    std::cout << "THE PLAN HAS BEEN FUSED!!!!" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << this->to_string() << std::endl;
     */

    // ====================================
    //    TYPE INFERENCE (SCHEMAS)
    // ====================================
    this->infer_types_deep();

    // std::cout << "type inference finished." << std::endl;

    assert(!operator_);

    // ====================================
    //    LOGICAL->PHYSICAL QUERY PLAN
    // ====================================
    this->convert_to_physical_deep();

    assert(!operator_);

    // std::cout << "converted the operators to physical." << std::endl;

    // ====================================
    //    BREAKING DAG -> PIPELINES
    // ====================================
    auto root_base = shared_from_this();
    auto root      = std::dynamic_pointer_cast<QueryPlan>(root_base);
    // std::cout << "before dag broken into pipelines" << std::endl;
    std::unordered_map<int, std::shared_ptr<Pipeline>> bcast_op_id_to_inner_pipeline;
    std::ignore = create_pipelines(root, -1, bcast_op_id_to_inner_pipeline);

    /*
    std::cout << "dag broken into pipelines" << std::endl;

    std::cout << "#################################" << std::endl;
    std::cout << "#      PHYSICAL QUERY PLAN      #" << std::endl;
    std::cout << "#################################" << std::endl;
    std::cout << this->to_string() << std::endl;
    std::cout << "################################" << std::endl;

    std::cout << "#################################" << std::endl;
    std::cout << "#          PIPELINES            #" << std::endl;
    std::cout << "#################################" << std::endl;
    for (int i = 0; i < pipelines.size(); ++i) {
        auto& pipeline = pipelines[i];
        std::cout << pipeline->to_string() << std::endl;
        if (i < pipelines.size() - 1) {
            std::cout << "============================" << std::endl;
        }
    }
    */

    assert(!operator_);

    return pipelines;
}

void QueryPlan::add_pipeline(std::shared_ptr<Pipeline> pipeline) {
    // freeze the pipeline so that no more operators can be added to it.
    // and push it to the pipelines vector.
    pipelines.push_back(std::move(pipeline));
    assert(!operator_);
}

TablePtr QueryPlan::result() {
    assert(!operator_);
    assert(in_degree() == 1 && inputs.size() == 1);
    assert(out_degree() == 0 && outputs.size() == 0);

    assert(logical_type == NodeType::QUERY_PLAN_ROOT);

    auto& last_node = inputs[0];

    assert(last_node);

    assert(last_node->is_sink());

    if (last_node->is_sink()) {
        assert(last_node->get_operator());
        auto last_operator = last_node->get_operator();
        // assert(last_operator->has_more_batches());
        auto table = last_operator->export_table();
        // assert(table);
        return std::move(table);
    }

    return nullptr;
}


std::string QueryPlan::physical_plan() const {
    assert(pipelines.size() > 0);

    std::stringstream ss;
    std::cout << "###############################" << std::endl;
    std::cout << "#      PIPELINES CREATED      #" << std::endl;
    std::cout << "###############################" << std::endl;
    for (int i = 0; i < pipelines.size(); ++i) {
        auto& pipeline = pipelines[i];
        std::cout << pipeline->to_string() << std::endl;
        if (i < pipelines.size() - 1) {
            std::cout << "============================" << std::endl;
        }
    }

    return ss.str();
}
}  // namespace maximus
