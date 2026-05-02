#include <maximus/dag/query_node.hpp>
#include <maximus/dag/query_plan.hpp>
#include <maximus/frontend/query_plan_api.hpp>
#include <maximus/operators/engine.hpp>
#include <maximus/operators/properties.hpp>
#include <maximus/types/node_type.hpp>
#include <memory>

// acero operators
#include <maximus/operators/acero/distinct_operator.hpp>
#include <maximus/operators/acero/filter_operator.hpp>
#include <maximus/operators/acero/fused_operator.hpp>
#include <maximus/operators/acero/group_by_operator.hpp>
#include <maximus/operators/acero/hash_join_operator.hpp>
#include <maximus/operators/acero/limit_operator.hpp>
#include <maximus/operators/acero/order_by_operator.hpp>
#include <maximus/operators/acero/project_operator.hpp>

// native operators
#include <maximus/operators/native/limit_operator.hpp>
#include <maximus/operators/native/limit_per_group_operator.hpp>
#include <maximus/operators/native/local_broadcast_operator.hpp>
#include <maximus/operators/native/scatter_operator.hpp>
#include <maximus/operators/native/gather_operator.hpp>
#include <maximus/operators/native/take_operator.hpp>
#include <maximus/operators/native/random_table_source_operator.hpp>
#include <maximus/operators/native/table_sink_operator.hpp>
#ifdef MAXIMUS_WITH_DATASET_API
#include <maximus/operators/native/table_source_filter_project_operator.hpp>
#endif

#include <maximus/operators/native/table_source_operator.hpp>

#ifdef MAXIMUS_WITH_CUDA
#include <maximus/operators/gpu/cudf/distinct_operator.hpp>
#include <maximus/operators/gpu/cudf/filter_operator.hpp>
#include <maximus/operators/gpu/cudf/group_by_operator.hpp>
#include <maximus/operators/gpu/cudf/hash_join_operator.hpp>
#include <maximus/operators/gpu/cudf/limit_operator.hpp>
#include <maximus/operators/gpu/cudf/limit_per_group_operator.hpp>
#include <maximus/operators/gpu/cudf/local_broadcast_operator.hpp>
#include <maximus/operators/gpu/cudf/order_by_operator.hpp>
#include <maximus/operators/gpu/cudf/scatter_operator.hpp>
#include <maximus/operators/gpu/cudf/project_operator.hpp>
#include <maximus/operators/gpu/cudf/table_source_operator.hpp>
#include <maximus/operators/gpu/cudf/gather_operator.hpp>
#endif

#ifdef MAXIMUS_WITH_FAISS
#include <maximus/operators/faiss/join_exhaustive_operator.hpp>
#include <maximus/operators/faiss/join_indexed_operator.hpp>
#include <maximus/operators/faiss/project_distance_operator.hpp>

#ifdef MAXIMUS_WITH_CUDA
#include <maximus/operators/faiss/gpu/project_distance_operator.hpp>
#include <maximus/operators/faiss/gpu/join_exhaustive_operator.hpp>
#include <maximus/operators/faiss/gpu/join_indexed_operator.hpp>
#endif

#endif

namespace maximus {
QueryNode::QueryNode() = default;

QueryNode::QueryNode(const EngineType engine,
                     const DeviceType device,
                     const NodeType logical_type,
                     std::shared_ptr<NodeProperties> properties,
                     std::shared_ptr<MaximusContext> &ctx)
        : engine_type(engine)
        , device_type(device)
        , logical_type(logical_type)
        , properties(std::move(properties))
        , ctx_(ctx) {
}

QueryNode::~QueryNode() {
    // std::cout << "Destroying QueryNode of type " <<
    // node_type_to_string(logical_type) << std::endl; std::cout << "Destroying
    // QueryNode with operator_id = " << (operator_ ? ""+operator_->get_id() :
    // "NULL") << std::endl;
}

std::shared_ptr<AbstractOperator> &QueryNode::get_operator() {
    return operator_;
}

std::size_t QueryNode::in_degree() const {
    assert(inputs.size() == in_edges.size());
    return inputs.size();
}

std::size_t QueryNode::out_degree() const {
    assert(outputs.size() == out_edges.size());
    return out_edges.size();
}

bool QueryNode::is_source() const {
    return inputs.empty() && operator_;
}

bool QueryNode::is_sink() const {
    bool outputs_empty = outputs.empty() || last_before_root();
    return outputs_empty && operator_;
}

bool QueryNode::last_before_root() const {
    return outputs.size() == 1 && get_outputs()[0]->is_query_plan_root();
}

void QueryNode::add_input(std::shared_ptr<QueryNode> source) {
    // std::cout << "Before adding input to query node " <<
    // node_type_to_string(logical_type) << std::endl;
    assert(source);
    int source_port = source->out_degree();
    int target_port = in_degree();

    Edge e(source_port, target_port);

    auto shared_this = this->shared_from_this();

    source->outputs.push_back(shared_this);
    source->out_edges.push_back(e);

    this->inputs.push_back(source);
    this->in_edges.push_back(e);
    assert(in_edges.size() == in_degree());
    // std::cout << "After adding input to query node " <<
    // node_type_to_string(logical_type) << std::endl;
}

void QueryNode::add_output(std::shared_ptr<QueryNode> target) {
    assert(target);
    int source_port = out_degree();
    int target_port = target->in_degree();

    Edge e(source_port, target_port);

    this->outputs.push_back(target);

    auto shared_this = this->shared_from_this();
    target->inputs.push_back(shared_this);

    this->out_edges.push_back(e);
    target->in_edges.push_back(e);
}

void QueryNode::rewire(std::shared_ptr<QueryNode> source,
                       std::shared_ptr<QueryNode> target,
                       int src_out_port,
                       int target_in_port) {
    assert(source->out_degree() > src_out_port);
    assert(target->in_degree() > target_in_port);
    assert(source->outputs[src_out_port].lock());
    assert(target->inputs[target_in_port]);

    Edge e(src_out_port, target_in_port);

    source->outputs[src_out_port]   = target;
    source->out_edges[src_out_port] = e;

    target->inputs[target_in_port] = source, target->in_edges[target_in_port] = e;
}

std::vector<std::shared_ptr<QueryNode>> QueryNode::get_inputs() const {
    return inputs;
}

std::vector<std::shared_ptr<QueryNode>> &QueryNode::get_inputs() {
    return inputs;
}

std::vector<std::shared_ptr<QueryNode>> QueryNode::get_outputs() const {
    std::vector<std::shared_ptr<QueryNode>> shared_outputs(outputs.size());
    for (unsigned i = 0u; i < outputs.size(); ++i) {
        assert(!outputs[i].expired());

        shared_outputs[i] = outputs[i].lock();

        assert(shared_outputs[i]);
    }
    return shared_outputs;
}

void QueryNode::convert_to_physical() {
}

void QueryNode::convert_to_physical_deep() {
    // first recursively convert its inputs to physical nodes
    for (unsigned i = 0u; i < inputs.size(); ++i) {
        inputs[i]->convert_to_physical_deep();
    }
}

bool QueryNode::is_query_plan_root() const {
    if (logical_type == NodeType::QUERY_PLAN_ROOT) {
        // try to case the pointer to the QueryPlan pointer
        auto root_base = shared_from_this();
        assert(root_base);
        auto root = std::dynamic_pointer_cast<const QueryPlan>(root_base);
        assert(root);
        assert(!operator_);
        return true;
    }
    return false;
}

void QueryNode::validate_inputs(std::size_t expected_in_degree,
                                std::size_t expected_out_degree) const {
    assert(in_degree() == expected_in_degree);
    assert(out_degree() == expected_out_degree);

    auto input_schemas = get_input_schemas();
    assert(input_schemas.size() == in_degree());

    for (unsigned i = 0u; i < in_degree(); ++i) {
        assert(inputs[i] && "All inputs must be defined before calling infer_types()");
        assert(inputs[i]->get_output_schema() &&
               "All input schemas must be known before calling infer_types()");
        assert(input_schemas[i] && "All input schemas must be known before calling infer_types()");
        assert(*input_schemas[i] == *inputs[i]->get_output_schema());
        assert(input_schemas[i]->size() > 0 && "Schema size must be > 0");
    }
}

void QueryNode::validate_outputs(std::size_t expected_in_degree,
                                 std::size_t expected_out_degree) const {
    // validate the inputs again
    // to ensure nothing has moved (std::move)
    validate_inputs(expected_in_degree, expected_out_degree);

    assert(out_degree() == expected_out_degree);

    auto outputs = get_outputs();

    for (unsigned i = 0u; i < out_degree(); ++i) {
        assert(outputs[i] && "All outputs must be defined before calling infer_types()");
    }

    assert(get_output_schema() &&
           "The output schema must be known after the operator has been created");
    assert(get_output_schema()->size() > 0 && "Output schema size must be > 0");
}

void QueryNode::make_filter_operator() {
    validate_inputs(1, 1);

    // cast the properties to the correct derived type (FilterProperties)
    auto filter_properties = std::dynamic_pointer_cast<FilterProperties>(properties);

    assert(filter_properties && "Filter properties must be set before calling infer_types()");

    auto input_schema = get_input_schemas()[0];

    switch (engine_type) {
        case EngineType::ACERO:
            operator_ = std::make_shared<acero::FilterOperator>(
                ctx_, std::move(input_schema), std::move(filter_properties));
            break;
        case EngineType::CUDF:
#ifdef MAXIMUS_WITH_CUDA
            assert(ctx_ && ctx_->gcontext);
            operator_ = std::make_shared<cudf::FilterOperator>(
                ctx_, std::move(input_schema), std::move(filter_properties));
#else
            throw std::runtime_error("You have to rebuild Maximus with CUDA support");
#endif
            break;
        default:
            throw std::runtime_error("Unsupported engine type for the Filter Node");
    }

    // ensure the input schema is not moved
    validate_outputs(1, 1);
}

void QueryNode::make_distinct_operator() {
    validate_inputs(1, 1);

    // cast the properties to the correct derived type (FilterProperties)
    auto distinct_properties = std::dynamic_pointer_cast<DistinctProperties>(properties);

    assert(distinct_properties && "Distinct properties must be set before calling infer_types()");

    auto input_schema = get_input_schemas()[0];

    switch (engine_type) {
        case EngineType::ACERO:
            operator_ = std::make_shared<acero::DistinctOperator>(
                ctx_, std::move(input_schema), std::move(distinct_properties));
            break;
        case EngineType::CUDF:
#ifdef MAXIMUS_WITH_CUDA
            assert(ctx_ && ctx_->gcontext);
            operator_ = std::make_shared<cudf::DistinctOperator>(
                ctx_, std::move(input_schema), std::move(distinct_properties));
#else
            throw std::runtime_error("You have to rebuild Maximus with CUDA support");
#endif
            break;
        default:
            throw std::runtime_error("Unsupported engine type for the Distinct Node");
    }

    // ensure the input schema is not moved
    validate_outputs(1, 1);
}

void QueryNode::make_project_operator() {
    validate_inputs(1, 1);

    auto input_schema       = inputs[0]->get_output_schema();
    auto project_properties = std::dynamic_pointer_cast<ProjectProperties>(properties);
    assert(project_properties);

    switch (engine_type) {
        case EngineType::ACERO:
            operator_ = std::make_shared<acero::ProjectOperator>(
                ctx_, std::move(input_schema), std::move(project_properties));
            break;
        case EngineType::CUDF:
#ifdef MAXIMUS_WITH_CUDA
            assert(ctx_ && ctx_->gcontext);
            operator_ = std::make_shared<cudf::ProjectOperator>(
                ctx_, std::move(input_schema), std::move(project_properties));
#else
            throw std::runtime_error("You have to rebuild Maximus with CUDA support");
#endif
            break;
        default:
            throw std::runtime_error("Unsupported engine type for the Project Node");
    }

    // ensure the input schema is not moved
    validate_outputs(1, 1);
}

void QueryNode::make_limit_operator() {
    validate_inputs(1, 1);

    auto input_schema     = inputs[0]->get_output_schema();
    auto limit_properties = std::dynamic_pointer_cast<LimitProperties>(properties);
    assert(limit_properties);

    switch (engine_type) {
        case EngineType::ACERO:
            operator_ = std::make_shared<acero::LimitOperator>(
                ctx_, std::move(input_schema), std::move(limit_properties));
            break;
        case EngineType::NATIVE:
            operator_ = std::make_shared<native::LimitOperator>(
                ctx_, std::move(input_schema), std::move(limit_properties));
            break;
        case EngineType::CUDF:
#ifdef MAXIMUS_WITH_CUDA
            assert(ctx_ && ctx_->gcontext);
            operator_ = std::make_shared<cudf::LimitOperator>(
                ctx_, std::move(input_schema), std::move(limit_properties));
#else
            throw std::runtime_error("You have to rebuild Maximus with CUDA support");
#endif
            break;
        default:
            throw std::runtime_error("Unsupported engine type for the Limit Node");
    }

    // ensure the input schema is not moved
    validate_outputs(1, 1);
}

void QueryNode::make_vector_join_operator() {
    validate_inputs(2, 1);

    std::vector<std::shared_ptr<Schema>> schemas = {inputs[0]->get_output_schema(),
                                                    inputs[1]->get_output_schema()};
    auto join_indexed_properties =
        std::dynamic_pointer_cast<VectorJoinIndexedProperties>(properties);
    auto join_exhaustive_properties =
        std::dynamic_pointer_cast<VectorJoinExhaustiveProperties>(properties);
    assert((join_indexed_properties != nullptr) || (join_exhaustive_properties != nullptr));

    switch (engine_type) {
        case EngineType::FAISS:
#ifdef MAXIMUS_WITH_FAISS
            assert(ctx_);
            switch (device_type) {
                case DeviceType::GPU:
#ifdef MAXIMUS_WITH_CUDA
                    if (join_indexed_properties) {
                        operator_ = std::make_shared<maximus::faiss::gpu::JoinIndexedOperator>(
                            ctx_, std::move(schemas), std::move(join_indexed_properties));
                    } else {
                        assert(join_exhaustive_properties);
                        operator_ = std::make_shared<maximus::faiss::gpu::JoinExhaustiveOperator>(
                            ctx_, std::move(schemas), std::move(join_exhaustive_properties));
                    }
                break;
#else
                    throw std::runtime_error("You have to rebuild Maximus with FAISS and GPU support");
#endif
                case DeviceType::CPU:
                    if (join_indexed_properties)
                        operator_ = std::make_shared<maximus::faiss::JoinIndexedOperator>(
                            ctx_, std::move(schemas), std::move(join_indexed_properties));
                    else {
                        assert(join_exhaustive_properties);
                        operator_ = std::make_shared<maximus::faiss::JoinExhaustiveOperator>(
                            ctx_, std::move(schemas), std::move(join_exhaustive_properties));
                    }
                break;
            }
#else
            throw std::runtime_error("You have to rebuild Maximus with FAISS support");
#endif
            break;
        default:
            throw std::runtime_error("Unsupported engine type for the Vector Search Node");
    }

    // ensure the input schema is not moved
    validate_outputs(2, 1);
}

void QueryNode::make_project_vector_distance_operator() {
    validate_inputs(2, 1);

    std::vector<std::shared_ptr<Schema>> schemas = {inputs[0]->get_output_schema(),
                                                    inputs[1]->get_output_schema()};
    auto project_distance_properties =
        std::dynamic_pointer_cast<VectorProjectDistanceProperties>(properties);
    assert(properties);

    switch (engine_type) {
        case EngineType::FAISS:
#ifdef MAXIMUS_WITH_FAISS
            assert(ctx_);
            switch (device_type) {
                case DeviceType::GPU:
#ifdef MAXIMUS_WITH_CUDA
                    operator_ = std::make_shared<maximus::faiss::gpu::ProjectDistanceOperator>(
                        ctx_, std::move(schemas), std::move(project_distance_properties));
                break;
#else
                    throw std::runtime_error("You have to rebuild Maximus with FAISS and GPU support");
#endif
                case DeviceType::CPU:
                    operator_ = std::make_shared<maximus::faiss::ProjectDistanceOperator>(
                        ctx_, std::move(schemas), std::move(project_distance_properties));
                break;
            }
#else
            throw std::runtime_error("You have to rebuild Maximus with FAISS support");
#endif
            break;
        default:
            throw std::runtime_error("Unsupported engine type for the Vector Search Node");
    }

    // ensure the input schema is not moved
    validate_outputs(2, 1);
}

void QueryNode::make_local_broadcast_operator() {
    validate_inputs(1, out_degree());
    auto input_schema         = inputs[0]->get_output_schema();
    auto broadcast_properties = std::dynamic_pointer_cast<LocalBroadcastProperties>(properties);
    assert(broadcast_properties);

    // assert(engine_type == EngineType::NATIVE);

    switch (engine_type) {
        case EngineType::NATIVE:
            operator_ = std::make_shared<native::LocalBroadcastOperator>(
                ctx_, std::move(input_schema), std::move(broadcast_properties));
            break;
        case EngineType::CUDF:
#ifdef MAXIMUS_WITH_CUDA
            assert(ctx_ && ctx_->gcontext);
            operator_ = std::make_shared<cudf::LocalBroadcastOperator>(
                ctx_, std::move(input_schema), std::move(broadcast_properties));
#else
            throw std::runtime_error("You have to rebuild Maximus with CUDA support");
#endif
            break;
        default:
            throw std::runtime_error("Unsupported engine type for the LocalBroadcast Node");
    }
}

void QueryNode::make_scatter_operator() {
    validate_inputs(1, out_degree());
    auto input_schema = inputs[0]->get_output_schema();
    auto scatter_properties = std::dynamic_pointer_cast<ScatterProperties>(properties);
    assert(scatter_properties);

    switch (engine_type) {
        case EngineType::NATIVE:
            operator_ = std::make_shared<native::ScatterOperator>(
                ctx_, std::move(input_schema), std::move(scatter_properties));
            break;
#ifdef MAXIMUS_WITH_CUDA
        case EngineType::CUDF:
            assert(ctx_ && ctx_->get_gpu_context());
            operator_ = std::make_shared<cudf::ScatterOperator>(
                ctx_, std::move(input_schema), std::move(scatter_properties));
            break;
#endif
        default:
            throw std::runtime_error("Unsupported engine type for the Scatter Node");
    }

    // Multi-output like LocalBroadcast - don't validate single output
}

void QueryNode::make_gather_operator() {
    validate_inputs(in_degree(), 1);
    auto input_schema = inputs[0]->get_output_schema();
    auto gather_properties = std::dynamic_pointer_cast<GatherProperties>(properties);
    assert(gather_properties);

    switch (engine_type) {
        case EngineType::NATIVE:
            operator_ = std::make_shared<native::GatherOperator>(
                ctx_, std::move(input_schema), std::move(gather_properties));
            break;
#ifdef MAXIMUS_WITH_CUDA
        case EngineType::CUDF:
            assert(ctx_ && ctx_->get_gpu_context());
            operator_ = std::make_shared<cudf::GatherOperator>(
                ctx_, std::move(input_schema), std::move(gather_properties));
            break;
#endif
        default:
            throw std::runtime_error("Unsupported engine type for the Gather Node");
    }

    validate_outputs(in_degree(), 1);
}


void QueryNode::make_limit_per_group_operator() {
    validate_inputs(1, 1);

    auto input_schema = inputs[0]->get_output_schema();
    auto lpg_properties = std::dynamic_pointer_cast<LimitPerGroupProperties>(properties);
    assert(lpg_properties);

    switch (engine_type) {
        case EngineType::NATIVE:
            operator_ = std::make_shared<native::LimitPerGroupOperator>(
                ctx_, std::move(input_schema), std::move(lpg_properties));
            break;
#ifdef MAXIMUS_WITH_CUDA
        case EngineType::CUDF:
            assert(ctx_ && ctx_->get_gpu_context());
            operator_ = std::make_shared<cudf::LimitPerGroupOperator>(
                ctx_, std::move(input_schema), std::move(lpg_properties));
            break;
#endif
        default:
            throw std::runtime_error("Unsupported engine type for the LimitPerGroup Node");
    }

    validate_outputs(1, 1);
}

void QueryNode::make_take_operator() {
    validate_inputs(2, 1);

    auto data_schema = inputs[0]->get_output_schema();
    auto index_schema = inputs[1]->get_output_schema();

    auto take_properties = std::dynamic_pointer_cast<TakeProperties>(properties);
    assert(take_properties && "Take properties must be set before calling infer_types()");

    switch (engine_type) {
        case EngineType::NATIVE:
            operator_ = std::make_shared<native::TakeOperator>(
                ctx_, std::move(data_schema), std::move(index_schema),
                std::move(take_properties));
            break;
        default:
            throw std::runtime_error(
                "Unsupported engine type for the Take Node (CPU/NATIVE only)");
    }

    validate_outputs(2, 1);
}


void QueryNode::make_fused_operator() {
    validate_inputs(1, 1);

    auto input_schema     = inputs[0]->get_output_schema();
    auto fused_properties = std::dynamic_pointer_cast<FusedProperties>(properties);
    assert(fused_properties);

    switch (engine_type) {
        case EngineType::ACERO:
            operator_ = std::make_shared<acero::FusedOperator>(
                ctx_, std::move(input_schema), std::move(fused_properties));
            break;
            //     case EngineType::CUDF:
            // #ifdef MAXIMUS_WITH_CUDA
            //         assert(ctx_ && ctx_->gcontext);
            //         operator_ = std::make_shared<cudf::FusedOperator>(
            //             ctx_, std::move(input_schema),
            //             std::move(fused_properties));
            // #else
            //         throw std::runtime_error(
            //             "You have to rebuild Maximus with CUDA support");
            // #endif
            //         break;
        default:
            throw std::runtime_error("Unsupported engine type for the Fused Node");
    }

    // ensure the input schema is not moved
    validate_outputs(1, 1);
}

void QueryNode::make_hash_join_operator() {
    validate_inputs(2, 1);

    auto left_schema  = inputs[0]->get_output_schema();
    auto right_schema = inputs[1]->get_output_schema();

    auto join_properties = std::dynamic_pointer_cast<JoinProperties>(properties);
    assert(join_properties);

    switch (engine_type) {
        case EngineType::ACERO:
            operator_ = std::make_shared<acero::HashJoinOperator>(
                ctx_, std::move(left_schema), std::move(right_schema), std::move(join_properties));
            break;
        case EngineType::CUDF:
#ifdef MAXIMUS_WITH_CUDA
            assert(ctx_ && ctx_->gcontext);
            operator_ = std::make_shared<cudf::HashJoinOperator>(
                ctx_, std::move(left_schema), std::move(right_schema), std::move(join_properties));
#else
            throw std::runtime_error("You have to rebuild Maximus with the CUDA support");
#endif
            break;
        default:
            throw std::runtime_error("Unsupported engine type for the Hash-Join Node");
    }

    // ensure the input schema is not moved
    validate_outputs(2, 1);
}

void QueryNode::make_order_by_operator() {
    validate_inputs(1, 1);

    auto input_schema = inputs[0]->get_output_schema();
    assert(input_schema);
    auto order_by_properties = std::dynamic_pointer_cast<OrderByProperties>(properties);
    assert(order_by_properties);

    switch (engine_type) {
        case EngineType::ACERO:
            operator_ = std::make_shared<acero::OrderByOperator>(
                ctx_, std::move(input_schema), std::move(order_by_properties));
            break;
        case EngineType::CUDF:
#ifdef MAXIMUS_WITH_CUDA
            assert(ctx_ && ctx_->gcontext);
            operator_ = std::make_shared<cudf::OrderByOperator>(
                ctx_, std::move(input_schema), std::move(order_by_properties));
#else
            throw std::runtime_error("You have to rebuild Maximus with CUDA support");
#endif
            break;
        default:
            throw std::runtime_error("Unsupported engine type for the OrderBy Node");
    }

    // ensure the input schema is not moved
    validate_outputs(1, 1);
}

void QueryNode::make_group_by_operator() {
    validate_inputs(1, 1);

    assert(inputs.size() > 0);  // to check that inputs has at least 1 operator
    assert(inputs[0]);          // to check that inputs[0] is not null
    auto input_schema = inputs[0]->get_output_schema();
    assert(input_schema);  // to check that the schema is not null
    auto group_by_properties = std::dynamic_pointer_cast<GroupByProperties>(properties);

    assert(group_by_properties);

    switch (engine_type) {
        case EngineType::ACERO:
            operator_ = std::make_shared<acero::GroupByOperator>(
                ctx_, std::move(input_schema), std::move(group_by_properties));
            break;
        case EngineType::CUDF:
#ifdef MAXIMUS_WITH_CUDA
            assert(ctx_ && ctx_->gcontext);
            operator_ = std::make_shared<cudf::GroupByOperator>(
                ctx_, std::move(input_schema), std::move(group_by_properties));
#else
            throw std::runtime_error("You have to rebuild Maximus with the CUDA support");
#endif
            break;
        default:
            throw std::runtime_error("Unsupported engine type for the GroupBy Node");
    }

    // ensure the input schema is not moved
    validate_outputs(1, 1);
}

void QueryNode::make_random_table_source_operator() {
    validate_inputs(0, 1);

    auto random_scan_properties =
        std::dynamic_pointer_cast<RandomTableSourceProperties>(properties);
    assert(random_scan_properties);

    auto input_schema = random_scan_properties->output_schema;

    operator_ = std::make_shared<RandomTableSourceOperator>(
        ctx_, std::move(input_schema), std::move(random_scan_properties));

    // ensure the input schema is not moved
    validate_outputs(0, 1);
}

void QueryNode::make_table_source_operator() {
    validate_inputs(0, 1);

    auto table_source_properties = std::dynamic_pointer_cast<TableSourceProperties>(properties);

    assert(table_source_properties);

    switch (engine_type) {
        case EngineType::NATIVE:
            operator_ = std::make_shared<native::TableSourceOperator>(
                ctx_, std::move(table_source_properties));
            break;
        case EngineType::CUDF:
#ifdef MAXIMUS_WITH_CUDA
            operator_ = std::make_shared<cudf::TableSourceOperator>(
                ctx_, std::move(table_source_properties));
#else
            throw std::runtime_error("You have to rebuild Maximus with the CUDA support");
#endif
            break;
        default:
            throw std::runtime_error("Unsupported engine type for the TableSource Node");
    }

    // check both the inputs and outputs
    validate_outputs(0, 1);
}

#ifdef MAXIMUS_WITH_DATASET_API
void QueryNode::make_table_source_filter_project_operator() {
    validate_inputs(0, 1);

    auto table_source_properties =
        std::dynamic_pointer_cast<TableSourceFilterProjectProperties>(properties);

    assert(table_source_properties);

    switch (engine_type) {
        case EngineType::NATIVE:
            operator_ = std::make_shared<native::TableSourceFilterProjectOperator>(
                ctx_, std::move(table_source_properties));
            break;
        default:
            throw std::runtime_error("Unsupported engine type for the TableSource Node");
    }

    // check both the inputs and outputs
    validate_outputs(0, 1);
}
#endif

void QueryNode::make_table_sink_operator() {
    validate_inputs(1, 1);

    auto input_schema          = inputs[0]->get_output_schema();
    auto table_sink_properties = std::dynamic_pointer_cast<TableSinkProperties>(properties);
    operator_ =
        std::make_shared<TableSinkOperator>(ctx_, input_schema, std::move(table_sink_properties));
    assert(operator_ && "TableSinkOperator must be created");

    validate_outputs(1, 1);
}

void QueryNode::infer_types() {
    assert(logical_type != NodeType::UNDEFINED &&
           "Logical type must be set before calling infer_types()");

    assert(inputs.size() == in_degree());

    assert(properties && "Properties must be set before calling infer_types()");

    // std::cout << "Before inferring types for node: " <<
    // node_type_to_string(logical_type) << "\n";

    switch (logical_type) {
        case NodeType::FILTER: {
            make_filter_operator();
            break;
        }
        case NodeType::DISTINCT: {
            make_distinct_operator();
            break;
        }
        case NodeType::PROJECT: {
            make_project_operator();
            break;
        }
        case NodeType::HASH_JOIN: {
            make_hash_join_operator();
            break;
        }
        case NodeType::ORDER_BY: {
            make_order_by_operator();
            break;
        }
        case NodeType::GROUP_BY: {
            make_group_by_operator();
            break;
        }
        case NodeType::LIMIT: {
            make_limit_operator();
            break;
        }
        case NodeType::RANDOM_TABLE_SOURCE: {
            make_random_table_source_operator();
            break;
        }
        case NodeType::TABLE_SOURCE: {
            make_table_source_operator();
            break;
        }
#ifdef MAXIMUS_WITH_DATASET_API
        case NodeType::TABLE_SOURCE_FILTER_PROJECT: {
            make_table_source_filter_project_operator();
            break;
        }
#endif
        case NodeType::TABLE_SINK: {
            make_table_sink_operator();
            break;
        }
        case NodeType::FUSED: {
            make_fused_operator();
            break;
        }
        case NodeType::VECTOR_JOIN: {
            make_vector_join_operator();
            break;
        }
        case NodeType::VECTOR_PROJECT_DISTANCE: {
            make_project_vector_distance_operator();
            break;
        }
        case NodeType::LOCAL_BROADCAST: {
            make_local_broadcast_operator();
            break;
        }
        case NodeType::SCATTER: {
            make_scatter_operator();
            break;
        }
        case NodeType::GATHER: {
            make_gather_operator();
            break;
        }
        case NodeType::LIMIT_PER_GROUP: {
            make_limit_per_group_operator();
            break;
        }
        case NodeType::TAKE: {
            make_take_operator();
            break;
        }
        case NodeType::QUERY_PLAN_ROOT: {
            break;
        }
        default:
            throw std::runtime_error("Unsupported logical operator type");
    }

    assert(operator_);
    assert(get_output_schema());
    // std::cout << "After inferring types for node: " <<
    // node_type_to_string(logical_type) << ", with id = " <<
    // operator_->get_id() << "\n";

    // std::cout << "output schema = " << get_output_schema()->to_string() <<
    // std::endl;
}

void QueryNode::infer_types_deep() {
    // NOTE: SCATTER is a shared node, unlike the LOCAL_BROADCAST
    // Skip if already processed (important for shared nodes like SCATTER)
    if (operator_ != nullptr) {
        return;
    }
    
    // std::cout << "Num of inputs = " << inputs.size() << std::endl;
    for (unsigned i = 0u; i < inputs.size(); ++i) {
        assert(inputs[i]);
        inputs[i]->infer_types_deep();
    }
    this->infer_types();

    // schema must be known after calling this function
    assert(is_query_plan_root() || get_output_schema());
}

void QueryNode::fuse_deep() {
    if (inputs.size() == 0) return;

    // keep the pointer, so that this doesn't get destroyed
    auto reserve_ptr = shared_from_this();

    for (unsigned i = 0u; i < inputs.size(); ++i) {
        assert(inputs[i]);
        inputs[i]->fuse_deep();
    }

    fuse();
}

void QueryNode::unfuse_deep() {
    for (unsigned i = 0u; i < inputs.size(); ++i) {
        assert(inputs[i]);
        inputs[i]->unfuse_deep();
    }
    this->unfuse();
}

void QueryNode::fuse() {
    if (logical_type == NodeType::FUSED) return;
    if (logical_type == NodeType::UNDEFINED) return;
    if (engine_type != EngineType::ACERO && engine_type != EngineType::NATIVE) return;

    // std::cout << "Fusing node:\n" << this->to_string() << std::endl;
    assert(logical_type != NodeType::UNDEFINED && "Logical type must be set before calling fuse()");
    assert(logical_type != NodeType::FUSED &&
           "Logical type must not be FUSED before calling fuse()");

    if (in_degree() != 1) return;

    // fusing can only be done if the current node has a single input and a
    // single output and if the nodes belong to the same engine (a source node
    // with 0-inputs will be fused by the next node, if they belong to the same
    // engine)
    assert(inputs.size() == 1);
    assert(inputs[0]);

    // the input and output degrees of all fused nodes must be 1
    if (this->inputs[0]->out_degree() != 1 || out_degree() > 1) return;

    // if the previous node is a table source
    bool prev_table_source = this->inputs[0]->logical_type == NodeType::TABLE_SOURCE;
    // if the previous node is a table source, we only care if it is not an in-memory table or loaded from a file
    if (prev_table_source)
        prev_table_source = prev_table_source && !std::dynamic_pointer_cast<TableSourceProperties>(
                                                      this->inputs[0]->properties)
                                                      ->has_table();
#ifdef MAXIMUS_WITH_DATASET_API
    bool merging_table_source =
        ctx_->dataset_api &&
        (
            /* If we have table source -> filter */
            (prev_table_source && this->logical_type == NodeType::FILTER) ||
            /* If we have table source -> project */
            (prev_table_source && this->logical_type == NodeType::PROJECT) ||
            /* If we have table source project filter (with filter only) -> project */
            (this->inputs[0]->logical_type == NodeType::TABLE_SOURCE_FILTER_PROJECT &&
             this->logical_type == NodeType::PROJECT &&
             std::dynamic_pointer_cast<TableSourceFilterProjectProperties>(
                 this->inputs[0]->properties)
                 ->is_filter_only()));
#endif

    // now we know that the current node and the input node:
    // - have the same engine type (or we're building a table source filter project)
    // - inputs[0] has a single output (= the current node) and the current node
    // has at most 1 output node
    // - it doesn't matter if they are pipeline breakers or not

    assert(this->out_degree() <= 1);
    assert(this->inputs[0]->out_degree() == 1);

#ifdef MAXIMUS_WITH_DATASET_API
    // fuse the current node with inputs[0]
    if (merging_table_source) {
        // if the previous node was already a table source filter project, just
        // include the project properties from the current node.
        if (this->inputs[0]->logical_type == NodeType::TABLE_SOURCE_FILTER_PROJECT) {
            // The current node must be a project
            assert(this->logical_type == NodeType::PROJECT);
            if (this->engine_type != EngineType::NATIVE && this->engine_type != EngineType::ACERO)
                return;

            auto curr_properties   = std::dynamic_pointer_cast<ProjectProperties>(this->properties);
            auto source_properties = std::dynamic_pointer_cast<TableSourceFilterProjectProperties>(
                this->inputs[0]->properties);
            source_properties->exprs        = curr_properties->project_expressions;
            source_properties->column_names = curr_properties->column_names;
        } else {
            auto source_properties =
                std::dynamic_pointer_cast<TableSourceProperties>(this->inputs[0]->properties);
            std::vector<std::string> paths = {source_properties->path};
            auto properties                = std::make_shared<TableSourceFilterProjectProperties>(
                paths, std::move(source_properties->schema), source_properties->include_columns);

            if (this->logical_type == NodeType::FILTER) {
                auto filter_properties =
                    std::dynamic_pointer_cast<FilterProperties>(this->properties);
                properties->filter_expr = filter_properties->filter_expression;
            } else if (this->logical_type == NodeType::PROJECT) {
                auto project_properties =
                    std::dynamic_pointer_cast<ProjectProperties>(this->properties);
                properties->exprs        = project_properties->project_expressions;
                properties->column_names = project_properties->column_names;
            }

            this->inputs[0]->properties   = std::move(properties);
            this->inputs[0]->logical_type = NodeType::TABLE_SOURCE_FILTER_PROJECT;
        }
    } else
#endif
    {
        // if the input and output nodes belong to different engines, don't fuse
        if (this->inputs[0]->engine_type != engine_type) return;
        if (engine_type != EngineType::ACERO) return;

        if (inputs[0]->logical_type == NodeType::FUSED) {
            // if the previous node is already fuse, add the current node to it
            auto fused_properties =
                std::dynamic_pointer_cast<FusedProperties>(this->inputs[0]->properties);
            assert(fused_properties);
            fused_properties->properties.emplace_back(properties);
            fused_properties->node_types.emplace_back(logical_type);
        } else {
            if (this->inputs[0]->in_degree() > 1) return;
            // overwrite the properties and the logical type of the input node to
            // become FUSED
            std::vector<std::shared_ptr<NodeProperties>> properties = {inputs[0]->properties,
                                                                       this->properties};
            std::vector<NodeType> node_types = {this->inputs[0]->logical_type, logical_type};
            auto fused_properties =
                std::make_shared<FusedProperties>(std::move(properties), std::move(node_types));
            this->inputs[0]->logical_type = NodeType::FUSED;
            this->inputs[0]->properties   = std::move(fused_properties);
            // inputs[0]->infer_types();
        }
    }

    auto outputs = get_outputs();

    if (out_degree() == 1) {  // it must be 1 at this point
        auto next      = outputs[0];
        auto next_edge = out_edges[0];
        auto next_port = next_edge.target_port;

        // std::cout << "before rewiring = \n";
        // std::cout << this->to_string() << std::endl;
        rewire(inputs[0], next, 0, next_port);
        // std::cout << "after rewiring = \n";
        // std::cout << next->to_string() << std::endl;
    } else {
        // remove all outputs
        assert(out_degree() == 0);
        this->inputs[0]->outputs.pop_back();
        this->inputs[0]->out_edges.pop_back();
    }

    // detach the current node from the rest of the plan
    this->logical_type = NodeType::UNDEFINED;
    this->inputs.clear();
    this->outputs.clear();
    this->in_edges.clear();
    this->out_edges.clear();
}

void QueryNode::unfuse() {
    if (logical_type != NodeType::FUSED) return;

    auto fused_properties = std::dynamic_pointer_cast<FusedProperties>(properties);
    auto properties_vec   = fused_properties->properties;
    auto node_types_vec   = fused_properties->node_types;

    auto output_node   = out_degree() > 0 ? get_outputs()[0] : nullptr;
    auto output_port   = out_degree() > 0 ? out_edges[0].target_port : -1;
    auto output_degree = out_degree();

    assert(output_node);
    assert(output_port >= 0);

    properties   = properties_vec[0];
    logical_type = node_types_vec[0];

    assert(logical_type != NodeType::LIMIT);

    std::shared_ptr<QueryNode> prev = shared_from_this();  // output_node->inputs[output_port];
    outputs.clear();
    out_edges.clear();

    for (unsigned i = 1u; i < properties_vec.size(); ++i) {
        // The limit node is specific as Acero doesn't allow it to be used in
        // isolation. Acero requires to have a guaranteed ordering in the input
        // to the Limit operator. When used in isolation, Acero doesn't know the
        // previous node and thus throws an error. For this reason, when
        // unfusing the operators, we have to switch to the native limit
        // operator.
        auto new_engine_type =
            node_types_vec[i] == NodeType::LIMIT ? EngineType::NATIVE : engine_type;
        auto node = std::make_shared<QueryNode>(
            new_engine_type, device_type, node_types_vec[i], properties_vec[i], ctx_);
        // std::cout << "Adding input from \n" << prev->to_string() << " to \n"
        // << node->to_string() << std::endl;
        node->add_input(prev);
        // node->infer_types();
        prev = node;
        // std::cout << "prev = \n" << prev->to_string() << std::endl;
    }

    output_node->inputs[output_port] = prev;
    Edge e(0, output_port);
    output_node->in_edges[output_port] = e;

    assert(prev->out_degree() == 0);
    prev->outputs.push_back(output_node);
    prev->out_edges.push_back(e);

    assert(output_port >= 0);
    // rewire(prev, output_node, 0, output_port);

    // std::cout << "Final = \n" << output_node->to_string() << std::endl;

    output_node->infer_types_deep();
    output_node->convert_to_physical_deep();

    // std::cout << "Final after inferring types\n" << output_node->to_string()
    // << std::endl; output_node->convert_to_physical_deep();
}

void QueryNode::break_cycles() {
    // Skip nodes that are already multi-output operators:
    // - LOCAL_BROADCAST: replicates data to all outputs
    // - SCATTER: scatters data across outputs by key
    if (out_degree() > 1 &&
        logical_type != NodeType::LOCAL_BROADCAST &&
        logical_type != NodeType::SCATTER) {
        auto outputs = get_outputs();
        assert(outputs.size() > 1);
        assert(outputs[0]);
        auto engine =
            is_gpu_engine(outputs[0]->engine_type) ? EngineType::CUDF : EngineType::NATIVE;
        // auto engine  = engine_type == EngineType::CUDF ? EngineType::CUDF : EngineType::NATIVE;

        auto bcast_properties = std::make_shared<LocalBroadcastProperties>(outputs.size());

        auto bcast_node = std::make_shared<QueryNode>(
            engine, device_type, NodeType::LOCAL_BROADCAST, std::move(bcast_properties), ctx_);

        // connect the bcast_node to same outputs as the current node
        bcast_node->out_edges = this->out_edges;
        bcast_node->outputs   = this->outputs;

        this->out_edges.clear();
        this->outputs.clear();

        bcast_node->add_input(shared_from_this());

        outputs = bcast_node->get_outputs();

        for (unsigned i = 0u; i < outputs.size(); ++i) {
            assert(bcast_node->out_edges[i].source_port == i);
            rewire(bcast_node,
                   outputs[i],
                   bcast_node->out_edges[i].source_port,
                   bcast_node->out_edges[i].target_port);
        }
    }
}

void QueryNode::break_cycles_deep() {
    for (unsigned i = 0u; i < inputs.size(); ++i) {
        assert(inputs[i]);
        inputs[i]->break_cycles_deep();
    }
    this->break_cycles();
}

// this function collects all source nodes
void QueryNode::collect_source_nodes(std::vector<std::shared_ptr<QueryNode>> &source_nodes) {
    if (this->is_source()) {
        source_nodes.push_back(shared_from_this());
    } else {
        for (unsigned i = 0u; i < this->in_degree(); ++i) {
            assert(this->inputs[i]);
            this->inputs[i]->collect_source_nodes(source_nodes);
        }
    }
}

std::string node_type_to_string(NodeType type, std::shared_ptr<AbstractOperator> &op) {
    std::stringstream ss;
    ss << node_type_to_string(type);
    ss << "(" << (op ? op->to_string(14) : "null") << ")" << std::endl;
    return ss.str();
}

std::string QueryNode::to_string_recursive(const std::string &prefix, bool isTail,
                                           std::unordered_set<const QueryNode*>& visited) const {
    std::string result = prefix;

    result += is_query_plan_root() ? "" : (isTail ? "└── " : "├── ");

    // Only deduplicate SCATTER subtrees (scatter -> repeated subplan -> gather)
    bool already_visited = (logical_type == NodeType::SCATTER) && visited.count(this) > 0;
    
    // Append information about the current node
    std::string op_extra = operator_ ? operator_->to_string_extra() : "";
    if (!op_extra.empty()) {
        op_extra = "(" + op_extra + ")";
    }
    
    // Add node ID to help identify shared nodes
    std::string node_id = operator_ ? "[" + std::to_string(operator_->get_id()) + "]" : "";
    
    // For FAISS engine, SCATTER, and GATHER, include device type to distinguish CPU vs GPU execution
    std::string engine_str = engine_type_to_string(engine_type);
    if (engine_type == EngineType::FAISS) {
        engine_str += "::" + device_type_to_string(device_type);
    }
    std::string node_type_str = node_type_to_string(logical_type);
    if (logical_type == NodeType::VECTOR_JOIN) {
        node_type_str = "VECTOR_SEARCH";
    }
    result += engine_str + "::" + node_type_str +
              op_extra + node_id;
    
    // If already visited, indicate it's a reference to an earlier node
    if (already_visited) {
        result += " (see above)\n";
        return result;
    }
    
    result += "\n";
    visited.insert(this);

    // Append properties or other information about the node if needed

    // Recursive call for each output node
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto &input = inputs[i];
        assert(input);
        result +=
            input->to_string_recursive(prefix + (isTail ? "    " : "│   "), i == inputs.size() - 1, visited);
    }

    return result;
}

std::string QueryNode::to_string() const {
    std::unordered_set<const QueryNode*> visited;
    return to_string_recursive("", true, visited);
}

std::vector<std::shared_ptr<Schema>> QueryNode::get_input_schemas() const {
    std::vector<std::shared_ptr<Schema>> schemas;
    for (unsigned i = 0u; i < in_degree(); ++i) {
        assert(inputs[i] && "inputs[i] is null");
        assert(inputs[i]->get_output_schema() && "inputs[i]->get_output_schema() is null");
        schemas.push_back(inputs[i]->get_output_schema());
    }

    return std::move(schemas);
}

std::shared_ptr<Schema> QueryNode::get_output_schema() const {
    assert(operator_);
    return operator_->output_schema;
}

std::shared_ptr<Pipeline> QueryNode::create_pipelines(
    std::shared_ptr<QueryPlan> &root,
    int parent_op_id,
    std::unordered_map<int, std::shared_ptr<Pipeline>> &bcast_op_id_to_inner_pipeline) {
    assert(root && "root is null");
    // if the current node is a root, we just propagate the call, but don't
    // return any pipelines
    if (is_query_plan_root()) {
        assert(in_degree() == 1);
        assert(inputs[0]->get_operator());
        std::ignore = inputs[0]->create_pipelines(root, -1, bcast_op_id_to_inner_pipeline);
        return nullptr;
    }

    assert(operator_);

    // if the current node has no inputs, create a new pipeline, with it as the
    // source
    if (is_source()) {
        std::shared_ptr<Pipeline> pipeline = std::make_shared<Pipeline>();
        assert(out_degree() == 1);
        // every operator has at most 1 output, except of the broadcast
        // operator, which has multiple outputs
        pipeline->add_operator(operator_);  // out_edges[0].source_port);
        root->add_pipeline(pipeline);

        return pipeline;
    }

    // if the current node is a broadcast or partition operator, create a new pipeline for
    // each output
    if (out_degree() > 1) {
        assert(operator_->type == PhysicalOperatorType::LOCAL_BROADCAST ||
               operator_->type == PhysicalOperatorType::SCATTER);

        int output_idx  = -1;
        auto my_outputs = get_outputs();
        for (unsigned i = 0u; i < out_degree(); ++i) {
            if (my_outputs[i]->get_operator()->get_id() == parent_op_id) {
                output_idx = i;
                break;
            }
        }
        assert(output_idx >= 0);
        assert(in_degree() == 1);

        std::shared_ptr<Pipeline> pipeline = std::make_shared<Pipeline>();
        pipeline->add_operator(operator_);
        root->add_pipeline(pipeline);

        std::shared_ptr<Pipeline> inner_pipeline;
        auto pipeline_iter = bcast_op_id_to_inner_pipeline.find(operator_->get_id());
        // if this is the first time we are creating inner pipelines for this
        // broadcast operator then create and cache it in
        // bcast_op_id_to_inner_pipeline
        if (pipeline_iter == bcast_op_id_to_inner_pipeline.end()) {
            inner_pipeline = inputs[0]->create_pipelines(
                root, operator_->get_id(), bcast_op_id_to_inner_pipeline);
            inner_pipeline->add_operator(operator_, in_edges[0]);
            bcast_op_id_to_inner_pipeline[operator_->get_id()] = inner_pipeline;
        } else {
            inner_pipeline = pipeline_iter->second;
        }

        inner_pipeline->add_child_pipeline(pipeline);

        return pipeline;
    }

    // now we know that the current node is not a query root and that it has
    // inputs it also has to have an output, (at least the query root will be
    // the output)
    assert(in_degree() > 0);
    assert(out_degree() == 1);

    // std::cout << "Creating pipelines for in_degree = " << in_degree() <<
    // std::endl;
    std::vector<std::shared_ptr<Pipeline>> blocking_pipelines;
    std::vector<std::shared_ptr<Pipeline>> streaming_pipelines;

    // first create all child pipelines and separate them into
    // blocking pipelines and streaming pipelines
    for (unsigned i = 0u; i < in_degree(); ++i) {
        assert(inputs[i] && "inputs[i] is null");
        auto child_pipeline =
            inputs[i]->create_pipelines(root, operator_->get_id(), bcast_op_id_to_inner_pipeline);
        assert(child_pipeline && "child_pipeline is null");
        // we want to add the current node to both streaming and blocking
        // pipelines
        assert(operator_);
        child_pipeline->add_operator(operator_, in_edges[i]);
        if (operator_ && operator_->port_types[i] == PortType::BLOCKING) {
            blocking_pipelines.push_back(child_pipeline);
        } else {
            streaming_pipelines.push_back(child_pipeline);
        }
    }

    /*
    std::string op = operator_ ? physical_operator_to_string(operator_->type) :
    "ROOT"; std::cout << op << ": blocking pipelines: " <<
    blocking_pipelines.size() << std::endl; std::cout << op << ": streaming
    pipelines: " << streaming_pipelines.size() << std::endl;
    */

    // the blocking pipelines have to be executed before the streaming pipelines
    for (unsigned i = 0u; i < streaming_pipelines.size(); ++i) {
        // if there are any blocking pipelines, set them as parent pipelines
        // since the blocking pipelines have to be executed before the
        // streaming pipelines with the same parent
        for (unsigned j = 0u; j < blocking_pipelines.size(); ++j) {
            assert(streaming_pipelines[i] && "streaming_pipelines[i] is null");
            assert(blocking_pipelines[j] && "blocking_pipelines[j] is null");

            blocking_pipelines[j]->add_child_pipeline(streaming_pipelines[i]);
        }
    }

    // if there are no streaming pipelines, it is the beginning of a new
    // pipeline
    if (streaming_pipelines.size() == 0) {
        std::shared_ptr<Pipeline> pipeline = std::make_shared<Pipeline>();
        assert(operator_);
        assert(in_degree() > 0);
        pipeline->add_operator(operator_);
        // if this operator is blocking, then the new pipeline
        // has to wait for all the blocking pipelines to finish first
        for (unsigned i = 0u; i < blocking_pipelines.size(); ++i) {
            blocking_pipelines[i]->add_child_pipeline(pipeline);
        }

        root->add_pipeline(pipeline);

        // std::cout << "Finished create pipelines for in_degree = " <<
        // in_degree() << std::endl;
        return pipeline;
    } else {
        // std::cout << "Finished create pipelines for in_degree = " <<
        // in_degree() << std::endl;
        return streaming_pipelines.back();
    }
}

std::shared_ptr<MaximusContext> &QueryNode::get_context() {
    return ctx_;
}

}  // namespace maximus
