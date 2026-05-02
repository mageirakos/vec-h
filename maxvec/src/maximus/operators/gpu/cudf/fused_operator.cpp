#include <maximus/operators/gpu/cudf/distinct_operator.hpp>
#include <maximus/operators/gpu/cudf/engine.hpp>
#include <maximus/operators/gpu/cudf/filter_operator.hpp>
#include <maximus/operators/gpu/cudf/fused_operator.hpp>
#include <maximus/operators/gpu/cudf/group_by_operator.hpp>
#include <maximus/operators/gpu/cudf/hash_join_operator.hpp>
#include <maximus/operators/gpu/cudf/limit_operator.hpp>
#include <maximus/operators/gpu/cudf/order_by_operator.hpp>
#include <maximus/operators/gpu/cudf/project_operator.hpp>
#include <typeinfo>

namespace maximus::cudf {

FusedOperator::FusedOperator(std::shared_ptr<MaximusContext> &_ctx,
                             std::shared_ptr<Schema> _input_schema,
                             std::shared_ptr<FusedProperties> _properties)
        : AbstractFusedOperator(_ctx, std::move(_input_schema), std::move(_properties)) {
    gctx = _ctx->gcontext;
    ctx  = _ctx;

    assert(gctx != nullptr && "MaximusGContext must be initialized "
                              "before creating a GPU FusedOperator");

    assert(typeid(*gctx) == typeid(maximus::gpu::MaximusCudaContext) &&
           "MaximusGContext must be a MaximusCudaContext");

    // Initialize the operators
    std::shared_ptr<maximus::AbstractOperator> oper;
    std::vector<std::shared_ptr<Schema>> output_schemas = input_schemas;
    std::transform(
        properties->node_types.begin(),
        properties->node_types.end(),
        properties->properties.begin(),
        std::back_inserter(operators),
        [&](NodeType const &type, std::shared_ptr<NodeProperties> &props) {
            switch (type) {
                case NodeType::DISTINCT:
                    oper = std::make_shared<DistinctOperator>(
                        ctx,
                        std::make_shared<maximus::Schema>(*output_schemas[0]),
                        std::static_pointer_cast<DistinctProperties>(props));
                    output_schemas = {std::make_shared<Schema>(*(oper->output_schema))};
                    return std::move(std::dynamic_pointer_cast<maximus::gpu::GpuOperator>(oper));
                case NodeType::FILTER:
                    oper = std::make_shared<FilterOperator>(
                        ctx,
                        std::make_shared<maximus::Schema>(*output_schemas[0]),
                        std::static_pointer_cast<FilterProperties>(props));
                    output_schemas = {std::make_shared<Schema>(*(oper->output_schema))};
                    return std::move(std::dynamic_pointer_cast<maximus::gpu::GpuOperator>(oper));
                case NodeType::GROUP_BY:
                    oper = std::make_shared<DistinctOperator>(
                        ctx,
                        std::make_shared<maximus::Schema>(*output_schemas[0]),
                        std::static_pointer_cast<DistinctProperties>(props));
                    output_schemas = {std::make_shared<Schema>(*(oper->output_schema))};
                    return std::move(std::dynamic_pointer_cast<maximus::gpu::GpuOperator>(oper));
                case NodeType::HASH_JOIN:
                    oper = std::make_shared<HashJoinOperator>(
                        ctx,
                        std::make_shared<maximus::Schema>(*output_schemas[0]),
                        std::make_shared<maximus::Schema>(*output_schemas[1]),
                        std::static_pointer_cast<JoinProperties>(props));
                    output_schemas = {std::make_shared<Schema>(*(oper->output_schema))};
                    return std::move(std::dynamic_pointer_cast<maximus::gpu::GpuOperator>(oper));
                case NodeType::LIMIT:
                    oper = std::make_shared<LimitOperator>(
                        ctx,
                        std::make_shared<maximus::Schema>(*output_schemas[0]),
                        std::static_pointer_cast<LimitProperties>(props));
                    output_schemas = {std::make_shared<Schema>(*(oper->output_schema))};
                    return std::move(std::dynamic_pointer_cast<maximus::gpu::GpuOperator>(oper));
                case NodeType::ORDER_BY:
                    oper = std::make_shared<OrderByOperator>(
                        ctx,
                        std::make_shared<maximus::Schema>(*output_schemas[0]),
                        std::static_pointer_cast<OrderByProperties>(props));
                    output_schemas = {std::make_shared<Schema>(*(oper->output_schema))};
                    return std::move(std::dynamic_pointer_cast<maximus::gpu::GpuOperator>(oper));
                case NodeType::PROJECT:
                    oper = std::make_shared<ProjectOperator>(
                        ctx,
                        std::make_shared<maximus::Schema>(*output_schemas[0]),
                        std::static_pointer_cast<ProjectProperties>(props));
                    output_schemas = {std::make_shared<Schema>(*(oper->output_schema))};
                    return std::move(std::dynamic_pointer_cast<maximus::gpu::GpuOperator>(oper));
                default:
                    assert(false && "Unsupported node type in GPU Operator");
            }
        });

    // Initialize streaming flag and number of input ports
    int input_ports = 1;
    std::for_each(
        properties->node_types.begin(), properties->node_types.end(), [&](NodeType const &type) {
            streaming &=
                (type == NodeType::FILTER || type == NodeType::PROJECT || type == NodeType::LIMIT);
            input_ports += (int) (type == NodeType::HASH_JOIN);
        });

    input_tables.resize(input_ports);
    set_device_type(DeviceType::GPU);
    set_engine_type(EngineType::CUDF);

    operator_name = name();
}

void FusedOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);
    assert(device_input.on_gpu());
    auto input = device_input.as_gpu();

    assert(input);

    if (streaming) {
        // Stream data across pipeline
        gpu::GTableBatchPtr output = input;
        for (auto &op : operators) {
            op->on_add_native_input(output, port);
            output = op->export_next_native_batch();
            port   = 0;
        }

        output_tables.push_back(output);
    } else {
        // Collect all data before processing
        input_tables.push_back(input);
    }
}

void FusedOperator::on_no_more_input(int port) {
    if (streaming) {
        input_finished_flag = true;
    } else {
        // Combine all data before processing
        gpu::GTableBatchPtr output = input_tables[0];
        for (auto &op : operators) {
            op->on_add_native_input(output, port);
            output = op->export_next_native_batch();
            port   = 0;
        }

        output_tables.push_back(output);
    }
}

bool FusedOperator::has_more_batches_impl(bool blocking) {
    return !output_tables.empty() || (streaming && !input_finished_flag);
}

DeviceTablePtr FusedOperator::export_next_batch_impl() {
    assert(!output_tables.empty());
    gpu::GTableBatchPtr result = output_tables.front();
    output_tables.pop_front();
    return std::move(result);
}
}  // namespace maximus::cudf