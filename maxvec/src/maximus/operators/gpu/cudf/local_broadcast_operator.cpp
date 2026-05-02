#include <cudf/copying.hpp>
#include <cudf/transform.hpp>
#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/gpu/cudf/local_broadcast_operator.hpp>
#include <typeinfo>

namespace maximus::cudf {

LocalBroadcastOperator::LocalBroadcastOperator(
    std::shared_ptr<MaximusContext> &_ctx,
    std::shared_ptr<Schema> _input_schema,
    std::shared_ptr<LocalBroadcastProperties> _properties)
        : AbstractLocalBroadcastOperator(_ctx, _input_schema, std::move(_properties))
        , GpuOperator(_ctx, {_input_schema}, get_id()) {
    assert(ctx_);
    auto gctx = ctx_->get_gpu_context();

    assert(gctx != nullptr && "MaximusGContext must be initialized "
                              "before creating a GPU LocalBroadcastOperator");

    assert(typeid(*gctx) == typeid(maximus::gpu::MaximusCudaContext) &&
           "MaximusGContext must be a MaximusCudaContext");

    // set output schema
    output_schema = input_schemas[0];

    // set number of input and output ports
    num_broadcast_ports = properties->num_output_ports;
    port_progress.assign(num_broadcast_ports, 0);
    replicate_flag = properties->should_replicate;

    set_device_type(DeviceType::GPU);
    set_engine_type(EngineType::CUDF);

    operator_name = name();
}

void LocalBroadcastOperator::on_add_input(DeviceTablePtr device_input, int port) {
    proxy_add_input(device_input, port);
}

void LocalBroadcastOperator::run_kernel(std::shared_ptr<MaximusContext> &ctx,
                                        std::vector<CudfTablePtr> &input_tables,
                                        std::vector<CudfTablePtr> &output_tables) {
    // there is only one input port
    assert(input_tables.size() == 1);
    assert(input_tables[0]);

    auto &input = input_tables[0];

    broadcasted_ports.push_back(num_broadcast_ports);
    output_tables.emplace_back(std::move(input));
    transferred_output_tables.emplace_back(output_tables.back());
    ++num_inputs;
}

void LocalBroadcastOperator::on_no_more_input(int port) {
    ctx_->h2d_stream.synchronize();
}

bool LocalBroadcastOperator::has_more_batches_impl(bool blocking, int port) {
    return (port_progress[port] < num_inputs);
}

DeviceTablePtr LocalBroadcastOperator::export_next_batch_impl(int port) {
    assert(!_output_tables.empty());
    int output_table = port_progress[port];
    if (replicate_flag) {
        broadcasted_ports[output_table]--;
        port_progress[port]++;

        CudfTablePtr replicated_table;

        if (!broadcasted_ports[output_table]) {
            replicated_table = std::move(transferred_output_tables[output_table].as_cudf_table());
        } else {
            replicated_table = std::make_shared<::cudf::table>(
                *transferred_output_tables[output_table].as_cudf_table());
        }

        assert(replicated_table);
        return DeviceTablePtr(replicated_table);
    }
    if (!transferred_output_tables[output_table].is_gtable()) {
        transferred_output_tables[output_table].convert_to<GTablePtr>(ctx_, output_schema);
    }
    GTablePtr result = transferred_output_tables[output_table].as_gtable();
    broadcasted_ports[output_table]--;
    port_progress[port]++;
    if (!broadcasted_ports[output_table]) return DeviceTablePtr(std::move(result));
    return DeviceTablePtr(result);
}

}  // namespace maximus::cudf
