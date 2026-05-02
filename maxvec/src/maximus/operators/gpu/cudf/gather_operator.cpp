#include <cudf/concatenate.hpp>
#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/gpu/cudf/gather_operator.hpp>
#include <typeinfo>

namespace maximus::cudf {

GatherOperator::GatherOperator(
    std::shared_ptr<MaximusContext>& _ctx,
    std::shared_ptr<Schema> _input_schema,
    std::shared_ptr<GatherProperties> _properties)
        : AbstractGatherOperator(_ctx, _input_schema, std::move(_properties))
        , GpuOperator(_ctx,
            // Create input schemas for all ports (all same schema)
            std::vector<std::shared_ptr<Schema>>(this->properties->num_inputs, _input_schema),
            get_id(),
            // All ports are blocking - we need all inputs before concatenating
            [this]() {
                std::vector<int> blocking;
                for (int i = 0; i < this->properties->num_inputs; ++i) {
                    blocking.push_back(i);
                }
                return blocking;
            }())
{
    assert(ctx_);
    auto gctx = ctx_->get_gpu_context();

    assert(gctx != nullptr && "MaximusGContext must be initialized "
                              "before creating a GPU GatherOperator");

    assert(typeid(*gctx) == typeid(maximus::gpu::MaximusCudaContext) &&
           "MaximusGContext must be a MaximusCudaContext");

    // Set output schema (same as input)
    output_schema = _input_schema;

    num_inputs_ = properties->num_inputs;
    port_finished_.assign(num_inputs_, false);
    port_batches_.resize(num_inputs_);

    set_device_type(DeviceType::GPU);
    set_engine_type(EngineType::CUDF);

    operator_name = name();
}

void GatherOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(port >= 0 && port < num_inputs_);

    // Convert to cudf table and store
    const auto& op_name = name();
    profiler::close_regions({op_name, "add_input"});
    device_input.convert_to<CudfTablePtr>(ctx_, input_schemas[port]);
    profiler::open_regions({op_name, "add_input"});

    if (device_input.is_cudf_table()) {
        port_batches_[port].push_back(device_input.as_cudf_table());
    }
}

void GatherOperator::run_kernel(std::shared_ptr<MaximusContext>& ctx,
                                std::vector<CudfTablePtr>& input_tables,
                                std::vector<CudfTablePtr>& output_tables) {
    // This is called after all inputs are received
    // Collect all non-null tables from all ports
    std::vector<::cudf::table_view> table_views;

    for (int port = 0; port < num_inputs_; ++port) {
        for (auto& table : port_batches_[port]) {
            if (table && table->num_rows() > 0) {
                table_views.push_back(table->view());
            }
        }
    }

    if (table_views.empty()) {
        concatenation_done_ = true;
        return;
    }

    if (table_views.size() == 1) {
        // Only one table, no need to concatenate
        // Find and move the single table
        for (int port = 0; port < num_inputs_; ++port) {
            for (auto& table : port_batches_[port]) {
                if (table && table->num_rows() > 0) {
                    concatenated_result_ = std::move(table);
                    break;
                }
            }
            if (concatenated_result_) break;
        }
    } else {
        // Concatenate all tables
        concatenated_result_ = ::cudf::concatenate(
            ::cudf::host_span<::cudf::table_view const>(table_views));
    }

    concatenation_done_ = true;

    // Push to AbstractOperator's outputs_ for has_more_batches/export_next_batch
    if (concatenated_result_) {
        outputs_.push_back(DeviceTablePtr(std::move(concatenated_result_)));
    }

    // Clear port batches to free memory
    for (auto& batches : port_batches_) {
        batches.clear();
    }
}

void GatherOperator::on_no_more_input(int port) {
    assert(port >= 0 && port < num_inputs_);
    port_finished_[port] = true;

    // Check if all ports are finished
    bool all_finished = true;
    for (bool finished : port_finished_) {
        if (!finished) {
            all_finished = false;
            break;
        }
    }

    if (all_finished && !concatenation_done_) {
        // Run concatenation
        std::vector<CudfTablePtr> empty_inputs;
        std::vector<CudfTablePtr> empty_outputs;
        run_kernel(ctx_, empty_inputs, empty_outputs);
    }
}

}  // namespace maximus::cudf
