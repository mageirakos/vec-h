#pragma once

#include <cudf/concatenate.hpp>
#include <maximus/gpu/cudf/cudf_types.hpp>
#include <maximus/gpu/gpu_api.hpp>
#include <maximus/operators/abstract_operator.hpp>
#include <maximus/utils/arrow_helpers.hpp>
#include <memory>

namespace maximus {
namespace gpu {

class GpuOperator {
public:
    GpuOperator() = default;

    GpuOperator(std::shared_ptr<MaximusContext>& ctx,
                std::vector<std::shared_ptr<Schema>> input_schemas,
                int operator_id                 = -1,
                std::vector<int> blocking_ports = {})
            : _ctx(ctx), _input_schemas(std::move(input_schemas)), _operator_id(operator_id) {
        int num_ports = _input_schemas.size();
        _device_input_batches.resize(num_ports);
        _host_input_batches.resize(num_ports);
        _input_tables.resize(num_ports);
        _port_types    = std::vector<PortType>(num_ports, PortType::STREAMING);
        _no_more_input = std::vector<bool>(num_ports, false);

        for (const int& port : blocking_ports) {
            set_blocking_port(port);
        }
        _all_ports_are_blocking = blocking_ports.size() == num_ports;
    }

    CudfTablePtr merge_batches(int port, std::shared_ptr<MaximusContext>& ctx) {
        profiler::open_regions({"cudf::concatenate"});
        auto num_ports = _input_schemas.size();
        assert(port < num_ports);
        assert(port >= 0);

        CudfTablePtr table;

        if (_device_input_batches[port].empty()) {
            profiler::close_regions({"cudf::concatenate"});
            return table;
        }

        // this is where the concatenated batches are stored
        // has to be empty before merging
        assert(!_input_tables[port]);

        // if there is only one batch, we can just use it
        // here we avoid invoking the concatenate function
        // since it will always copy the data
        if (_device_input_batches[port].size() == 1) {
            table = std::move(_device_input_batches[port][0]);
        } else {
            // otherwise, we have to copy the data, since cudf doesnt support in-place concatenation
            std::vector<::cudf::table_view> table_views;
            table_views.reserve(_device_input_batches[port].size());
            for (const auto& batch : _device_input_batches[port]) {
                table_views.push_back(batch->view());
            }

            table = std::move(
                ::cudf::concatenate(::cudf::host_span<::cudf::table_view const>(table_views)));
        }

        // since the batches have been merged into a CudfTablePtr, we can clear the input batches
        _device_input_batches[port].clear();

        profiler::close_regions({"cudf::concatenate"});
        return std::move(table);
    }

    void proxy_add_input(DeviceTablePtr device_input, int port) {
        assert(_ctx);
        assert(port >= 0 && "Invalid port");
        assert(port < _input_schemas.size() && "Invalid port");
        assert(device_input);

        assert(operator_name != "");
        if (device_input.on_gpu()) {
            // convert to cudf table
            // here we don't want the profiling regions of convert_to to be nested inside the add_input regions
            profiler::close_regions({operator_name, "add_input"});
            device_input.convert_to<CudfTablePtr>(_ctx, _input_schemas[port]);

            profiler::open_regions({operator_name, "add_input"});
            assert(device_input.is_cudf_table());

            auto input = device_input.as_cudf_table();

            _device_input_batches[port].push_back(std::move(input));
        }

        if (device_input.on_cpu()) {
            should_sync = true;
            profiler::close_regions({operator_name, "add_input"});
            device_input.convert_to<ArrowTableBatchPtr>(_ctx, _input_schemas[port]);
            profiler::open_regions({operator_name, "add_input"});
            assert(device_input.is_arrow_table_batch());

            auto input = device_input.as_arrow_table_batch();

            _host_input_batches[port].push_back(std::move(input));
        }

        if (is_streaming_port(port)) {
            if (_host_input_batches[port].size() > 0) {
                assert(_host_input_batches[port].size() == 1);
                auto device_batch = DeviceTablePtr(std::move(_host_input_batches[port][0]));
                profiler::close_regions({operator_name, "add_input"});
                device_batch.convert_to<CudfTablePtr>(_ctx, _input_schemas[port]);
                // ensure the cpu table is alive while the cpu->gpu async copy is happening
                profiler::open_regions({operator_name, "add_input"});
                _device_input_batches[port].push_back(std::move(device_batch.as_cudf_table()));
                _host_input_batches[port].clear();
            }
            // since this is a streaming port, we run the kernel immediately
            assert(_device_input_batches[port].size() == 1);

            // all blocking inputs must be received before any streaming input
            // this is guaranteed by the Executor
            assert(received_all_blocking_inputs());
            assert(!_input_tables[port]);
            _input_tables[port] = merge_batches(port, _ctx);

            run_kernel_full(_ctx, _input_tables, _output_tables, "add_input");

            _input_tables[port] = nullptr;
        }
    }

    // takes input_tables and fills in output_tables
    void run_kernel_full(std::shared_ptr<MaximusContext>& ctx,
                         std::vector<CudfTablePtr>& input_tables,
                         std::vector<CudfTablePtr>& output_tables,
                         const std::string& region) {
        // if there is at least one empty input, check if the operator can still produce output
        bool has_null_input = false;
        for (int i = 0; i < input_tables.size(); i++) {
            if (!input_tables[i]) {
                has_null_input = true;
                break;
            }
        }
        if (has_null_input && !handle_empty_inputs(input_tables)) {
            return;
        }

        // make sure the tables from the cpu have been copied
        if (should_sync) {
            profiler::close_regions({operator_name, region});
            profiler::open_regions({"DataTransformation", "CPU_GPU_SYNC"});
            ctx->h2d_stream.synchronize();
            profiler::close_regions({"DataTransformation", "CPU_GPU_SYNC"});
            profiler::open_regions({operator_name, region});
        }
        run_kernel(ctx, input_tables, output_tables);
        clear();
    }

    // takes input_tables and fills in output_tables
    virtual void run_kernel(std::shared_ptr<MaximusContext>& ctx,
                            std::vector<CudfTablePtr>& input_tables,
                            std::vector<CudfTablePtr>& output_tables) = 0;

    // Override in subclasses that can still produce output with empty inputs
    // (e.g., outer joins must emit the non-empty side with NULL-filled columns).
    virtual bool handle_empty_inputs(std::vector<CudfTablePtr>& input_tables) {
        return false;
    }

    void proxy_no_more_input(int port) {
        assert(_ctx);
        assert(port < _no_more_input.size());
        assert(port >= 0);
        _no_more_input[port] = true;

        if (is_blocking_port(port)) {
            if (_host_input_batches[port].size() > 0) {
                auto pool = _ctx->get_pinned_memory_pool();
                assert(pool);
                profiler::open_regions({"arrow::concatenate"});
                auto concatenated_batch =
                    to_record_batch(_host_input_batches[port], _ctx->get_pinned_memory_pool());
                profiler::close_regions({"arrow::concatenate"});
                assert(concatenated_batch);
                DeviceTablePtr device_batch = DeviceTablePtr(std::move(concatenated_batch));
                profiler::close_regions({operator_name, "no_more_input"});
                device_batch.convert_to<CudfTablePtr>(_ctx, _input_schemas[port]);
                profiler::open_regions({operator_name, "no_more_input"});
                _device_input_batches[port].push_back(std::move(device_batch.as_cudf_table()));
                _host_input_batches[port].clear();
            }

            // if this is a blocking port, the merging has to be done in no_more_input
            assert(!_input_tables[port]);
            _input_tables[port] = merge_batches(port, _ctx);
            // this is where the concatenated batches are stored
            // it can be empty if the input didn't produce any output
            // assert(_input_tables[port]);
        }

        // if all inputs are received, run the kernel, only if all the ports are blocking
        // if there is at least one streaming port, the kernel will be run with each streaming batch
        if (received_all_inputs()) {
            if (_all_ports_are_blocking) {
                run_kernel_full(_ctx, _input_tables, _output_tables, "no_more_input");
            }
        }
    }

    void clear() {
        // we clear only the inputs, since the outputs are moved when exported
        _device_input_batches = std::vector<std::vector<CudfTablePtr>>(_input_schemas.size());
        _host_input_batches   = std::vector<std::vector<ArrowTableBatchPtr>>(_input_schemas.size());
        _input_tables         = std::vector<CudfTablePtr>(_input_schemas.size());
    }

    [[nodiscard]] bool proxy_has_more_batches(bool blocking) {
        return _output_tables_exported < _output_tables.size();
    }

    [[nodiscard]] DeviceTablePtr proxy_export_next_batch() {
        assert(!_output_tables.empty());
        auto& result = _output_tables[_output_tables_exported];
        ++_output_tables_exported;
        // output_tables.pop_front();
        return DeviceTablePtr(std::move(result));
    }

    bool is_streaming_port(int port) {
        assert(port < _port_types.size());
        assert(_port_types[port] == PortType::STREAMING || _port_types[port] == PortType::BLOCKING);
        return _port_types[port] == PortType::STREAMING;
    }

    bool is_blocking_port(int port) {
        assert(port < _port_types.size());
        assert(_port_types[port] == PortType::STREAMING || _port_types[port] == PortType::BLOCKING);
        return _port_types[port] == PortType::BLOCKING;
    }

    void set_blocking_port(int port) {
        assert(port < _port_types.size());
        _port_types[port] = PortType::BLOCKING;
    }

    bool received_all_blocking_inputs() {
        bool received = true;
        for (int i = 0; i < _port_types.size(); i++) {
            if (is_blocking_port(i) && !_no_more_input[i]) {
                received = false;
                break;
            }
        }
        return received;
    }

    bool received_all_inputs() {
        bool received = true;
        for (int i = 0; i < _port_types.size(); i++) {
            if (!_no_more_input[i]) {
                received = false;
                break;
            }
        }
        return received;
    }

protected:
    std::shared_ptr<MaximusContext> _ctx = nullptr;

    std::vector<std::shared_ptr<Schema>> _input_schemas;

    // the input batches are merged into input_tables for pipeline breakers
    std::vector<std::vector<CudfTablePtr>> _device_input_batches;
    std::vector<std::vector<ArrowTableBatchPtr>> _host_input_batches;
    std::vector<CudfTablePtr> _input_tables;
    std::vector<CudfTablePtr> _output_tables;

    std::vector<PortType> _port_types;
    std::vector<bool> _no_more_input;

    // if at least one input is from cpu, we have to sync
    bool should_sync = false;

    int _output_tables_exported = 0;

    int _operator_id = -1;

    bool _all_ports_are_blocking = false;

    std::string operator_name = "";
};

}  // namespace gpu
}  // namespace maximus
