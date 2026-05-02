#pragma once

#include <cassert>
#include <maximus/context.hpp>
#include <maximus/operators/engine.hpp>
#include <maximus/types/device_table_ptr.hpp>
#include <maximus/types/node_type.hpp>
#include <maximus/types/physical_operator_type.hpp>
#include <maximus/types/schema.hpp>
#include <maximus/types/table.hpp>
#include <maximus/types/table_batch.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace maximus {

enum class PortType : uint8_t { STREAMING, BLOCKING };

class AbstractOperator {
public:
    AbstractOperator() = default;
    AbstractOperator(const PhysicalOperatorType type,  // type of the operator
                     std::shared_ptr<MaximusContext>& ctx)
            : type(type), ctx_(ctx) {
        assert(ctx_);
        static int id_op_counter_ = 0;
        id_                       = id_op_counter_++;
    }

    // There is always only 1 output schema, even if the operator has multiple downstream operators.
    AbstractOperator(const PhysicalOperatorType type,  // type of the operator
                     std::shared_ptr<MaximusContext>& ctx,
                     std::vector<std::shared_ptr<Schema>> input_schemas,  // input schema
                     std::shared_ptr<Schema> output_schema                // output schema
                     )
            : AbstractOperator(type, ctx) {
        assign_input_schemas(std::move(input_schemas));
        assign_output_schema(std::move(output_schema));
    }

    AbstractOperator(const PhysicalOperatorType type,  // type of the operator
                     std::shared_ptr<MaximusContext>& ctx,
                     std::vector<std::shared_ptr<Schema>> input_schemas  // input schema
                     )
            : AbstractOperator(type, ctx) {
        assign_input_schemas(std::move(input_schemas));
    }

    AbstractOperator(const PhysicalOperatorType type,  // type of the operator
                     std::shared_ptr<MaximusContext>& ctx,
                     std::shared_ptr<Schema> input_schema  // input schema
                     )
            : AbstractOperator(type, ctx) {
        std::vector<std::shared_ptr<Schema>> schemas;
        schemas.reserve(1);
        schemas.emplace_back(std::move(input_schema));
        assign_input_schemas(std::move(schemas));
    }

    AbstractOperator(const PhysicalOperatorType type,  // type of the operator
                     std::shared_ptr<MaximusContext>& ctx,
                     std::shared_ptr<Schema> input_schema,  // input schema
                     std::shared_ptr<Schema> output_schema  // output schema
                     )
            : AbstractOperator(type, ctx) {
        std::vector<std::shared_ptr<Schema>> schemas;
        schemas.reserve(1);
        schemas.emplace_back(std::move(input_schema));
        assign_input_schemas(std::move(schemas));
        assign_output_schema(std::move(output_schema));
    }

    virtual ~AbstractOperator() = default;

    // true if it can accept more input
    // it needs input if:
    // - there is more input
    // - still haven't finished computing
    [[nodiscard]] virtual bool needs_input(int port) const {
        return !is_finished() && !no_more_input_[port];
    }

    void add_input(DeviceTablePtr input, int port = 0) {
        const auto& operator_name = name();

        // entering profiling regions
        profiler::open_regions(
            {"OPERATORS", device_type_to_string(device_type), operator_name, "add_input"});

        assert(num_ports() > 0 && "add_input() called on an operator which has no input ports.");
        assert(input_schemas[0] && "Input schema must be set before add_input is called!");
        assert(port < num_ports() && "Port index out of bounds.");
        assert(output_schema && "Output schema must be set before add_input is called!");
        assert(device_type != DeviceType::UNDEFINED &&
               "Device type must be set before add_input is called!");
        assert(engine_type != EngineType::UNDEFINED &&
               "Engine type must be set before add_input is called!");

        if (!input || input.empty()) {
            // close the profiling regions before returning
            profiler::close_regions(
                {"OPERATORS", device_type_to_string(device_type), operator_name, "add_input"});
            return;
        }
        assert(needs_input(port) && "add_input() called on an operator which doesn't need input.");

        // if the batch does not reside on the same device as this operator,
        // copy it to the same device as the operator
        // PE("device_copy");
        // AbstractOperator::convert_table(ctx_, input, input_schemas[port], engine_type, device_type);
        // assert(input.on_device(device_type));
        // PL("device_copy");

        on_add_input(input, port);

        // leaving profiling regions
        profiler::close_regions(
            {"OPERATORS", device_type_to_string(device_type), operator_name, "add_input"});
    }

    virtual void on_add_input(DeviceTablePtr input, int port) = 0;

    // informs the operator that it will receive no further input
    void no_more_input(int port = 0) {
        // entering profiling regions
        const auto& operator_name = name();
        profiler::open_regions(
            {"OPERATORS", device_type_to_string(device_type), operator_name, "no_more_input"});

        assert(port < num_ports() && "Port index out of bounds.");
        assert(no_more_input_.size() == num_ports() && "no_more_input_.size() != num_ports().");
        assert(device_type != DeviceType::UNDEFINED &&
               "Device type must be set before no_more_input is called!");
        assert(engine_type != EngineType::UNDEFINED &&
               "Engine type must be set before add_input is called!");

        no_more_input_[port] = true;
        on_no_more_input(port);

        // leaving profiling regions
        profiler::close_regions(
            {"OPERATORS", device_type_to_string(device_type), operator_name, "no_more_input"});
    }

    virtual void on_no_more_input(int port) = 0;

    // returns true if completely finished processing
    // important for LIMIT, as it may finish before other operators
    // even before no_more_input() was called
    [[nodiscard]] bool is_finished() const {
        assert(device_type != DeviceType::UNDEFINED &&
               "Device type must be set before is_finished is called!");
        return finished_;
    }

    bool has_more_batches(bool blocking) {
        // entering profiling regions
        const auto& operator_name = name();
        if (operator_name == "CPU_NATIVE_TABLE_SINK") {
            profiler::open_regions({"Executor::execute"});
        }
        profiler::open_regions(
            {"OPERATORS", device_type_to_string(device_type), operator_name, "has_more_batches"});

        assert(device_type != DeviceType::UNDEFINED &&
               "Device type must be set before has_more_batches is called!");
        assert(engine_type != EngineType::UNDEFINED &&
               "Engine type must be set before add_input is called!");

        bool has_batches = this->has_more_batches_impl(blocking);

        profiler::close_regions(
            {"OPERATORS", device_type_to_string(device_type), operator_name, "has_more_batches"});
        if (operator_name == "CPU_NATIVE_TABLE_SINK") {
            profiler::close_regions({"Executor::execute"});
        }
        return has_batches;
    }

    virtual bool has_more_batches_impl(bool blocking) {
        // assert(!outputs_.empty() && "has_output_batches() called on an operator which has no output batches.");
        return current_output_batch_ < outputs_.size();
    }

    bool has_more_batches(bool blocking, int output_port) {
        // entering profiling regions
        const auto& operator_name = name();
        profiler::open_regions(
            {"OPERATORS", device_type_to_string(device_type), operator_name, "has_more_batches"});

        assert(device_type != DeviceType::UNDEFINED &&
               "Device type must be set before has_more_batches is called!");
        assert(engine_type != EngineType::UNDEFINED &&
               "Engine type must be set before add_input is called!");

        assert((type == PhysicalOperatorType::LOCAL_BROADCAST ||
                type == PhysicalOperatorType::SCATTER) &&
               "has_more_batches() with multiple output ports called on an operator which is not a "
               "local broadcast or scatter operator.");

        bool has_batches = this->has_more_batches_impl(blocking, output_port);

        profiler::close_regions(
            {"OPERATORS", device_type_to_string(device_type), operator_name, "has_more_batches"});

        return has_batches;
    }

    virtual bool has_more_batches_impl(bool blocking, int output_port) {
        throw std::runtime_error("has_more_batches_impl() for multiple output ports is only "
                                 "supported by the special LocalBroadcastOperator.");
    }

    DeviceTablePtr export_next_batch() {
        const auto& operator_name = name();

        assert(device_type != DeviceType::UNDEFINED &&
               "Device type must be set before export_next_batch is called!");
        assert(has_more_batches(false) &&
               "export_next_batch() called on an operator which has no output "
               "batches.");

        if (operator_name == "CPU_NATIVE_TABLE_SINK") {
            profiler::open_regions({"Executor::execute"});
        }
        profiler::open_regions(
            {"OPERATORS", device_type_to_string(device_type), operator_name, "export_next_batch"});

        auto batch = this->export_next_batch_impl();

        profiler::close_regions(
            {"OPERATORS", device_type_to_string(device_type), operator_name, "export_next_batch"});

        if (operator_name == "CPU_NATIVE_TABLE_SINK") {
            profiler::close_regions({"Executor::execute"});
        }
        return std::move(batch);
    }

    virtual DeviceTablePtr export_next_batch_impl() {
        return std::move(outputs_[current_output_batch_++]);
    }

    DeviceTablePtr export_next_batch(int output_port) {
        const auto& operator_name = name();

        assert(device_type != DeviceType::UNDEFINED &&
               "Device type must be set before export_next_batch is called!");
        assert(has_more_batches(false, output_port) &&
               "export_next_batch() called on an operator which has no output "
               "batches.");

        assert((type == PhysicalOperatorType::LOCAL_BROADCAST ||
                type == PhysicalOperatorType::SCATTER) &&
               "export_next_batch() with multiple output ports called on an operator which is not "
               "a local broadcast or scatter operator.");

        profiler::open_regions(
            {"OPERATORS", device_type_to_string(device_type), operator_name, "export_next_batch"});

        auto batch = this->export_next_batch_impl(output_port);

        profiler::close_regions(
            {"OPERATORS", device_type_to_string(device_type), operator_name, "export_next_batch"});

        return batch;
    }

    virtual DeviceTablePtr export_next_batch_impl(int output_port) {
        throw std::runtime_error("export_next_batch_impl() for multiple output ports is only "
                                 "supported by the special LocalBroadcastOperator.");
    }

    // returns all the batches produced by the operator in a single table
    // the operator's data is moved to the table and the returned table is owned by the caller
    TablePtr export_table() {
        const auto& operator_name = name();
        assert(current_output_batch_ == 0);

        assert(device_type == DeviceType::CPU &&
               "export_table() called on an operator which is not on the CPU.");
        // assert(!outputs_.empty() && "export_table() called on an operator which has no output batches."); assert(has_more_batches() && "export_table() called on an operator which has no output batches.");
        if (operator_name == "CPU_NATIVE_TABLE_SINK") {
            profiler::open_regions({"Executor::execute"});
        }
        profiler::open_regions(
            {"OPERATORS", device_type_to_string(device_type), operator_name, "export_table"});

        TablePtr table_ptr;
        DeviceTablePtr table(std::move(table_ptr));

        profiler::close_regions(
            {"OPERATORS", device_type_to_string(device_type), operator_name, "export_table"});
        if (operator_name == "CPU_NATIVE_TABLE_SINK") {
            profiler::close_regions({"Executor::execute"});
        }

        std::vector<DeviceTablePtr> pending_outputs;

        while (has_more_batches(true)) {
            pending_outputs.emplace_back(export_next_batch());
        }

        if (!pending_outputs.empty()) {
            if (operator_name == "CPU_NATIVE_TABLE_SINK") {
                profiler::open_regions({"Executor::execute"});
            }
            profiler::open_regions(
                {"OPERATORS", device_type_to_string(device_type), operator_name, "export_table"});

            table = merge_batches(ctx_, pending_outputs, output_schema);
            assert(table);

            profiler::close_regions(
                {"OPERATORS", device_type_to_string(device_type), operator_name, "export_table"});
            if (operator_name == "CPU_NATIVE_TABLE_SINK") {
                profiler::close_regions({"Executor::execute"});
            }
        }

        table.convert_to<TablePtr>(ctx_, output_schema);

        assert(table.is_table());

        return std::move(table.as_table());
    }

    virtual std::string to_string_extra() { return ""; }

    std::string to_string(int indent = 0) {
        std::ostringstream oss;
        std::string spaces(indent, ' ');

        oss << "AbstractOperator(\n"
            << spaces << "    type                = " << physical_operator_to_string(type);

        auto extra = to_string_extra();
        if (!extra.empty()) {
            oss << " (" << extra << ")\n";
        }

        for (std::size_t i = 0; i < input_schemas.size(); ++i) {
            oss << spaces << "    input_schemas[" << i << "]    = "
                << (input_schemas[i] ? input_schemas[i]->to_string(indent + 26) : "null") << "\n";
        }

        oss << spaces << "    output_schema       = "
            << (output_schema ? output_schema->to_string(indent + 26) : "null") << "\n"
            << spaces << ")";

        return oss.str();
    }

    [[nodiscard]] std::string name() const {
        std::ostringstream oss;
        assert(engine_type != EngineType::UNDEFINED &&
               "Engine type must be set before name() is called.");
        assert(device_type != DeviceType::UNDEFINED &&
               "Device type must be set before name() is called.");
        assert(type != PhysicalOperatorType::UNDEFINED &&
               "Physical operator type must be set before name() is called.");

        oss << device_type_to_string(device_type) << "_" << engine_type_to_string(engine_type)
            << "_" << physical_operator_to_string(type);
        return oss.str();
    }

    [[nodiscard]] std::size_t num_ports() const {
        assert(input_schemas.size() == no_more_input_.size() &&
               "num_ports() != no_more_input_.size().");
        return input_schemas.size();
    }

    void assign_input_schemas(std::vector<std::shared_ptr<Schema>> schemas) {
        assert(input_schemas.empty() && "assign_schema() called on an operator "
                                        "which already has input schemas.");
        assert(!schemas.empty() && "assign_schema() called with empty schemas.");
        input_schemas = std::move(schemas);
        assert(!input_schemas.empty() && "assign_schema() called with empty schemas.");
        assert(input_schemas[0] && "assign_schema() called with empty schemas.");

        auto num_ports = input_schemas.size();

        no_more_input_ = std::vector<bool>(num_ports, false);
        port_types     = std::vector<PortType>(num_ports, PortType::STREAMING);
    }

    void assign_output_schema(std::shared_ptr<Schema> output) {
        assert(output);
        output_schema = std::move(output);
    }

    void set_blocking_port(int port) {
        assert(port < port_types.size() && "port_types out of bounds");
        port_types[port] = PortType::BLOCKING;
    }

    void set_streaming_port(int port) {
        assert(port < port_types.size() && "port_types out of bounds");
        port_types[port] = PortType::STREAMING;
    }

    [[nodiscard]] virtual bool is_sink() const { return false; }

    [[nodiscard]] int get_id() const { return id_; }

    std::shared_ptr<MaximusContext>& get_context() { return ctx_; }

    void set_device_type(DeviceType device_type) {
        assert(device_type != DeviceType::UNDEFINED && "Device type must be set to a valid value.");
        this->device_type = device_type;
    }

    void set_engine_type(EngineType engine_type) {
        assert(engine_type != EngineType::UNDEFINED && "Engine type must be set to a valid value.");
        this->engine_type = engine_type;
    }

public:
    // the type of the operator
    PhysicalOperatorType type = PhysicalOperatorType::UNDEFINED;

    // the schema of the input ports
    std::vector<std::shared_ptr<Schema>> input_schemas;

    // the schema of the output ports
    std::shared_ptr<Schema> output_schema;

    // whether a port is streaming or blocking
    std::vector<PortType> port_types;

    // device type
    DeviceType device_type = DeviceType::UNDEFINED;

    // engine type
    EngineType engine_type = EngineType::UNDEFINED;

    EngineType next_engine_type       = EngineType::UNDEFINED;
    PhysicalOperatorType next_op_type = PhysicalOperatorType::UNDEFINED;

protected:
    std::shared_ptr<MaximusContext> ctx_;

    std::vector<DeviceTablePtr> outputs_;
    std::size_t current_output_batch_ = 0;

    bool finished_ = false;
    std::vector<bool> no_more_input_;

    int id_ = -1;
};
}  // namespace maximus
