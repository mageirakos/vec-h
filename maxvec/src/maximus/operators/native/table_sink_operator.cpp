#include <maximus/operators/native/table_sink_operator.hpp>

namespace maximus {
TableSinkOperator::TableSinkOperator(std::shared_ptr<MaximusContext>& ctx,
                                     std::shared_ptr<Schema> schema,
                                     std::shared_ptr<TableSinkProperties> properties)
        : AbstractOperator(PhysicalOperatorType::TABLE_SINK, ctx, schema, schema) {
    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::NATIVE);
}

void TableSinkOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);
    const auto& operator_name = name();
    profiler::close_regions({operator_name, "add_input"});
    device_input.convert_to<TableBatchPtr>(ctx_, input_schemas[port]);
    profiler::open_regions({operator_name, "add_input"});
    assert(port == 0);
    outputs_.push_back(std::move(device_input));
}

void TableSinkOperator::on_no_more_input(int port) {
}
}  // namespace maximus
