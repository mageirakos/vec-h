#include <iostream>
#include <maximus/operators/native/limit_operator.hpp>

namespace maximus::native {

LimitOperator::LimitOperator(std::shared_ptr<MaximusContext>& ctx,
                             std::shared_ptr<Schema> input_schema,
                             std::shared_ptr<LimitProperties> properties)
        : AbstractLimitOperator(ctx, input_schema, std::move(properties)) {
    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::NATIVE);
}

void LimitOperator::on_add_input(DeviceTablePtr device_input, int port) {
    if (finished_) return;

    assert(device_input);

    const auto& operator_name = name();
    profiler::close_regions({operator_name, "add_input"});
    device_input.convert_to<TableBatchPtr>(ctx_, input_schemas[port]);
    profiler::open_regions({operator_name, "add_input"});
    auto input = device_input.as_table_batch();

    assert(properties->offset >= 0);
    assert(properties->limit >= 0);

    assert(port == 0 && "LimitOperator only supports one input port");
    assert(!is_finished() && "LimitOperator is already finished");
    assert(num_rows < properties->limit && "LimitOperator has already reached the limit");

    auto batch_size = input->num_rows();

    if (properties->limit == 0) return;

    if (offset + batch_size <= properties->offset) {
        offset += input->num_rows();
        return;
    }
    if (offset < properties->offset) {
        auto num_rows_to_ignore = properties->offset - offset;
        assert(num_rows_to_ignore < input->num_rows());
        input = input->slice(num_rows_to_ignore, input->num_rows());
        offset += num_rows_to_ignore;
        batch_size = input->num_rows();
    }

    if (num_rows + batch_size <= properties->limit) {
        outputs_.push_back(DeviceTablePtr(std::move(input)));
        num_rows += batch_size;
    } else {
        auto num_rows_to_keep = properties->limit - num_rows;
        assert(num_rows_to_keep > 0);
        auto sliced = input->slice(0, num_rows_to_keep);
        outputs_.push_back(DeviceTablePtr(std::move(sliced)));
        num_rows += num_rows_to_keep;
    }
    if (num_rows == properties->limit) {
        finished_ = true;
    }
}

void LimitOperator::on_no_more_input(int port) {
}

}  // namespace maximus::native
