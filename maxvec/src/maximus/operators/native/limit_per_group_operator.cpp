#include <maximus/operators/native/limit_per_group_operator.hpp>
#include <arrow/array.h>

namespace maximus::native {

LimitPerGroupOperator::LimitPerGroupOperator(std::shared_ptr<MaximusContext>& ctx,
                                             std::shared_ptr<Schema> input_schema,
                                             std::shared_ptr<LimitPerGroupProperties> properties)
        : AbstractLimitPerGroupOperator(ctx, input_schema, std::move(properties)) {
    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::NATIVE);
}
// This operator assumes input is sorted by the group key column. It emits contiguous slices of rows for each group, up to the specified limit per group.
// DO NOT USE if previous to the operator there is an operator that can change the order of rows (e.g. filter, hash-based aggregate, etc.) without a subsequent sort.
// OR if there is a data transfer GPU->CPU. It would need patching for such cases.
void LimitPerGroupOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);
    assert(port == 0 && "LimitPerGroupOperator only supports one input port");

    const auto& operator_name = name();
    profiler::close_regions({operator_name, "add_input"});
    device_input.convert_to<TableBatchPtr>(ctx_, input_schemas[port]);
    profiler::open_regions({operator_name, "add_input"});
    auto input = device_input.as_table_batch();

    int64_t num_rows = input->num_rows();
    if (num_rows == 0) return;

    // Resolve key column index on first call
    if (key_col_idx_ < 0) {
        auto schema = input->get_table_batch()->schema();
        key_col_idx_ = schema->GetFieldIndex(properties->group_key);
        if (key_col_idx_ < 0) {
            throw std::runtime_error(
                "LimitPerGroupOperator: group key column not found: " + properties->group_key);
        }
    }

    auto key_array = input->get_table_batch()->column(key_col_idx_);
    auto key_type = key_array->type_id();

    // Lambda to get key value at index, supporting int64 and int32
    auto get_key = [&](int64_t i) -> int64_t {
        if (key_type == arrow::Type::INT64) {
            return std::static_pointer_cast<arrow::Int64Array>(key_array)->Value(i);
        } else if (key_type == arrow::Type::INT32) {
            return static_cast<int64_t>(
                std::static_pointer_cast<arrow::Int32Array>(key_array)->Value(i));
        }
        throw std::runtime_error("LimitPerGroupOperator: unsupported key type");
    };

    int64_t limit_k = properties->limit_k;

    // Scan rows and emit slices for kept ranges
    int64_t range_start = -1;  // start of current kept range

    for (int64_t i = 0; i < num_rows; ++i) {
        int64_t key = get_key(i);

        // Detect group change
        if (!initialized_ || key != current_group_key_) {
            // Flush pending kept range before switching groups
            if (range_start >= 0) {
                auto sliced = input->slice(range_start, i - range_start);
                outputs_.push_back(DeviceTablePtr(std::move(sliced)));
                range_start = -1;
            }
            current_group_key_ = key;
            current_group_emitted_ = 0;
            initialized_ = true;
        }

        if (current_group_emitted_ < limit_k) {
            // This row is kept
            if (range_start < 0) {
                range_start = i;
            }
            current_group_emitted_++;
        } else {
            // This row is skipped (group already at limit)
            if (range_start >= 0) {
                auto sliced = input->slice(range_start, i - range_start);
                outputs_.push_back(DeviceTablePtr(std::move(sliced)));
                range_start = -1;
            }
        }
    }

    // Flush any remaining kept range
    if (range_start >= 0) {
        auto sliced = input->slice(range_start, num_rows - range_start);
        outputs_.push_back(DeviceTablePtr(std::move(sliced)));
    }
}

void LimitPerGroupOperator::on_no_more_input(int port) {
}

}  // namespace maximus::native
