#include <maximus/operators/native/scatter_operator.hpp>
#include <arrow/compute/api.h>
#include <arrow/table.h>
#include <unordered_map>
#include <algorithm>

namespace maximus::native {

/**
 * IMPORTANT: This is a BLOCKING operator.
 *
 * All input batches are collected in on_add_input() and processed together in on_no_more_input().
 * This ensures that:
 * 1. The same key always maps to the same partition (batch-insensitive)
 * 2. Partition assignment is deterministic and based on sorted unique keys across ALL data
 *
 * Performance note: Current implementation uses GetScalar() per row for key extraction,
 * which has overhead. For very large datasets, consider using typed array accessors instead.
 */

ScatterOperator::ScatterOperator(std::shared_ptr<MaximusContext>& ctx,
                                 std::shared_ptr<Schema> input_schema,
                                 std::shared_ptr<ScatterProperties> properties)
        : AbstractScatterOperator(ctx, input_schema, std::move(properties)) {
    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::NATIVE);

    // num_partitions will be inferred from unique keys if not explicitly set
    // For now, reserve space - will resize in on_no_more_input if needed
}

void ScatterOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);
    assert(port == 0 && "ScatterOperator only supports one input port");

    const auto& operator_name = name();
    profiler::close_regions({operator_name, "add_input"});
    device_input.convert_to<TableBatchPtr>(ctx_, input_schemas[port]);
    profiler::open_regions({operator_name, "add_input"});

    auto batch = device_input.as_table_batch();
    input_batches_.push_back(batch);
}

void ScatterOperator::on_no_more_input(int port) {
    if (input_batches_.empty()) {
        // Initialize empty outputs for the expected number of partitions
        int num_partitions = properties->num_partitions > 0 ? properties->num_partitions : 1;
        partition_outputs_.resize(num_partitions);
        current_batch_index_.resize(num_partitions, 0);
        return;
    }

    // Convert all RecordBatches to a single Table
    std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches;
    for (const auto& batch : input_batches_) {
        record_batches.push_back(batch->get_table_batch());
    }

    auto table_result = arrow::Table::FromRecordBatches(record_batches);
    if (!table_result.ok()) {
        throw std::runtime_error("Failed to create table from batches: " + table_result.status().ToString());
    }
    auto full_table = *table_result;

    if (full_table->num_rows() == 0) {
        int num_partitions = properties->num_partitions > 0 ? properties->num_partitions : 1;
        partition_outputs_.resize(num_partitions);
        current_batch_index_.resize(num_partitions, 0);
        return;
    }

    // Get the partition key column
    assert(!properties->partition_keys.empty());
    const auto& key_name = *properties->partition_keys[0].name();
    auto key_idx = full_table->schema()->GetFieldIndex(key_name);
    if (key_idx < 0) {
        throw std::runtime_error("Partition key column not found: " + key_name);
    }

    // Combine chunks for easier iteration
    auto combine_result = full_table->CombineChunks();
    if (!combine_result.ok()) {
        throw std::runtime_error("Failed to combine chunks: " + combine_result.status().ToString());
    }
    full_table = *combine_result;

    auto key_column = full_table->column(key_idx);
    auto key_array = key_column->chunk(0);
    int64_t num_rows = full_table->num_rows();

    // Step 1: Collect unique key values and sort them to create key -> partition_index mapping
    std::vector<int64_t> unique_keys;
    std::unordered_map<int64_t, int> key_to_partition;

    for (int64_t i = 0; i < num_rows; ++i) {
        auto scalar_result = key_array->GetScalar(i);
        if (!scalar_result.ok()) {
            throw std::runtime_error("Failed to get scalar: " + scalar_result.status().ToString());
        }

        int64_t key_value = 0;
        auto scalar = *scalar_result;
        if (scalar->type->id() == arrow::Type::INT64) {
            key_value = std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value;
        } else if (scalar->type->id() == arrow::Type::INT32) {
            key_value = std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value;
        } else {
            throw std::runtime_error("Partition key must be integer type");
        }

        if (key_to_partition.find(key_value) == key_to_partition.end()) {
            unique_keys.push_back(key_value);
            key_to_partition[key_value] = -1; // Will be set after sorting
        }
    }

    // Sort unique keys to create deterministic partition assignment
    std::sort(unique_keys.begin(), unique_keys.end());

    // Validate partition count: unique keys must not exceed num_partitions
    // Otherwise partition indices would be invalid
    if (properties->num_partitions > 0 &&
        static_cast<int>(unique_keys.size()) > properties->num_partitions) {
        throw std::runtime_error(
            "ScatterOperator: Number of unique keys (" +
            std::to_string(unique_keys.size()) +
            ") exceeds num_partitions (" +
            std::to_string(properties->num_partitions) +
            "). Increase num_partitions or reduce the number of unique partition key values.");
    }

    // Create the mapping: sorted key -> partition index
    for (size_t i = 0; i < unique_keys.size(); ++i) {
        key_to_partition[unique_keys[i]] = static_cast<int>(i);
    }

    int num_partitions = static_cast<int>(unique_keys.size());

    // Override with explicit num_partitions if set and larger
    if (properties->num_partitions > 0 && properties->num_partitions > num_partitions) {
        num_partitions = properties->num_partitions;
    }

    // Resize output structures
    partition_outputs_.resize(num_partitions);
    current_batch_index_.resize(num_partitions, 0);

    // Step 2: Build row indices for each partition using the key -> partition_index mapping
    std::vector<std::vector<int64_t>> partition_indices(num_partitions);

    for (int64_t i = 0; i < num_rows; ++i) {
        auto scalar_result = key_array->GetScalar(i);
        if (!scalar_result.ok()) {
            throw std::runtime_error("Failed to get scalar: " + scalar_result.status().ToString());
        }

        int64_t key_value = 0;
        auto scalar = *scalar_result;
        if (scalar->type->id() == arrow::Type::INT64) {
            key_value = std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value;
        } else if (scalar->type->id() == arrow::Type::INT32) {
            key_value = std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value;
        }

        int partition_id = key_to_partition[key_value];
        if (partition_id >= 0 && partition_id < num_partitions) {
            partition_indices[partition_id].push_back(i);
        }
    }

    // Step 3: Create output batches for each partition
    for (int p = 0; p < num_partitions; ++p) {
        if (partition_indices[p].empty()) {
            continue;  // Empty partition - no batch to add
        }

        // Build index array for this partition
        arrow::Int64Builder index_builder;
        auto status = index_builder.AppendValues(partition_indices[p]);
        if (!status.ok()) {
            throw std::runtime_error("Failed to build indices: " + status.ToString());
        }
        std::shared_ptr<arrow::Array> index_array;
        status = index_builder.Finish(&index_array);
        if (!status.ok()) {
            throw std::runtime_error("Failed to finish indices: " + status.ToString());
        }

        // Take rows for this partition
        auto take_result = arrow::compute::Take(full_table, index_array);
        if (!take_result.ok()) {
            throw std::runtime_error("Failed to take rows: " + take_result.status().ToString());
        }
        auto partition_table = take_result.ValueOrDie().table();

        // Combine to single batch
        auto combine_result2 = partition_table->CombineChunks();
        if (!combine_result2.ok()) {
            throw std::runtime_error("Failed to combine: " + combine_result2.status().ToString());
        }

        auto batch_reader = arrow::TableBatchReader(**combine_result2);
        auto batch_result = batch_reader.Next();
        if (batch_result.ok() && *batch_result != nullptr) {
            auto output_batch = std::make_shared<TableBatch>(ctx_, *batch_result);
            partition_outputs_[p].push_back(DeviceTablePtr(std::move(output_batch)));
        }
    }
}

bool ScatterOperator::has_more_batches_impl(bool blocking, int port) {
    assert(port >= 0 && port < properties->num_partitions);
    return current_batch_index_[port] < partition_outputs_[port].size();
}

DeviceTablePtr ScatterOperator::export_next_batch_impl(int port) {
    assert(has_more_batches_impl(true, port));
    assert(port >= 0 && port < properties->num_partitions);

    auto& batch = partition_outputs_[port][current_batch_index_[port]];
    ++current_batch_index_[port];
    return std::move(batch);
}

}  // namespace maximus::native
