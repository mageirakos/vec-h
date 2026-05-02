#include <maximus/operators/native/take_operator.hpp>
#include <arrow/compute/api.h>
#include <arrow/table.h>
#include <algorithm>
#include <numeric>

namespace maximus::native {

TakeOperator::TakeOperator(std::shared_ptr<MaximusContext>& ctx,
                           std::shared_ptr<Schema> data_schema,
                           std::shared_ptr<Schema> index_schema,
                           std::shared_ptr<TakeProperties> properties)
        : AbstractTakeOperator(ctx, std::move(data_schema), std::move(index_schema),
                               std::move(properties)) {
    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::NATIVE);
}

// ---------------------------------------------------------------------------
// Helper: read int key values from a typed Arrow array in zero-copy fashion.
// Returns a vector of int64_t keys.  Handles Int32 and Int64 column types.
// ---------------------------------------------------------------------------
static std::vector<int64_t> read_int_keys(const std::shared_ptr<arrow::Array>& array,
                                           int64_t num_rows) {
    std::vector<int64_t> keys(num_rows);
    if (array->type_id() == arrow::Type::INT64) {
        auto typed = std::static_pointer_cast<arrow::Int64Array>(array);
        for (int64_t i = 0; i < num_rows; ++i) {
            keys[i] = typed->Value(i);
        }
    } else if (array->type_id() == arrow::Type::INT32) {
        auto typed = std::static_pointer_cast<arrow::Int32Array>(array);
        for (int64_t i = 0; i < num_rows; ++i) {
            keys[i] = static_cast<int64_t>(typed->Value(i));
        }
    } else {
        throw std::runtime_error("TakeOperator: key column must be integer type, got: " +
                                  array->type()->ToString());
    }
    return keys;
}

void TakeOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);
    assert(port == 0 || port == 1);

    const auto& operator_name = name();

    if (port == 0) {
        // Data side (blocking): accumulate batches
        profiler::close_regions({operator_name, "add_input"});
        device_input.convert_to<TablePtr>(ctx_, input_schemas[port]);
        profiler::open_regions({operator_name, "add_input"});
        data_batches_.push_back(device_input.as_table()->get_table());
    } else {
        // Index side (streaming): look up keys and gather rows
        assert(port == 1);
        assert(map_built_ && "Data-side key map must be built before index-side input arrives");

        profiler::close_regions({operator_name, "add_input"});
        device_input.convert_to<TablePtr>(ctx_, input_schemas[port]);
        profiler::open_regions({operator_name, "add_input"});
        auto index_table = device_input.as_table()->get_table();

        const auto& index_key_name = properties->index_key;
        int index_key_idx = index_table->schema()->GetFieldIndex(index_key_name);
        if (index_key_idx < 0) {
            throw std::runtime_error("TakeOperator: index key column not found: " + index_key_name);
        }

        // Combine chunks for easier iteration
        auto combined_result = index_table->CombineChunks(ctx_->get_memory_pool());
        if (!combined_result.ok()) {
            throw std::runtime_error("TakeOperator: CombineChunks failed: " +
                                     combined_result.status().ToString());
        }
        index_table = *combined_result;

        auto index_key_array = index_table->column(index_key_idx)->chunk(0);
        int64_t num_index_rows = index_table->num_rows();

        // Read index keys using typed array access (zero-copy, no heap alloc per row)
        auto index_keys = read_int_keys(index_key_array, num_index_rows);

        // Build data-side row indices and index-side row indices
        // Pre-reserve capacity: in the common case most index keys match
        std::vector<int64_t> data_indices;
        std::vector<int64_t> index_indices;
        data_indices.reserve(num_index_rows);
        index_indices.reserve(num_index_rows);

        for (int64_t i = 0; i < num_index_rows; ++i) {
            auto it = key_to_rows_.find(index_keys[i]);
            if (it != key_to_rows_.end()) {
                for (int64_t data_row : it->second) {
                    data_indices.push_back(data_row);
                    index_indices.push_back(i);
                }
            }
            // If key not found in data, row is simply not emitted (inner-join semantics)
        }

        if (data_indices.empty()) {
            return; // No matches for this batch
        }

        // Sort by data-side row index for cache-friendly access on large columns
        // (e.g. large_list<float> embeddings).  This turns random gather into
        // a near-sequential scan, dramatically reducing cache misses.
        int64_t n = static_cast<int64_t>(data_indices.size());
        std::vector<int64_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
            return data_indices[a] < data_indices[b];
        });

        // Apply the permutation to both index vectors
        std::vector<int64_t> sorted_data_indices(n);
        std::vector<int64_t> sorted_index_indices(n);
        for (int64_t i = 0; i < n; ++i) {
            sorted_data_indices[i] = data_indices[order[i]];
            sorted_index_indices[i] = index_indices[order[i]];
        }

        // Build Arrow arrays from sorted vectors (zero-copy wrap)
        auto data_indices_array = std::make_shared<arrow::Int64Array>(
            n, arrow::Buffer::Wrap(sorted_data_indices));
        auto index_indices_array = std::make_shared<arrow::Int64Array>(
            n, arrow::Buffer::Wrap(sorted_index_indices));

        // Take from data table (now with sorted indices → sequential memory access)
        auto data_take_result = arrow::compute::Take(
            data_table_, data_indices_array,
            arrow::compute::TakeOptions::Defaults(),
            ctx_->get_exec_context());
        if (!data_take_result.ok()) {
            throw std::runtime_error("TakeOperator: Take on data table failed: " +
                                     data_take_result.status().ToString());
        }
        auto data_gathered = data_take_result->table();

        // Take from index table (non-key columns only)
        auto index_table_no_key = index_table->RemoveColumn(index_key_idx).ValueOrDie();
        
        if (index_table_no_key->num_columns() > 0) {
            auto index_take_result = arrow::compute::Take(
                index_table_no_key, index_indices_array,
                arrow::compute::TakeOptions::Defaults(),
                ctx_->get_exec_context());
            if (!index_take_result.ok()) {
                throw std::runtime_error("TakeOperator: Take on index table failed: " +
                                         index_take_result.status().ToString());
            }
            auto index_gathered = index_take_result->table();

            // Concatenate data columns + index non-key columns
            auto merged_fields = data_gathered->schema()->fields();
            auto index_fields = index_gathered->schema()->fields();
            merged_fields.insert(merged_fields.end(), index_fields.begin(), index_fields.end());

            auto merged_schema = arrow::schema(merged_fields);
            auto merged_columns = data_gathered->columns();
            auto index_columns = index_gathered->columns();
            merged_columns.insert(merged_columns.end(), index_columns.begin(), index_columns.end());

            auto merged_table = arrow::Table::Make(merged_schema, merged_columns);
            outputs_.push_back(DeviceTablePtr(std::make_shared<Table>(ctx_, merged_table)));
        } else {
            // Index table only had the key column, just output data side
            outputs_.push_back(DeviceTablePtr(std::make_shared<Table>(ctx_, data_gathered)));
        }
    }
}

void TakeOperator::on_no_more_input(int port) {
    if (port == 0) {
        // Build the materialized data table and the key → row index map
        if (data_batches_.empty()) {
            map_built_ = true;
            return;
        }

        if (data_batches_.size() == 1) {
            data_table_ = data_batches_[0];
        } else {
            auto concat_result = arrow::ConcatenateTables(
                data_batches_, arrow::ConcatenateTablesOptions::Defaults(),
                ctx_->get_memory_pool());
            if (!concat_result.ok()) {
                throw std::runtime_error("TakeOperator: ConcatenateTables failed: " +
                                         concat_result.status().ToString());
            }
            data_table_ = *concat_result;
        }

        // Combine chunks for uniform key access
        auto combined = data_table_->CombineChunks(ctx_->get_memory_pool());
        if (!combined.ok()) {
            throw std::runtime_error("TakeOperator: CombineChunks failed: " +
                                     combined.status().ToString());
        }
        data_table_ = *combined;

        // Build key → row indices map using typed array access (no GetScalar overhead)
        const auto& data_key_name = properties->data_key;
        int data_key_idx = data_table_->schema()->GetFieldIndex(data_key_name);
        if (data_key_idx < 0) {
            throw std::runtime_error("TakeOperator: data key column not found: " + data_key_name);
        }

        auto key_array = data_table_->column(data_key_idx)->chunk(0);
        int64_t num_rows = data_table_->num_rows();

        auto keys = read_int_keys(key_array, num_rows);

        key_to_rows_.reserve(num_rows);
        for (int64_t i = 0; i < num_rows; ++i) {
            key_to_rows_[keys[i]].push_back(i);
        }

        map_built_ = true;
        data_batches_.clear(); // Free memory
    }
}

bool TakeOperator::has_more_batches_impl(bool blocking) {
    return current_output_batch_ < outputs_.size();
}

DeviceTablePtr TakeOperator::export_next_batch_impl() {
    if (current_output_batch_ >= outputs_.size()) {
        return DeviceTablePtr();
    }
    return std::move(outputs_[current_output_batch_++]);
}

}  // namespace maximus::native
