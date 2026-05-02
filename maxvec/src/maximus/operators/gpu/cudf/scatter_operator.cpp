#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/concatenate.hpp>
#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/gpu/cudf/scatter_operator.hpp>
#include <typeinfo>

namespace maximus::cudf {

/**
 * BLOCKING operator.
 *
 * All input batches are collected and merged before partitioning occurs.
 * This ensures that:
 * 1. The same key always maps to the same partition (batch-insensitive)
 * 2. Partition assignment is deterministic and based on sorted unique keys
 *
 * Port 0 is marked as blocking in the constructor, which means GpuOperator::proxy_no_more_input()
 * will wait for all inputs before calling run_kernel().
 */

ScatterOperator::ScatterOperator(
    std::shared_ptr<MaximusContext>& _ctx,
    std::shared_ptr<Schema> _input_schema,
    std::shared_ptr<ScatterProperties> _properties)
        : AbstractScatterOperator(_ctx, _input_schema, std::move(_properties))
        , GpuOperator(_ctx, {_input_schema}, get_id(), {0})  // Port 0 is blocking
{
    assert(ctx_);
    auto gctx = ctx_->get_gpu_context();

    assert(gctx != nullptr && "MaximusGContext must be initialized "
                              "before creating a GPU ScatterOperator");

    assert(typeid(*gctx) == typeid(maximus::gpu::MaximusCudaContext) &&
           "MaximusGContext must be a MaximusCudaContext");

    // Set output schema (same as input)
    output_schema = input_schemas[0];

    // Initialize partition tracking
    num_partitions_ = properties->num_partitions;
    port_progress_.assign(num_partitions_, 0);
    partition_tables_.resize(num_partitions_);

    set_device_type(DeviceType::GPU);
    set_engine_type(EngineType::CUDF);

    operator_name = name();
}

void ScatterOperator::on_add_input(DeviceTablePtr device_input, int port) {
    proxy_add_input(device_input, port);
}

void ScatterOperator::run_kernel(std::shared_ptr<MaximusContext>& ctx,
                                 std::vector<CudfTablePtr>& input_tables,
                                 std::vector<CudfTablePtr>& output_tables) {
    // There is only one input port
    assert(input_tables.size() == 1);
    if (!input_tables[0]) return;

    auto& input = input_tables[0];
    auto input_view = input->view();

    if (input_view.num_rows() == 0) {
        return;
    }

    // Get the partition key column index
    assert(!properties->partition_keys.empty());
    const auto& key_name = *properties->partition_keys[0].name();

    // Find key column index in schema
    int key_idx = -1;
    auto schema = input_schemas[0]->get_schema();
    for (int i = 0; i < schema->num_fields(); ++i) {
        if (schema->field(i)->name() == key_name) {
            key_idx = i;
            break;
        }
    }
    if (key_idx < 0) {
        throw std::runtime_error("Partition key column not found: " + key_name);
    }

    auto key_col = input_view.column(key_idx);

    // Step 1: Get unique sorted keys to create key -> partition_index mapping
    // Create a table view with just the key column for distinct/sort operations
    ::cudf::table_view key_table_view{{key_col}};

    // Get distinct values
    auto unique_keys_table = ::cudf::distinct(
        key_table_view,
        {0},  // key columns
        ::cudf::duplicate_keep_option::KEEP_ANY
    );

    // Sort the unique keys
    auto sorted_unique_keys = ::cudf::sort(
        unique_keys_table->view(),
        {::cudf::order::ASCENDING},
        {::cudf::null_order::AFTER}
    );

    // Validate partition count: unique keys must not exceed num_partitions
    // Otherwise cudf::partition will produce out-of-bounds indices
    int64_t num_unique_keys = sorted_unique_keys->num_rows();
    if (num_unique_keys > num_partitions_) {
        throw std::runtime_error(
            "ScatterOperator: Number of unique keys (" +
            std::to_string(num_unique_keys) +
            ") exceeds num_partitions (" +
            std::to_string(num_partitions_) +
            "). Increase num_partitions or reduce the number of unique partition key values.");
    }

    // Step 2: Build partition_map using lower_bound
    // For each row's key, find its position in sorted unique keys (that's the partition index)
    auto partition_map = ::cudf::lower_bound(
        sorted_unique_keys->view(),      // haystack: sorted unique keys
        key_table_view,                   // needles: all key values from input
        {::cudf::order::ASCENDING},
        {::cudf::null_order::AFTER}
    );

    // Step 3: Call cudf::partition to reorder table and get offsets
    auto [partitioned_table, offsets] = ::cudf::partition(
        input_view,
        partition_map->view(),
        num_partitions_
    );

    // Step 4: Split into per-partition tables using offsets
    // offsets has num_partitions + 1 elements: [0, end_of_p0, end_of_p1, ..., num_rows]
    // For cudf::split, we need the interior split points (excluding first 0 and last num_rows)
    if (offsets.size() > 2) {
        std::vector<::cudf::size_type> split_indices(offsets.begin() + 1, offsets.end() - 1);
        auto splits = ::cudf::split(partitioned_table->view(), split_indices);

        // Store each partition as an owned table.
        // NOTE on copying: cudf::split returns table_views (references into partitioned_table).
        // We must create owned cudf::table instances because:
        // 1. Each partition needs independent lifetime for downstream operators
        // 2. partitioned_table goes out of scope after run_kernel returns
        // This copy is intentional and necessary for correctness.
        for (int i = 0; i < num_partitions_ && i < static_cast<int>(splits.size()); ++i) {
            if (splits[i].num_rows() > 0) {
                partition_tables_[i] = std::make_shared<::cudf::table>(splits[i]);
            }
        }
    } else if (offsets.size() == 2) {
        // Only one partition
        if (partitioned_table->num_rows() > 0) {
            partition_tables_[0] = std::move(partitioned_table);
        }
    }

    partitioning_done_ = true;
}

void ScatterOperator::on_no_more_input(int port) {
    proxy_no_more_input(port);
}

bool ScatterOperator::has_more_batches_impl(bool blocking, int port) {
    assert(port >= 0 && port < num_partitions_);
    // Each port produces at most one batch
    return port_progress_[port] == 0 && partition_tables_[port] != nullptr;
}

DeviceTablePtr ScatterOperator::export_next_batch_impl(int port) {
    assert(port >= 0 && port < num_partitions_);
    assert(has_more_batches_impl(true, port));

    port_progress_[port]++;

    // Move out the partition table for this port
    return DeviceTablePtr(std::move(partition_tables_[port]));
}

}  // namespace maximus::cudf
