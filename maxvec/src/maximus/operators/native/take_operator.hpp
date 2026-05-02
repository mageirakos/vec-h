#pragma once

#include <maximus/operators/abstract_take_operator.hpp>
#include <unordered_map>

namespace maximus::native {

/**
 * Native (CPU) Take operator.
 *
 * Builds a hash map of data_key → [row_indices] from the data table (port 0),
 * then for each row in the index table (port 1), gathers the matching data
 * rows using arrow::compute::Take.
 *
 * This bypasses Acero's hash join, which cannot handle list<> columns
 * in non-key fields.
 */
class TakeOperator : public AbstractTakeOperator {
public:
    TakeOperator(std::shared_ptr<MaximusContext>& ctx,
                 std::shared_ptr<Schema> data_schema,
                 std::shared_ptr<Schema> index_schema,
                 std::shared_ptr<TakeProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;
    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;
    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;

private:
    /// Accumulated data-side batches (port 0)
    std::vector<std::shared_ptr<arrow::Table>> data_batches_;

    /// Materialized data table after no_more_input on port 0
    std::shared_ptr<arrow::Table> data_table_;

    /// Key → row indices map built from data table
    std::unordered_map<int64_t, std::vector<int64_t>> key_to_rows_;

    /// Whether the key map has been built
    bool map_built_ = false;
};

}  // namespace maximus::native
