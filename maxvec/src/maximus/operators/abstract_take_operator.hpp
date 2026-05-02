#pragma once

#include <maximus/operators/abstract_operator.hpp>
#include <maximus/operators/properties.hpp>

namespace maximus {

/**
 * Abstract Take (Index-Gather) Operator.
 *
 * Two-input operator that performs a key-based row gather, bypassing
 * Acero's hash join (which cannot handle list<> columns in non-key fields).
 *
 * Port 0 (blocking/build): The DATA table — arbitrary schema including list<float>.
 * Port 1 (streaming/probe): The INDEX table — contains the lookup key column.
 *
 * Semantics: For each row in the index table, find all matching rows in
 * the data table by key, and emit the data-side columns (gathered via
 * arrow::compute::Take). The output includes all data-side columns
 * plus all non-key columns from the index side.
 *
 * NOTE: CPU only. On GPU, cuDF's hash join handles list<> types natively,
 *       so this operator is not needed.
 */
class AbstractTakeOperator : public AbstractOperator {
public:
    AbstractTakeOperator(std::shared_ptr<MaximusContext>& ctx,
                         std::shared_ptr<Schema> data_schema,
                         std::shared_ptr<Schema> index_schema,
                         std::shared_ptr<TakeProperties> properties);

    void on_no_more_input(int port) override = 0;

    void on_add_input(DeviceTablePtr input, int port) override = 0;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override = 0;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override = 0;

protected:
    std::shared_ptr<TakeProperties> properties;
};

}  // namespace maximus
