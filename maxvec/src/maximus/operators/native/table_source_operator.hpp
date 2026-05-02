#pragma once

#include <arrow/csv/reader.h>
#include <parquet/arrow/reader.h>

#include <maximus/operators/abstract_table_source_operator.hpp>

namespace maximus::native {
class TableSourceOperator : public AbstractTableSourceOperator {
public:
    TableSourceOperator(std::shared_ptr<MaximusContext>& ctx,
                        std::shared_ptr<TableSourceProperties> properties);

    bool has_more_batches_impl(bool blocking) override;

    DeviceTablePtr export_next_batch_impl() override;

protected:
    void read_next();

private:
    TablePtr full_table_;
    // if we are given a full table, we use a table_reader_;
    std::shared_ptr<arrow::TableBatchReader> table_reader_;

    ArrowTableBatchPtr output_;
};
}  // namespace maximus::native
