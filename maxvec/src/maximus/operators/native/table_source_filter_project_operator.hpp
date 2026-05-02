#pragma once

#include <arrow/csv/reader.h>

#include <maximus/operators/abstract_table_source_filter_project_operator.hpp>

namespace maximus::native {
class TableSourceFilterProjectOperator : public AbstractTableSourceFilterProjectOperator {
public:
    TableSourceFilterProjectOperator(
        std::shared_ptr<MaximusContext>& ctx,
        std::shared_ptr<TableSourceFilterProjectProperties> properties);

    bool has_more_batches_impl(bool blocking) override;

    DeviceTablePtr export_next_batch_impl() override;

protected:
    void read_next();

private:
    std::shared_ptr<arrow::RecordBatchReader> reader_;

    ArrowTableBatchPtr output_;
};
}  // namespace maximus::native
