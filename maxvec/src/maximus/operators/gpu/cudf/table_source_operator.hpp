#pragma once

#include <maximus/gpu/cuda_api.hpp>
#include <maximus/gpu/cudf/cudf_expr.hpp>
#include <maximus/operators/abstract_table_source_operator.hpp>
#include <maximus/operators/gpu/gpu_operator.hpp>

namespace maximus::cudf {

class TableSourceOperator : public maximus::AbstractTableSourceOperator {
public:
    TableSourceOperator(std::shared_ptr<MaximusContext> &ctx,
                        std::shared_ptr<TableSourceProperties> properties);

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;

private:
    GTablePtr table;
};

}  // namespace maximus::cudf
