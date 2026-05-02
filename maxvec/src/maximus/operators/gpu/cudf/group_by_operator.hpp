#pragma once

#include <cudf/groupby.hpp>
#include <maximus/operators/abstract_group_by_operator.hpp>
#include <maximus/operators/gpu/gpu_operator.hpp>

namespace maximus::cudf {

class GroupByOperator
        : public maximus::AbstractGroupByOperator
        , public maximus::gpu::GpuOperator {
public:
    GroupByOperator(std::shared_ptr<MaximusContext>& ctx,
                    std::shared_ptr<Schema> input_schema,
                    std::shared_ptr<GroupByProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;

    void run_kernel(std::shared_ptr<MaximusContext>& ctx,
                    std::vector<CudfTablePtr>& input_tables,
                    std::vector<CudfTablePtr>& output_tables) override;

private:
    std::vector<int> key_indices;
    std::vector<std::pair<int, std::unique_ptr<::cudf::groupby_aggregation>>> aggregations;
    std::vector<std::pair<int, std::string>> agg_strings;
};

}  // namespace maximus::cudf