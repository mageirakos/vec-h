#pragma once

#include <maximus/gpu/cuda_api.hpp>
#include <maximus/gpu/cudf/cudf_expr.hpp>
#include <maximus/operators/abstract_project_operator.hpp>
#include <maximus/operators/gpu/gpu_operator.hpp>

namespace maximus::cudf {

class ProjectOperator
        : public maximus::AbstractProjectOperator
        , public maximus::gpu::GpuOperator {
public:
    ProjectOperator(std::shared_ptr<MaximusContext>& ctx,
                    std::shared_ptr<Schema> input_schema,
                    std::shared_ptr<ProjectProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;

    void run_kernel(std::shared_ptr<MaximusContext>& ctx,
                    std::vector<CudfTablePtr>& input_tables,
                    std::vector<CudfTablePtr>& output_tables) override;

private:
    std::list<std::list<std::shared_ptr<::cudf::ast::expression>>> expression_list;
    std::list<std::list<std::shared_ptr<::cudf::scalar>>> scalar_list;
    std::list<std::list<maximus::gpu::ExtExpression>> ext_map;
    bool input_finished_flag = false;
    bool move_flag           = true;
};

}  // namespace maximus::cudf