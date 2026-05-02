#pragma once

#include <maximus/operators/abstract_project_operator.hpp>
#include <maximus/operators/acero/acero_operator.hpp>

namespace maximus::acero {
class ProjectOperator
        : public maximus::AbstractProjectOperator
        , public AceroOperator {
public:
    ProjectOperator(std::shared_ptr<MaximusContext>& ctx,
                    std::shared_ptr<Schema> input_schema,
                    std::shared_ptr<ProjectProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;
};
}  // namespace maximus::acero
