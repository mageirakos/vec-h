#pragma once

#include <maximus/operators/abstract_group_by_operator.hpp>
#include <maximus/operators/acero/acero_operator.hpp>

namespace maximus::acero {

class GroupByOperator
        : public maximus::AbstractGroupByOperator
        , public AceroOperator {
public:
    GroupByOperator(std::shared_ptr<MaximusContext>& ctx,
                    std::shared_ptr<Schema> input_schema,
                    std::shared_ptr<GroupByProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;
};
}  // namespace maximus::acero
