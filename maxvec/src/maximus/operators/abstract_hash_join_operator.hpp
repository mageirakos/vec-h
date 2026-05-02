#pragma once

#include <maximus/operators/abstract_operator.hpp>
#include <maximus/operators/properties.hpp>

namespace maximus {
class AbstractHashJoinOperator : public AbstractOperator {
public:
    AbstractHashJoinOperator(std::shared_ptr<MaximusContext>& ctx,
                             std::shared_ptr<Schema> left_schema,
                             std::shared_ptr<Schema> right_schema,
                             std::shared_ptr<JoinProperties> properties);

    void on_no_more_input(int port) override = 0;

    void on_add_input(DeviceTablePtr input, int port) override = 0;

    [[nodiscard]] virtual int get_build_port() const;

    [[nodiscard]] virtual int get_probe_port() const;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override = 0;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override = 0;

protected:
    std::shared_ptr<JoinProperties> properties;
};
}  // namespace maximus
