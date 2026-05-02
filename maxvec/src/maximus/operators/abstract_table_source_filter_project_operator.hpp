#pragma once

#include <maximus/operators/abstract_operator.hpp>
#include <maximus/operators/properties.hpp>

namespace maximus {
class AbstractTableSourceFilterProjectOperator : public AbstractOperator {
public:
    AbstractTableSourceFilterProjectOperator(
        std::shared_ptr<MaximusContext>& ctx,
        std::shared_ptr<TableSourceFilterProjectProperties> properties);

    bool has_more_batches_impl(bool blocking) override = 0;

    DeviceTablePtr export_next_batch_impl() override = 0;

    bool needs_input(int port) const override { return false; }

    void on_no_more_input(int port) override {
        throw std::runtime_error(
            "TableSourceFilterProjectOperator does not support on_no_more_input");
    }

    void on_add_input(DeviceTablePtr input, int port) override {
        throw std::runtime_error("TableSourceFilterProjectOperator does not support add_input");
    }

protected:
    std::shared_ptr<TableSourceFilterProjectProperties> properties;
};

}  // namespace maximus
