#pragma once

#include <maximus/operators/abstract_operator.hpp>
#include <maximus/operators/properties.hpp>

namespace maximus {
class TableSinkOperator : public AbstractOperator {
public:
    TableSinkOperator(std::shared_ptr<MaximusContext>& ctx,
                      std::shared_ptr<Schema> schema,
                      std::shared_ptr<TableSinkProperties> properties);

    /*
    DeviceTablePtr export_next_batch_impl() override {
        throw std::runtime_error("TableSinkOperator::export_next_batch() should never be called on "
                                 "a TableSinkOperator");
    }
    */

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    bool is_sink() const override { return true; }

protected:
    std::shared_ptr<TableSinkProperties> properties;
};

}  // namespace maximus
