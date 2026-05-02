#pragma once
#include <maximus/operators/abstract_operator.hpp>
#include <maximus/operators/properties.hpp>
#include <vector>

namespace maximus {

class AbstractOrderByOperator : public AbstractOperator {
public:
    AbstractOrderByOperator(std::shared_ptr<MaximusContext>& ctx,
                            std::shared_ptr<Schema> input_schema,
                            std::shared_ptr<OrderByProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override = 0;

    void on_no_more_input(int port) override = 0;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override = 0;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override = 0;

protected:
    std::shared_ptr<OrderByProperties> properties;
};

}  // namespace maximus
