#pragma once

#include <maximus/operators/abstract_order_by_operator.hpp>
#include <maximus/operators/acero/acero_operator.hpp>

namespace maximus::acero {

class OrderByOperator
        : public maximus::AbstractOrderByOperator
        , public AceroOperator {
public:
    OrderByOperator(std::shared_ptr<MaximusContext>& ctx,
                    std::shared_ptr<Schema> input_schema,
                    std::shared_ptr<OrderByProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;
};
}  // namespace maximus::acero
