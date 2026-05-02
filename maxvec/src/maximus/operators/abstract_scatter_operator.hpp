#pragma once

#include <maximus/operators/abstract_operator.hpp>
#include <maximus/operators/properties.hpp>

namespace maximus {

class AbstractScatterOperator : public AbstractOperator {
public:
    AbstractScatterOperator(std::shared_ptr<MaximusContext>& ctx,
                            std::shared_ptr<Schema> input_schema,
                            std::shared_ptr<ScatterProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override = 0;

    void on_no_more_input(int port) override = 0;

    // Multi-port output support (like LocalBroadcast)
    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;
    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;

    [[nodiscard]] virtual bool has_more_batches_impl(bool blocking, int port) override = 0;
    [[nodiscard]] virtual DeviceTablePtr export_next_batch_impl(int port) override = 0;

protected:
    std::shared_ptr<ScatterProperties> properties;
};

}  // namespace maximus
