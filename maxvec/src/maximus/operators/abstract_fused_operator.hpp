#pragma once

#include <functional>
#include <maximus/operators/abstract_operator.hpp>
#include <maximus/operators/properties.hpp>

namespace maximus {

class AbstractFusedOperator : public AbstractOperator {
public:
    AbstractFusedOperator(std::shared_ptr<MaximusContext>& ctx,
                          std::shared_ptr<Schema> input_schema,
                          std::shared_ptr<FusedProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override = 0;

    void on_no_more_input(int port) override = 0;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override = 0;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override = 0;

    [[nodiscard]] std::string to_string_extra() override;

protected:
    std::shared_ptr<FusedProperties> properties;
    std::vector<PhysicalOperatorType> physical_types;
};

}  // namespace maximus
