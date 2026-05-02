#pragma once

#include <maximus/operators/abstract_operator.hpp>
#include <maximus/operators/properties.hpp>

namespace maximus {

class AbstractLimitOperator : public AbstractOperator {
public:
    AbstractLimitOperator(std::shared_ptr<MaximusContext>& ctx,
                          std::shared_ptr<Schema> input_schema,
                          std::shared_ptr<LimitProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override = 0;

    void on_no_more_input(int port) override = 0;

protected:
    std::shared_ptr<LimitProperties> properties;
};

}  // namespace maximus
