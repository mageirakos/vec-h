#pragma once

#include <maximus/operators/abstract_operator.hpp>
#include <maximus/operators/properties.hpp>

namespace maximus {

class AbstractVectorProjectDistanceOperator : public AbstractOperator {
public:
    AbstractVectorProjectDistanceOperator(
        std::shared_ptr<MaximusContext>& ctx,
        std::vector<std::shared_ptr<maximus::Schema>> input_schemas,
        std::shared_ptr<VectorProjectDistanceProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override = 0;
    void on_no_more_input(int port) override                   = 0;

protected:
    std::shared_ptr<VectorProjectDistanceProperties> properties;
};

}  // namespace maximus
