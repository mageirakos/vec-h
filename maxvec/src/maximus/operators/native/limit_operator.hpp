#pragma once

#include <maximus/operators/abstract_limit_operator.hpp>
#include <maximus/operators/properties.hpp>
#include <maximus/types/expression.hpp>
#include <vector>

namespace maximus::native {

class LimitOperator : public AbstractLimitOperator {
public:
    LimitOperator(std::shared_ptr<MaximusContext>& ctx,
                  std::shared_ptr<Schema> input_schema,
                  std::shared_ptr<LimitProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

protected:
    int64_t num_rows = 0;
    int64_t offset   = 0;
};

}  // namespace maximus::native
