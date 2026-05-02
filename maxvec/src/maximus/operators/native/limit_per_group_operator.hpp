#pragma once

#include <maximus/operators/abstract_limit_per_group_operator.hpp>
#include <maximus/operators/properties.hpp>

namespace maximus::native {

class LimitPerGroupOperator : public AbstractLimitPerGroupOperator {
public:
    LimitPerGroupOperator(std::shared_ptr<MaximusContext>& ctx,
                          std::shared_ptr<Schema> input_schema,
                          std::shared_ptr<LimitPerGroupProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

private:
    int64_t current_group_key_ = 0;
    int64_t current_group_emitted_ = 0;
    bool initialized_ = false;
    int key_col_idx_ = -1;
};

}  // namespace maximus::native
