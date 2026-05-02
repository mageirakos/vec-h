#pragma once

#include <maximus/operators/abstract_filter_operator.hpp>
#include <maximus/operators/acero/acero_operator.hpp>

namespace maximus::acero {

class FilterOperator
        : public AbstractFilterOperator
        , public AceroOperator {
public:
    FilterOperator(std::shared_ptr<MaximusContext>& ctx,
                   std::shared_ptr<Schema> input_schema,
                   std::shared_ptr<FilterProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;
};
}  // namespace maximus::acero
