#pragma once

#include <maximus/operators/abstract_operator.hpp>
#include <maximus/operators/properties.hpp>

namespace maximus {
class RandomTableSourceOperator : public AbstractOperator {
public:
    RandomTableSourceOperator(std::shared_ptr<MaximusContext>& ctx,
                              std::shared_ptr<Schema> schema,
                              std::shared_ptr<RandomTableSourceProperties> properties);

    bool has_more_batches_impl(bool blocking) override;

    DeviceTablePtr export_next_batch_impl() override;

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

protected:
    Status generate_random_data(TableBatchPtr& output_table);

    std::shared_ptr<RandomTableSourceProperties> properties;

    std::size_t generated_rows_ = 0;
};

}  // namespace maximus
