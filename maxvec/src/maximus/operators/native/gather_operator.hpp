#pragma once

#include <maximus/operators/abstract_gather_operator.hpp>

namespace maximus::native {

class GatherOperator : public AbstractGatherOperator {
public:
    GatherOperator(std::shared_ptr<MaximusContext>& ctx,
                   std::shared_ptr<Schema> input_schema,
                   std::shared_ptr<GatherProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;
    void on_no_more_input(int port) override;

private:
    std::vector<TableBatchPtr> input_batches_;
    std::vector<bool> port_completed_;
    int completed_ports_ = 0;
};

}  // namespace maximus::native
