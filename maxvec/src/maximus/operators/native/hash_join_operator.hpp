#pragma once

#include <maximus/operators/abstract_hash_join_operator.hpp>

namespace maximus::native {

class HashJoinOperator : public maximus::AbstractHashJoinOperator {
public:
    HashJoinOperator(std::shared_ptr<MaximusContext> &ctx,
                     std::shared_ptr<Schema> left_schema,
                     std::shared_ptr<Schema> right_schema,
                     std::shared_ptr<JoinProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] int get_build_port() const override;

    [[nodiscard]] int get_probe_port() const override;

protected:
    // the accumulated batches from the right side
    std::vector<TableBatchPtr> accumulated_batches;
};
}  // namespace maximus::native
// maximus
