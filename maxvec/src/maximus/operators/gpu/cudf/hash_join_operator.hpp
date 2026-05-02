#pragma once

#include <cudf/join/join.hpp>
#include <maximus/operators/abstract_hash_join_operator.hpp>
#include <maximus/operators/gpu/gpu_operator.hpp>

namespace maximus::cudf {

class HashJoinOperator
        : public maximus::AbstractHashJoinOperator
        , public maximus::gpu::GpuOperator {
public:
    HashJoinOperator(std::shared_ptr<MaximusContext>& ctx,
                     std::shared_ptr<Schema> left_schema,
                     std::shared_ptr<Schema> right_schema,
                     std::shared_ptr<JoinProperties> properties);

    void on_add_input(DeviceTablePtr input, int port) override;

    void on_no_more_input(int port) override;

    [[nodiscard]] virtual int get_build_port() const;

    [[nodiscard]] virtual int get_probe_port() const;

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override;

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override;

    void run_kernel(std::shared_ptr<MaximusContext>& ctx,
                    std::vector<CudfTablePtr>& input_tables,
                    std::vector<CudfTablePtr>& output_tables) override;

    bool handle_empty_inputs(std::vector<CudfTablePtr>& input_tables) override;

private:
    std::vector<int> build_key_indices, probe_key_indices;
    std::string build_suffix, probe_suffix;
    JoinType join_type;
};

}  // namespace maximus::cudf