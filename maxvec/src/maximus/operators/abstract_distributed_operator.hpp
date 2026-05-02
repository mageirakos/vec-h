#pragma once

#include <maximus/operators/abstract_operator.hpp>
#include <maximus/operators/properties.hpp>

namespace maximus {

class AbstractDistributedOperator : public AbstractOperator {
public:
    AbstractDistributedOperator() = default;
    AbstractDistributedOperator(const PhysicalOperatorType type,  // type of the operator
                                std::shared_ptr<MaximusContext>& ctx)
            : AbstractOperatortype(type, ctx) {}

    // There is always only 1 output schema, even if the operator has multiple downstream operators.
    AbstractDistributedOperator(const PhysicalOperatorType type,  // type of the operator
                                std::shared_ptr<MaximusContext>& ctx,
                                std::vector<std::shared_ptr<Schema>> input_schemas,  // input schema
                                std::shared_ptr<Schema> output_schema  // output schema
                                )
            : AbstractOperator(type, ctx, std::move(input_schemas), std::move(output_schema)) {}

    AbstractDistributedOperator(const PhysicalOperatorType type,  // type of the operator
                                std::shared_ptr<MaximusContext>& ctx,
                                std::vector<std::shared_ptr<Schema>> input_schemas  // input schema
                                )
            : AbstractOperator(type, ctx, std::move(input_schemas)) {}

    AbstractDistributedOperator(const PhysicalOperatorType type,  // type of the operator
                                std::shared_ptr<MaximusContext>& ctx,
                                std::shared_ptr<Schema> input_schema  // input schema
                                )
            : AbstractOperator(type, ctx, std::move(input_schema)) {}

    AbstractDistributedOperator(const PhysicalOperatorType type,  // type of the operator
                                std::shared_ptr<MaximusContext>& ctx,
                                std::shared_ptr<Schema> input_schema,  // input schema
                                std::shared_ptr<Schema> output_schema  // output schema
                                )
            : AbstractOperator(type, ctx, std::move(input_schema), std::move(output_schema)) {}

    void on_add_input(DeviceTablePtr device_input, int port) override {
        assert(device_input);
        assert(device_type);

        assert(device_input.on_device(device_type));

        input_batches.push_back(device_input);
    }

    virtual void run_kernel() = 0;

    void on_no_more_input(int port) override {
        assert(!cpu_input);
        assert(!gpu_input);

        auto table = merge_batches(ctx_, this->input_batches, output_schema);
        output_    = run_kernel(table);
    }

    [[nodiscard]] bool has_more_batches_impl(bool blocking) override { return output_; }

    [[nodiscard]] DeviceTablePtr export_next_batch_impl() override { return std::move(output_); }

protected:
    std::vector<DeviceTablePtr> input_batches;
};

}  // namespace maximus
