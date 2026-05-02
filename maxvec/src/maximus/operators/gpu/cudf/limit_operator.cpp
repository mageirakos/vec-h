#include <cudf/copying.hpp>
#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/gpu/cudf/limit_operator.hpp>
#include <typeinfo>

namespace maximus::cudf {

LimitOperator::LimitOperator(std::shared_ptr<MaximusContext> &_ctx,
                             std::shared_ptr<Schema> _input_schema,
                             std::shared_ptr<LimitProperties> _properties)
        : AbstractLimitOperator(_ctx, _input_schema, std::move(_properties))
        , GpuOperator(_ctx, {_input_schema}, get_id(), {}) {
    assert(ctx_);
    auto gctx = ctx_->get_gpu_context();

    assert(gctx != nullptr && "MaximusGContext must be initialized "
                              "before creating a GPU LimitOperator");

    assert(typeid(*gctx) == typeid(maximus::gpu::MaximusCudaContext) &&
           "MaximusGContext must be a MaximusCudaContext");

    // create the output schema
    output_schema = std::make_shared<Schema>(*input_schemas[0]);

    set_device_type(DeviceType::GPU);
    set_engine_type(EngineType::CUDF);

    operator_name = name();
}

void LimitOperator::on_add_input(DeviceTablePtr device_input, int port) {
    proxy_add_input(device_input, port);
}

void LimitOperator::run_kernel(std::shared_ptr<MaximusContext> &ctx,
                               std::vector<CudfTablePtr> &input_tables,
                               std::vector<CudfTablePtr> &output_tables) {
    if (finished_) return;

    // there is only one input port
    assert(input_tables.size() == 1);
    assert(input_tables[0]);
    auto &input = input_tables[0];

    auto cudf_view = input->view();

    assert(properties->offset >= 0);
    assert(properties->limit >= 0);

    assert(output_schema->get_schema() != nullptr);


    if (properties->limit == 0) return;

    auto batch_size = cudf_view.num_rows();

    if (offset + batch_size <= properties->offset) {
        offset += input->num_rows();
        return;
    }
    if (offset < properties->offset) {
        auto num_rows_to_ignore = properties->offset - offset;
        assert(num_rows_to_ignore < batch_size);
        input = std::make_shared<::cudf::table>(
            ::cudf::slice(cudf_view, {(int) num_rows_to_ignore, (int) batch_size})[0]);
        assert(input);
        cudf_view  = input->view();
        batch_size = cudf_view.num_rows();
        offset += num_rows_to_ignore;
    }

    if (num_rows + batch_size <= properties->limit) {
        output_tables.emplace_back(std::move(input));
        num_rows += batch_size;
    } else {
        auto num_rows_to_keep = properties->limit - num_rows;
        assert(num_rows_to_keep > 0);
        auto sliced = std::make_shared<::cudf::table>(
            ::cudf::slice(cudf_view, {0, (int) num_rows_to_keep})[0]);
        output_tables.emplace_back(std::move(sliced));
        num_rows += num_rows_to_keep;
    }

    if (num_rows == properties->limit) {
        finished_ = true;
    }
}

void LimitOperator::on_no_more_input(int port) {
    proxy_no_more_input(port);
}

bool LimitOperator::has_more_batches_impl(bool blocking) {
    return proxy_has_more_batches(blocking);
}

DeviceTablePtr LimitOperator::export_next_batch_impl() {
    return std::move(proxy_export_next_batch());
}

}  // namespace maximus::cudf
