#include <cudf/stream_compaction.hpp>
#include <maximus/operators/gpu/cudf/distinct_operator.hpp>
#include <maximus/operators/gpu/gpu_operator.hpp>
#include <typeinfo>

namespace maximus::cudf {

arrow::Status get_distinct_key_indices(const std::shared_ptr<arrow::Schema> &schema,
                                       const std::vector<arrow::FieldRef> &keys,
                                       std::vector<int> *indices) {
    arrow::FieldVector fields = schema->fields();
    for (const auto &key : keys) {
        std::vector<arrow::FieldPath> field_paths = key.FindAll(fields);
        assert(field_paths.size() == 1 && "Field path ambiguous");
        ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Field> field, field_paths[0].Get(fields));
        int index = std::find(fields.begin(), fields.end(), field) - fields.begin();
        indices->push_back(index);
    }
    return arrow::Status::OK();
}

DistinctOperator::DistinctOperator(std::shared_ptr<MaximusContext> &_ctx,
                                   std::shared_ptr<Schema> _input_schema,
                                   std::shared_ptr<DistinctProperties> _properties)
        : AbstractDistinctOperator(_ctx, _input_schema, std::move(_properties))
        , GpuOperator(_ctx, {_input_schema}, get_id(), {0}) {
    assert(ctx_);
    auto gctx = ctx_->get_gpu_context();

    assert(gctx != nullptr && "MaximusGContext must be initialized "
                              "before creating a GPU GroupByOperator");

    assert(typeid(*gctx) == typeid(maximus::gpu::MaximusCudaContext) &&
           "MaximusGContext must be a MaximusCudaContext");

    // get the key indices
    key_indices.clear();
    std::shared_ptr<Schema> input_schema = AbstractOperator::input_schemas[0];
    assert(input_schema != nullptr && "Input schema is null");
    std::shared_ptr<arrow::Schema> arrow_schema = input_schema->get_schema();
    arrow::FieldVector fields                   = arrow_schema->fields();
    arrow::Status status =
        get_distinct_key_indices(arrow_schema, properties->distinct_keys, &key_indices);
    assert(status.ok() && "Failed to get key indices");

    // create the output schema
    output_schema = std::make_shared<Schema>(*input_schemas[0]);

    set_device_type(DeviceType::GPU);
    set_engine_type(EngineType::CUDF);

    operator_name = name();
}

void DistinctOperator::on_add_input(DeviceTablePtr device_input, int port) {
    proxy_add_input(device_input, port);
}

void DistinctOperator::run_kernel(std::shared_ptr<MaximusContext> &ctx,
                                  std::vector<CudfTablePtr> &input_tables,
                                  std::vector<CudfTablePtr> &output_tables) {
    // there is only one input port
    assert(input_tables.size() == 1);
    assert(input_tables[0]);
    auto &input                           = input_tables[0];
    std::unique_ptr<::cudf::table> result = ::cudf::stable_distinct(input->view(), key_indices);
    output_tables.emplace_back(std::move(result));
}

void DistinctOperator::on_no_more_input(int port) {
    proxy_no_more_input(port);
}

bool DistinctOperator::has_more_batches_impl(bool blocking) {
    return proxy_has_more_batches(blocking);
}

DeviceTablePtr DistinctOperator::export_next_batch_impl() {
    return std::move(proxy_export_next_batch());
}

}  // namespace maximus::cudf
