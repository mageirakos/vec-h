#include <cudf/concatenate.hpp>
#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/gpu/cudf/order_by_operator.hpp>
#include <typeinfo>

namespace maximus::cudf {

arrow::Status get_sort_key_indices(const std::shared_ptr<arrow::Schema> &schema,
                                   const std::vector<SortKey> &keys,
                                   std::vector<int> *indices,
                                   std::vector<::cudf::order> *orders) {
    arrow::FieldVector fields = schema->fields();
    for (const auto &key : keys) {
        arrow::FieldRef key_ref                   = key.field;
        SortOrder order                           = key.order;
        std::vector<arrow::FieldPath> field_paths = key_ref.FindAll(fields);
        assert(field_paths.size() == 1 && "Field path ambiguous");
        ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Field> field, field_paths[0].Get(fields));
        int index = std::find(fields.begin(), fields.end(), field) - fields.begin();
        indices->push_back(index);
        orders->push_back(order == SortOrder::ASCENDING ? ::cudf::order::ASCENDING
                                                        : ::cudf::order::DESCENDING);
    }
    return arrow::Status::OK();
}

OrderByOperator::OrderByOperator(std::shared_ptr<MaximusContext> &_ctx,
                                 std::shared_ptr<Schema> _input_schema,
                                 std::shared_ptr<OrderByProperties> _properties)
        : AbstractOrderByOperator(_ctx, _input_schema, std::move(_properties))
        , GpuOperator(_ctx, {_input_schema}, get_id(), {0}) {
    assert(ctx_);
    auto gctx = ctx_->get_gpu_context();

    assert(gctx != nullptr && "MaximusGContext must be initialized "
                              "before creating a GPU OrderByOperator");

    assert(typeid(*gctx) == typeid(maximus::gpu::MaximusCudaContext) &&
           "MaximusGContext must be a MaximusCudaContext");

    // get the key indices
    key_indices.clear();
    std::shared_ptr<maximus::Schema> input_schema = input_schemas[0];
    assert(input_schema != nullptr && "Input schema is null");
    std::shared_ptr<arrow::Schema> arrow_schema = input_schema->get_schema();
    arrow::FieldVector fields                   = arrow_schema->fields();
    arrow::Status status =
        get_sort_key_indices(arrow_schema, properties->sort_keys, &key_indices, &key_orders);
    assert(status.ok() && "Failed to get key indices");

    // get the null orders
    // something interesting in cudf::sort I noticed, cudf takes a null_order
    // parameter for each column of key, while we just have one for the entire
    // key.
    null_orders.clear();
    null_orders.assign(key_indices.size(),
                       properties->null_order == NullOrder::FIRST ? ::cudf::null_order::BEFORE
                                                                  : ::cudf::null_order::AFTER);
    assert(null_orders.size() == key_indices.size() && "Null orders size mismatch");

    // create the output schema
    output_schema = input_schema;

    set_device_type(DeviceType::GPU);
    set_engine_type(EngineType::CUDF);

    operator_name = name();
}

void OrderByOperator::on_add_input(DeviceTablePtr device_input, int port) {
    proxy_add_input(device_input, port);
}

void OrderByOperator::on_no_more_input(int port) {
    proxy_no_more_input(port);
}

void OrderByOperator::run_kernel(std::shared_ptr<MaximusContext> &ctx,
                                 std::vector<CudfTablePtr> &input_tables,
                                 std::vector<CudfTablePtr> &output_tables) {
    // there is only one input port
    assert(input_tables.size() == 1);
    assert(input_tables[0]);

    auto &input = input_tables[0];

    std::shared_ptr<::cudf::table> combined_table = std::move(input);
    ::cudf::table_view complete_view              = combined_table->view();

    // perform sorting
    std::unique_ptr<::cudf::table> result = ::cudf::stable_sort_by_key(
        complete_view, complete_view.select(key_indices), key_orders, null_orders);

    // export the result to a GTable
    output_tables.emplace_back(std::move(result));
}

bool OrderByOperator::has_more_batches_impl(bool blocking) {
    return proxy_has_more_batches(blocking);
}

DeviceTablePtr OrderByOperator::export_next_batch_impl() {
    return std::move(proxy_export_next_batch());
}

}  // namespace maximus::cudf
