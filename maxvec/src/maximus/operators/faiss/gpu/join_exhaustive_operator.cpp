#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <maximus/operators/faiss/interop.hpp>

#include <maximus/operators/faiss/gpu/join_exhaustive_operator.hpp>
#include <maximus/operators/faiss/gpu/faiss_kernels.hpp>
#include <maximus/utils/cudf_helpers.hpp>

namespace maximus::faiss::gpu {

JoinExhaustiveOperator::JoinExhaustiveOperator(
    std::shared_ptr<MaximusContext> &_ctx,
    std::vector<std::shared_ptr<Schema>> _input_schemas,
    std::shared_ptr<VectorJoinExhaustiveProperties> _properties)
        : ::maximus::AbstractVectorJoinOperator(_ctx, _input_schemas, _properties)
        , ::maximus::gpu::GpuOperator(_ctx, _input_schemas, get_id(), {0, 1})
        , properties(_properties)
{
    set_device_type(DeviceType::GPU);
    set_engine_type(EngineType::FAISS);

    operator_name = name();

    assert(output_schema);

    // KNN search supported on GPUs, range search is not
    assert(properties->K.has_value());
    assert(!properties->radius.has_value());
    // filter bitmap is not supported on the GPU
    assert(!properties->filter_bitmap.has_value());
}

void JoinExhaustiveOperator::on_add_input(DeviceTablePtr device_input, int port) {
    proxy_add_input(device_input, port);
}

void JoinExhaustiveOperator::run_kernel(
    std::shared_ptr<MaximusContext>& ctx,
    std::vector<CudfTablePtr>& input_tables,
    std::vector<CudfTablePtr>& output_tables)
{
    assert(input_tables.size() == 2);
    assert(input_tables[0]);
    assert(input_tables[1]);

    auto const& props = properties;

    auto data_table  = input_tables[get_data_port()];
    auto query_table = input_tables[get_query_port()];

    auto stream = ctx->get_kernel_stream();
    auto& mr    = ctx->mr;

    // ---------------------------------------------------------------------
    // Extract vector columns
    // ---------------------------------------------------------------------
    auto data_schema  = input_schemas[get_data_port()]->get_schema();
    auto query_schema = input_schemas[get_query_port()]->get_schema();

    auto data_field  = props->data_vector_column.GetOne(*data_schema).ValueOrDie();
    auto query_field = props->query_vector_column.GetOne(*query_schema).ValueOrDie();

    int data_vec_idx  = data_schema->GetFieldIndex(data_field->name());
    int query_vec_idx = query_schema->GetFieldIndex(query_field->name());

    auto data_vec_col  = data_table->view().column(data_vec_idx);
    auto query_vec_col = query_table->view().column(query_vec_idx);

    assert(data_vec_col.type().id() == cudf::type_id::LIST);
    assert(query_vec_col.type().id() == cudf::type_id::LIST);

    int64_t D = embedding_dimension(data_vec_col);
    assert(D == embedding_dimension(query_vec_col));

    // ---------------------------------------------------------------------
    // Resolve K and metric
    // ---------------------------------------------------------------------
    assert(props->K.has_value());
    int64_t K = props->K.value();

    ::faiss::MetricType faiss_metric = to_faiss_metric(props->metric);

    // ---------------------------------------------------------------------
    // Run exhaustive KNN search
    // ---------------------------------------------------------------------
    PE("knn_search_gpu");
    GpuSearchResult knn_result = knn_search_gpu(
        data_vec_col,
        query_vec_col,
        D,
        K,
        faiss_metric,
        mr,
        stream);
    PL("knn_search_gpu");

    PE("build_result_gpu");
    auto output_table = join_after_knn_search_gpu(
        knn_result, data_table, query_table,
        data_schema, query_schema,
        output_schema->get_schema(),
        props->distance_column,
        mr,
        stream);
    PL("build_result_gpu");

    output_tables.emplace_back(std::move(output_table));
}

void JoinExhaustiveOperator::on_no_more_input(int port) {
    proxy_no_more_input(port);
}

bool JoinExhaustiveOperator::has_more_batches_impl(bool blocking) {
    return proxy_has_more_batches(blocking);
}

DeviceTablePtr JoinExhaustiveOperator::export_next_batch_impl() {
    return std::move(proxy_export_next_batch());
}

}  // namespace maximus::faiss
