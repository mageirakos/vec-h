#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>

#include <maximus/operators/faiss/gpu/join_indexed_operator.hpp>
#include <maximus/operators/faiss/gpu/faiss_kernels.hpp>
#include <maximus/utils/cudf_helpers.hpp>

namespace maximus::faiss::gpu {

JoinIndexedOperator::JoinIndexedOperator(
    std::shared_ptr<MaximusContext> &_ctx,
    std::vector<std::shared_ptr<Schema>> _input_schemas,
    std::shared_ptr<VectorJoinIndexedProperties> _properties)
        : ::maximus::AbstractVectorJoinOperator(_ctx, _input_schemas, _properties)
        , ::maximus::gpu::GpuOperator(_ctx, _input_schemas, get_id(), {0, 1})
        , properties(_properties)
{
    set_device_type(DeviceType::GPU);
    set_engine_type(EngineType::FAISS);

    index = std::dynamic_pointer_cast<FaissIndex>(this->properties->index);
    
    search_parameters =
        (this->properties->index_parameters)
            ? std::dynamic_pointer_cast<FaissSearchParameters>(this->properties->index_parameters)
                  ->params
            : nullptr;

    operator_name = name();

    assert(output_schema);

    // KNN search supported on GPUs, range search is not
    assert(properties->K.has_value());
    assert(!properties->radius.has_value());
    // filter bitmap is not supported on the GPU
    assert(!properties->filter_bitmap.has_value());
    assert(!properties->filter_expr);
    // index must exist (properties can be empty if index is "Flat")
    assert(properties->index);
}

void JoinIndexedOperator::on_add_input(DeviceTablePtr device_input, int port) {
    proxy_add_input(device_input, port);
}

void JoinIndexedOperator::run_kernel(
    std::shared_ptr<MaximusContext>& ctx,
    std::vector<CudfTablePtr>& input_tables,
    std::vector<CudfTablePtr>& output_tables)
{
    assert(input_tables.size() == 2);
    assert(input_tables[0]);
    assert(input_tables[1]);
    
    // Move index to GPU if it's currently on CPU (lazy movement during execution)
    // Close operator region before index movement to treat it like other data movement
    if (index && index->is_on_cpu()) {
        std::cout << "[GPU JoinIndexedOperator] Moving CPU index to GPU for execution" << std::endl;
        profiler::close_regions({operator_name, "no_more_input"});
        profiler::open_regions({"DataTransformation", "CPU->GPU", "index_movement"});
        index = index->to_gpu();
        profiler::close_regions({"DataTransformation", "CPU->GPU", "index_movement"});
        profiler::open_regions({operator_name, "no_more_input"});
    }

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
    
    // Query vector column is always required
    auto query_field = props->query_vector_column.GetOne(*query_schema).ValueOrDie();
    int query_vec_idx = query_schema->GetFieldIndex(query_field->name());
    auto query_vec_col = query_table->view().column(query_vec_idx);
    assert(query_vec_col.type().id() == cudf::type_id::LIST);

    // Data vector column may be empty when using trained index (index owns the data)
    auto data_vec_col_name = props->data_vector_column.name();
    int64_t D;
    if (data_vec_col_name && !data_vec_col_name->empty()) {
        auto data_field  = props->data_vector_column.GetOne(*data_schema).ValueOrDie();
        int data_vec_idx  = data_schema->GetFieldIndex(data_field->name());
        auto data_vec_col  = data_table->view().column(data_vec_idx);
        assert(data_vec_col.type().id() == cudf::type_id::LIST);
        D = embedding_dimension(data_vec_col);
    } else {
        // Get dimensionality from the index when data column is not provided
        D = index->D;
    }
    assert(D == embedding_dimension(query_vec_col));

    assert(props->K.has_value());
    int64_t K = props->K.value();

    PE("ann_search_gpu");
    GpuSearchResult knn_result = ann_search_gpu(
        this->index,
        query_vec_col,   // LIST<FLOAT32>
        D,
        K,
        mr,
        stream,
        this->search_parameters.get());
    PL("ann_search_gpu");

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

void JoinIndexedOperator::on_no_more_input(int port) {
    proxy_no_more_input(port);
}

bool JoinIndexedOperator::has_more_batches_impl(bool blocking) {
    return proxy_has_more_batches(blocking);
}

DeviceTablePtr JoinIndexedOperator::export_next_batch_impl() {
    return std::move(proxy_export_next_batch());
}

}  // namespace maximus::faiss
