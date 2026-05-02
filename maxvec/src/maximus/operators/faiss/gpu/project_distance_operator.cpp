#include <faiss/IndexFlat.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>

#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/lists/lists_column_view.hpp>

#include <maximus/operators/faiss/interop.hpp>
#include <maximus/operators/faiss/gpu/project_distance_operator.hpp>
#include <maximus/operators/faiss/gpu/faiss_kernels.hpp>
#include <maximus/utils/cudf_helpers.hpp>

namespace maximus::faiss::gpu {

ProjectDistanceOperator::ProjectDistanceOperator(
    std::shared_ptr<MaximusContext> &_ctx,
    std::vector<std::shared_ptr<Schema>> _input_schemas,
    std::shared_ptr<VectorProjectDistanceProperties> _properties)
        : ::maximus::AbstractVectorProjectDistanceOperator(_ctx, _input_schemas, std::move(_properties))
        , ::maximus::gpu::GpuOperator(_ctx, _input_schemas, get_id(), {0, 1})
{
    set_device_type(DeviceType::GPU);
    set_engine_type(EngineType::FAISS);

    operator_name = name();

    assert(output_schema);
}

void ProjectDistanceOperator::on_add_input(DeviceTablePtr device_input, int port) {
    proxy_add_input(device_input, port);
}

void ProjectDistanceOperator::run_kernel(
    std::shared_ptr<MaximusContext>& ctx,
    std::vector<CudfTablePtr>& input_tables,
    std::vector<CudfTablePtr>& output_tables)
{
    assert(input_tables.size() == 2);
    assert(input_tables[0]);
    assert(input_tables[1]);

    auto const& props = properties;

    auto left_table  = input_tables[0];
    auto right_table = input_tables[1];

    auto stream = ctx->get_kernel_stream();
    auto& mr    = ctx->mr;

    // ---------------------------------------------------------------------
    // Extract vector columns
    // ---------------------------------------------------------------------
    auto left_schema  = input_schemas[0]->get_schema();
    auto right_schema = input_schemas[1]->get_schema();

    auto left_field  = props->left_vector_column.GetOne(*left_schema).ValueOrDie();
    auto right_field = props->right_vector_column.GetOne(*right_schema).ValueOrDie();

    int left_vec_idx  = left_schema->GetFieldIndex(left_field->name());
    int right_vec_idx = right_schema->GetFieldIndex(right_field->name());

    auto left_vec_col  = left_table->view().column(left_vec_idx);
    auto right_vec_col = right_table->view().column(right_vec_idx);

    // LIST<FLOAT32> sanity checks
    assert(left_vec_col.type().id() == cudf::type_id::LIST);
    assert(right_vec_col.type().id() == cudf::type_id::LIST);

    // Dimension D (assume fixed-size embeddings)
    auto D = embedding_dimension(left_vec_col);

    // ---------------------------------------------------------------------
    // Compute pairwise distances (row-major: left x right)
    // ---------------------------------------------------------------------
    auto distances = pairwise_distances_gpu(
        /* data_vectors  = */ right_vec_col,  // nb
        /* query_vectors = */ left_vec_col,   // nq
        D,
        ::faiss::METRIC_L2,
        mr,
        stream);

    // distances size == left_rows * right_rows

    // ---------------------------------------------------------------------
    // Cross-join indices
    // ---------------------------------------------------------------------
    cudf::size_type n_left  = left_table->num_rows();
    cudf::size_type n_right = right_table->num_rows();
    cudf::size_type total   = n_left * n_right;

    // left_row_id = i / n_right
    // right_row_id = i % n_right
    auto row_ids =
        maximus::make_sequence_column(total, 1, mr, stream);

    cudf::numeric_scalar<int64_t> n_right_scalar(n_right, true, stream);

    auto left_ids = cudf::binary_operation(
        row_ids->view(),
        n_right_scalar,
        cudf::binary_operator::DIV,
        cudf::data_type{cudf::type_id::INT64},
        stream,
        mr);

    auto right_ids = cudf::binary_operation(
        row_ids->view(),
        n_right_scalar,
        cudf::binary_operator::MOD,
        cudf::data_type{cudf::type_id::INT64},
        stream,
        mr);

    // ---------------------------------------------------------------------
    // Gather columns
    // ---------------------------------------------------------------------
    std::vector<std::unique_ptr<cudf::column>> output_columns;

    // ---------------------------------------------------------------------
    // Gather right columns (Query) - FIRST
    // ---------------------------------------------------------------------
    for (int i = 0; i < right_table->num_columns(); ++i) {
        if (!props->keep_right_vector_column && i == right_vec_idx) {
            continue;
        }

        auto gathered = cudf::gather(
            cudf::table_view({ right_table->view().column(i) }),
            right_ids->view(),
            cudf::out_of_bounds_policy::DONT_CHECK,
            stream,
            mr);

        output_columns.push_back(std::move(gathered->release()[0]));
    }

    // ---------------------------------------------------------------------
    // Gather left columns (Data) - SECOND
    // ---------------------------------------------------------------------
    for (int i = 0; i < left_table->num_columns(); ++i) {
        if (!props->keep_left_vector_column && i == left_vec_idx) {
            continue;
        }

        auto gathered = cudf::gather(
            cudf::table_view({ left_table->view().column(i) }),
            left_ids->view(),
            cudf::out_of_bounds_policy::DONT_CHECK,
            stream,
            mr);

        output_columns.push_back(std::move(gathered->release()[0]));
    }

    // ---------------------------------------------------------------------
    // Append distance column
    // ---------------------------------------------------------------------
    output_columns.push_back(std::move(distances));

    // ---------------------------------------------------------------------
    // Final output table
    // ---------------------------------------------------------------------
    auto output_table =
        std::make_shared<cudf::table>(std::move(output_columns));

    output_tables.push_back(std::move(output_table));
}

void ProjectDistanceOperator::on_no_more_input(int port) {
    proxy_no_more_input(port);
}

bool ProjectDistanceOperator::has_more_batches_impl(bool blocking) {
    return proxy_has_more_batches(blocking);
}

DeviceTablePtr ProjectDistanceOperator::export_next_batch_impl() {
    return std::move(proxy_export_next_batch());
}

}  // namespace maximus::faiss
