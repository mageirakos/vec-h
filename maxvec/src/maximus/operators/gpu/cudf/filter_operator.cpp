#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/gpu/cudf/filter_operator.hpp>
#include <typeinfo>

namespace maximus::cudf {

FilterOperator::FilterOperator(std::shared_ptr<MaximusContext>& _ctx,
                               std::shared_ptr<Schema> _input_schema,
                               std::shared_ptr<FilterProperties> _properties)
        : AbstractFilterOperator(_ctx, _input_schema, std::move(_properties))
        , GpuOperator(_ctx, {_input_schema}, get_id(), {}) {
    assert(ctx_);
    auto gctx = ctx_->get_gpu_context();

    assert(gctx != nullptr && "MaximusGContext must be initialized "
                              "before creating a GPU FilterOperator");

    assert(typeid(*gctx) == typeid(maximus::gpu::MaximusCudaContext) &&
           "MaximusGContext must be a MaximusCudaContext");

    // convert expression from arrow to cudf
    expression_list.clear();
    scalar_list.clear();
    ext_map.clear();

    assert(_input_schema && _input_schema->size() > 0);
    assert(input_schemas[0] && input_schemas[0]->size() > 0);
    maximus::gpu::arrow_expr_to_cudf(*(properties->filter_expression->get_expression()),
                                     input_schemas[0],
                                     expression_list,
                                     scalar_list,
                                     ext_map);
    assert(!expression_list.empty());

    // create the output schema
    output_schema = std::make_shared<Schema>(*input_schemas[0]);
    assert(output_schema && output_schema->size() > 0);

    set_device_type(DeviceType::GPU);
    set_engine_type(EngineType::CUDF);

    operator_name = name();
}

void FilterOperator::on_add_input(DeviceTablePtr device_input, int port) {
    proxy_add_input(device_input, port);
}

void FilterOperator::run_kernel(std::shared_ptr<MaximusContext>& ctx,
                                std::vector<CudfTablePtr>& input_tables,
                                std::vector<CudfTablePtr>& output_tables) {
    // there is only one input port
    assert(input_tables.size() == 1);
    assert(input_tables.size() == 1);
    assert(input_tables[0]);
    auto& input = input_tables[0];

    ::cudf::table_view cudf_view     = input->view();
    ::cudf::table_view original_view = cudf_view;
    int num_columns                  = cudf_view.num_columns();

    std::vector<std::unique_ptr<::cudf::column>> new_cols;
    for (auto ext_expr : ext_map) {
        std::vector<std::unique_ptr<::cudf::column>> res_cols;
        switch (ext_expr.type) {
            case gpu::ExtExpression::ExtType::IF:
                std::transform(ext_expr.args.begin(),
                               ext_expr.args.end(),
                               std::back_inserter(res_cols),
                               [&](const std::shared_ptr<::cudf::ast::expression>& expr) {
                                   return ::cudf::compute_column(cudf_view, *expr);
                               });
                assert(res_cols.size() == 3);
                assert(new_cols.size() + num_columns == ext_expr.result_col);
                new_cols.push_back(::cudf::copy_if_else(
                    res_cols[1]->view(), res_cols[2]->view(), res_cols[0]->view()));
                break;
            case gpu::ExtExpression::ExtType::ENDS_WITH:
                assert(ext_expr.scalar != nullptr);
                assert(ext_expr.args.size() == 1);
                assert(new_cols.size() + num_columns == ext_expr.result_col);
                new_cols.push_back(::cudf::strings::ends_with(
                    ::cudf::strings_column_view(
                        cudf_view.column(std::static_pointer_cast<::cudf::ast::column_reference>(
                                             ext_expr.args.front())
                                             ->get_column_index())),
                    *std::static_pointer_cast<::cudf::string_scalar>(ext_expr.scalar)));
                break;
            case gpu::ExtExpression::ExtType::STARTS_WITH:
                assert(ext_expr.scalar != nullptr);
                assert(ext_expr.args.size() == 1);
                assert(new_cols.size() + num_columns == ext_expr.result_col);
                new_cols.push_back(::cudf::strings::starts_with(
                    ::cudf::strings_column_view(
                        cudf_view.column(std::static_pointer_cast<::cudf::ast::column_reference>(
                                             ext_expr.args.front())
                                             ->get_column_index())),
                    *std::static_pointer_cast<::cudf::string_scalar>(ext_expr.scalar)));
                break;
            case gpu::ExtExpression::ExtType::CONTAINS_RE:
                assert(ext_expr.regex != nullptr);
                assert(ext_expr.args.size() == 1);
                assert(new_cols.size() + num_columns == ext_expr.result_col);
                new_cols.push_back(::cudf::strings::contains_re(
                    ::cudf::strings_column_view(
                        cudf_view.column(std::static_pointer_cast<::cudf::ast::column_reference>(
                                             ext_expr.args.front())
                                             ->get_column_index())),
                    *ext_expr.regex));
                break;
            case gpu::ExtExpression::ExtType::SUBSTR:
                assert(ext_expr.scalar != nullptr && ext_expr.scalar_ext != nullptr);
                assert(ext_expr.args.size() == 1);
                assert(new_cols.size() + num_columns == ext_expr.result_col);
                new_cols.push_back(::cudf::strings::slice_strings(
                    ::cudf::strings_column_view(
                        cudf_view.column(std::static_pointer_cast<::cudf::ast::column_reference>(
                                             ext_expr.args.front())
                                             ->get_column_index())),
                    *std::static_pointer_cast<::cudf::numeric_scalar<int32_t>>(ext_expr.scalar),
                    *std::static_pointer_cast<::cudf::numeric_scalar<int32_t>>(
                        ext_expr.scalar_ext)));
                break;
            case gpu::ExtExpression::ExtType::DTYEAR:
                assert(ext_expr.args.size() == 1);
                assert(new_cols.size() + num_columns == ext_expr.result_col);
                new_cols.push_back(::cudf::cast(
                    ::cudf::datetime::extract_datetime_component(
                        cudf_view.column(std::static_pointer_cast<::cudf::ast::column_reference>(
                                             ext_expr.args.front())
                                             ->get_column_index()),
                        ::cudf::datetime::datetime_component::YEAR)
                        ->view(),
                    ::cudf::data_type(::cudf::type_id::INT64)));
                break;
            default:
                assert(false && "Unsupported expression");
        }

        ::cudf::column_view new_col = new_cols.back()->view();
        cudf_view = ::cudf::table_view({cudf_view, ::cudf::table_view({new_col})});
    }

    std::shared_ptr<::cudf::ast::expression> expr = expression_list.back();
    ::cudf::column_view filter_mask;
    std::unique_ptr<::cudf::column> filter_result = nullptr;
    profiler::open_regions({"filter_construct_output"});
    if (typeid(*expr) == typeid(::cudf::ast::column_reference))
        filter_mask = cudf_view.column(
            std::static_pointer_cast<::cudf::ast::column_reference>(expr)->get_column_index());
    else {
        filter_result = ::cudf::compute_column(cudf_view, *expr);
        filter_mask   = filter_result->view();
    }

    std::unique_ptr<::cudf::table> output_unique_table =
        ::cudf::apply_boolean_mask(original_view, filter_mask);
    profiler::close_regions({"filter_construct_output"});

    output_tables.emplace_back(std::move(output_unique_table));
    assert(output_schema->get_schema() != nullptr && output_schema->size() > 0);
}

void FilterOperator::on_no_more_input(int port) {
    proxy_no_more_input(port);
}

bool FilterOperator::has_more_batches_impl(bool blocking) {
    return proxy_has_more_batches(blocking);
}

DeviceTablePtr FilterOperator::export_next_batch_impl() {
    return std::move(proxy_export_next_batch());
}

}  // namespace maximus::cudf
