#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/interop.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/gpu/cudf/project_operator.hpp>
#include <set>
#include <typeinfo>

namespace maximus::cudf {

ProjectOperator::ProjectOperator(std::shared_ptr<MaximusContext> &_ctx,
                                 std::shared_ptr<Schema> _input_schema,
                                 std::shared_ptr<ProjectProperties> _properties)
        : AbstractProjectOperator(_ctx, _input_schema, std::move(_properties))
        , GpuOperator(_ctx, {_input_schema}, get_id(), {}) {
    assert(ctx_);
    auto gctx = ctx_->get_gpu_context();

    assert(gctx != nullptr && "MaximusGContext must be initialized "
                              "before creating a GPU ProjectOperator");

    assert(typeid(*gctx) == typeid(maximus::gpu::MaximusCudaContext) &&
           "MaximusGContext must be a MaximusCudaContext");

    // convert expressions from arrow to cudf
    expression_list.clear();
    scalar_list.clear();
    ext_map.clear();

    int num_expressions = properties->project_expressions.size(), exp = 0;
    expression_list.resize(num_expressions);
    scalar_list.resize(num_expressions);
    ext_map.resize(num_expressions);

    auto e_iter = expression_list.begin();
    auto s_iter = scalar_list.begin();
    auto i_iter = ext_map.begin();
    for (int exp = 0; exp < num_expressions; ++exp, ++e_iter, ++s_iter, ++i_iter) {
        maximus::gpu::arrow_expr_to_cudf(*(properties->project_expressions[exp]->get_expression()),
                                         input_schemas[0],
                                         *e_iter,
                                         *s_iter,
                                         *i_iter);
        assert(!e_iter->empty());
        if (typeid(*e_iter->back()) != typeid(::cudf::ast::column_reference) || !i_iter->empty())
            move_flag = false;
    }

    // create the output schema
    arrow::FieldVector in_fields = input_schemas[0]->get_schema()->fields(), out_fields;
    std::vector<::cudf::data_type> in_types;
    std::transform(in_fields.begin(),
                   in_fields.end(),
                   std::back_inserter(in_types),
                   [&](const std::shared_ptr<arrow::Field> &f) {
                       return maximus::gpu::to_cudf_type(f->type());
                   });
    for (exp = 0, e_iter = expression_list.begin(), i_iter = ext_map.begin();
         e_iter != expression_list.end();
         ++exp, ++e_iter, ++i_iter) {
        std::transform(
            i_iter->begin(),
            i_iter->end(),
            std::back_inserter(in_types),
            [&](const gpu::ExtExpression &ext_expr) {
                ::cudf::data_type result;
                ::cudf::data_type t1, t2, t3;
                std::list<std::shared_ptr<::cudf::ast::expression>>::const_iterator iter;
                switch (ext_expr.type) {
                    case gpu::ExtExpression::ExtType::IF:
                        assert(
                            maximus::gpu::get_expr_type(*(*ext_expr.args.begin()), in_types).id() ==
                                ::cudf::type_id::BOOL8 &&
                            maximus::gpu::get_expr_type(*(*std::next(ext_expr.args.begin(), 1)),
                                                        in_types)
                                    .id() == maximus::gpu::get_expr_type(
                                                 *(*std::next(ext_expr.args.begin(), 2)), in_types)
                                                 .id());

                        result = maximus::gpu::get_expr_type(*ext_expr.args.back(), in_types);
                        break;
                    case gpu::ExtExpression::ExtType::ENDS_WITH:
                        result = ::cudf::data_type(::cudf::type_id::BOOL8);
                        break;
                    case gpu::ExtExpression::ExtType::STARTS_WITH:
                        result = ::cudf::data_type(::cudf::type_id::BOOL8);
                        break;
                    case gpu::ExtExpression::ExtType::CONTAINS_RE:
                        result = ::cudf::data_type(::cudf::type_id::BOOL8);
                        break;
                    case gpu::ExtExpression::ExtType::SUBSTR:
                        assert(std::static_pointer_cast<::cudf::ast::column_reference>(
                            ext_expr.args.front()));
                        result = ::cudf::data_type(::cudf::type_id::STRING);
                        break;
                    case gpu::ExtExpression::ExtType::DTYEAR:
                        result = ::cudf::data_type(::cudf::type_id::INT64);
                        break;
                    default:
                        result = ::cudf::data_type(::cudf::type_id::EMPTY);
                }
                return result;
            });
        
        // For simple column references (rename/select), use input field type directly
        // to avoid cuDF AST parser which doesn't support LIST types
        if (typeid(*e_iter->back()) == typeid(::cudf::ast::column_reference)) {
            int col_index = std::static_pointer_cast<::cudf::ast::column_reference>(e_iter->back())
                                ->get_column_index();
            // Bounds check to avoid segfault
            if (col_index >= 0 && col_index < static_cast<int>(in_fields.size())) {
                out_fields.push_back(std::make_shared<arrow::Field>(
                    properties->column_names[exp],
                    in_fields[col_index]->type()));
            } else {
                // DEBUG: Log out-of-bounds access
                std::cerr << "[ProjectOperator] WARNING: col_index=" << col_index 
                          << " out of bounds for in_fields.size()=" << in_fields.size()
                          << " for column name=" << properties->column_names[exp] << std::endl;
                // Fallback to AST-based type inference if index is out of bounds
                out_fields.push_back(std::make_shared<arrow::Field>(
                    properties->column_names[exp],
                    maximus::gpu::to_arrow_type(maximus::gpu::get_expr_type(*(e_iter->back()), in_types))));
            }
        } else {
            // For complex expressions, use AST-based type inference
            out_fields.push_back(std::make_shared<arrow::Field>(
                properties->column_names[exp],
                maximus::gpu::to_arrow_type(maximus::gpu::get_expr_type(*(e_iter->back()), in_types))));
        }
        in_types.resize(in_fields.size());
    }

    output_schema = std::make_shared<Schema>(arrow::schema(out_fields));

    set_device_type(DeviceType::GPU);
    set_engine_type(EngineType::CUDF);

    operator_name = name();
}

void ProjectOperator::on_add_input(DeviceTablePtr device_input, int port) {
    proxy_add_input(device_input, port);
}

void ProjectOperator::run_kernel(std::shared_ptr<MaximusContext> &ctx,
                                 std::vector<CudfTablePtr> &input_tables,
                                 std::vector<CudfTablePtr> &output_tables) {
    // there is only one input port
    assert(input_tables.size() == 1);
    assert(input_tables[0]);

    auto &input = input_tables[0];

    ::cudf::table_view cudf_view = input->view();

    // perform the projection
    std::vector<std::unique_ptr<::cudf::column>> output_columns, new_columns;
    int num_cols = cudf_view.num_columns();

    auto e_iter = expression_list.begin();
    auto i_iter = ext_map.begin();

    if (move_flag) {
        std::vector<std::unique_ptr<::cudf::column>> old_cols = input->release();
        for (; e_iter != expression_list.end(); ++e_iter) {
            assert(typeid(*e_iter->back()) == typeid(::cudf::ast::column_reference));
            int index = std::static_pointer_cast<::cudf::ast::column_reference>(e_iter->back())
                            ->get_column_index();
            assert(index < num_cols);
            output_columns.push_back(std::move(old_cols[index]));
        }
        std::shared_ptr<::cudf::table> result =
            std::make_shared<::cudf::table>(std::move(output_columns));
        output_tables.emplace_back(std::move(result));
        return;
    }

    for (int exp = 0; e_iter != expression_list.end(); ++exp, ++e_iter, ++i_iter) {
        new_columns.clear();

        for (auto ext_expr : *i_iter) {
            std::vector<std::unique_ptr<::cudf::column>> res_cols;
            ::cudf::column::contents temp_conts;
            int null_count = 0;

            switch (ext_expr.type) {
                case gpu::ExtExpression::ExtType::IF:
                    std::transform(ext_expr.args.begin(),
                                   ext_expr.args.end(),
                                   std::back_inserter(res_cols),
                                   [&](const std::shared_ptr<::cudf::ast::expression> &expr) {
                                       return ::cudf::compute_column(cudf_view, *expr);
                                   });
                    assert(res_cols.size() == 3);
                    assert(new_columns.size() + num_cols == ext_expr.result_col);
                    new_columns.push_back(::cudf::copy_if_else(
                        res_cols[1]->view(), res_cols[2]->view(), res_cols[0]->view()));
                    break;
                case gpu::ExtExpression::ExtType::ENDS_WITH:
                    assert(ext_expr.scalar != nullptr);
                    assert(ext_expr.args.size() == 1);
                    assert(new_columns.size() + num_cols == ext_expr.result_col);
                    new_columns.push_back(::cudf::strings::ends_with(
                        ::cudf::strings_column_view(cudf_view.column(
                            std::static_pointer_cast<::cudf::ast::column_reference>(
                                ext_expr.args.front())
                                ->get_column_index())),
                        *std::static_pointer_cast<::cudf::string_scalar>(ext_expr.scalar)));
                    break;
                case gpu::ExtExpression::ExtType::STARTS_WITH:
                    assert(ext_expr.scalar != nullptr);
                    assert(ext_expr.args.size() == 1);
                    assert(new_columns.size() + num_cols == ext_expr.result_col);
                    new_columns.push_back(::cudf::strings::starts_with(
                        ::cudf::strings_column_view(cudf_view.column(
                            std::static_pointer_cast<::cudf::ast::column_reference>(
                                ext_expr.args.front())
                                ->get_column_index())),
                        *std::static_pointer_cast<::cudf::string_scalar>(ext_expr.scalar)));
                    break;
                case gpu::ExtExpression::ExtType::CONTAINS_RE:
                    assert(ext_expr.regex != nullptr);
                    assert(ext_expr.args.size() == 1);
                    assert(new_columns.size() + num_cols == ext_expr.result_col);
                    new_columns.push_back(::cudf::strings::contains_re(
                        ::cudf::strings_column_view(cudf_view.column(
                            std::static_pointer_cast<::cudf::ast::column_reference>(
                                ext_expr.args.front())
                                ->get_column_index())),
                        *ext_expr.regex));
                    break;
                case gpu::ExtExpression::ExtType::SUBSTR:
                    assert(ext_expr.scalar != nullptr && ext_expr.scalar_ext != nullptr);
                    assert(ext_expr.args.size() == 1);
                    assert(new_columns.size() + num_cols == ext_expr.result_col);
                    new_columns.push_back(::cudf::strings::slice_strings(
                        ::cudf::strings_column_view(cudf_view.column(
                            std::static_pointer_cast<::cudf::ast::column_reference>(
                                ext_expr.args.front())
                                ->get_column_index())),
                        *std::static_pointer_cast<::cudf::numeric_scalar<int32_t>>(ext_expr.scalar),
                        *std::static_pointer_cast<::cudf::numeric_scalar<int32_t>>(
                            ext_expr.scalar_ext)));
                    if (cudf_view
                                .column(std::static_pointer_cast<::cudf::ast::column_reference>(
                                            ext_expr.args.front())
                                            ->get_column_index())
                                .child(0)
                                .type()
                                .id() == ::cudf::type_id::INT64 &&
                        new_columns.back()->child(0).type().id() == ::cudf::type_id::INT32) {
                        null_count = new_columns.back()->null_count();
                        temp_conts = new_columns.back()->release();
                        new_columns.pop_back();
                        res_cols.push_back(::cudf::cast(temp_conts.children[0]->view(),
                                                        ::cudf::data_type(::cudf::type_id::INT64)));
                        new_columns.push_back(std::make_unique<::cudf::column>(
                            ::cudf::data_type(::cudf::type_id::STRING),
                            res_cols[0]->size() - 1,
                            std::move(*temp_conts.data),
                            std::move(*temp_conts.null_mask),
                            null_count,
                            std::move(res_cols)));
                    }
                    break;
                case gpu::ExtExpression::ExtType::DTYEAR:
                    assert(ext_expr.args.size() == 1);
                    assert(new_columns.size() + num_cols == ext_expr.result_col);
                    new_columns.push_back(::cudf::cast(
                        ::cudf::datetime::extract_datetime_component(
                            cudf_view.column(
                                std::static_pointer_cast<::cudf::ast::column_reference>(
                                    ext_expr.args.front())
                                    ->get_column_index()),
                            ::cudf::datetime::datetime_component::YEAR)
                            ->view(),
                        ::cudf::data_type(::cudf::type_id::INT64)));
                    break;
                default:
                    assert(false && "Unsupported expression");
            }
            cudf_view =
                ::cudf::table_view({cudf_view, ::cudf::table_view({new_columns.back()->view()})});
        }

        std::shared_ptr<::cudf::ast::expression> expr = e_iter->back();

        if (typeid(*expr) == typeid(::cudf::ast::column_reference)) {
            int index =
                std::static_pointer_cast<::cudf::ast::column_reference>(expr)->get_column_index();
            if (index < num_cols)
                output_columns.push_back(std::make_unique<::cudf::column>(cudf_view.column(index)));
            else
                output_columns.push_back(std::move(new_columns[index - num_cols]));
        } else
            output_columns.push_back(::cudf::compute_column(cudf_view, *expr));
    }

    auto result = std::make_shared<::cudf::table>(std::move(output_columns));

    output_tables.emplace_back(std::move(result));
    assert(output_schema->get_schema() != nullptr);
    assert(output_schema->size() > 0);
}

void ProjectOperator::on_no_more_input(int port) {
    proxy_no_more_input(port);
}

bool ProjectOperator::has_more_batches_impl(bool blocking) {
    return proxy_has_more_batches(blocking);
}

DeviceTablePtr ProjectOperator::export_next_batch_impl() {
    return std::move(proxy_export_next_batch());
}

}  // namespace maximus::cudf
