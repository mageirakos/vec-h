#include <arrow/c/bridge.h>

#include <cudf/ast/detail/expression_parser.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table_view.hpp>
#include <maximus/gpu/cudf/cudf_expr.hpp>
#include <maximus/gpu/cudf/cudf_types.hpp>
#include <maximus/types/types.hpp>
#include <regex>
#include <typeinfo>

#define MAXIMUS_CUDF_EXPR_BINARY(args, schema, op, outexpr, outscalar)                      \
    arrow_expr_to_cudf(args[0], schema, outexpr, outscalar, ext_exprs);                     \
    auto &lhs = *(outexpr.back());                                                          \
    arrow_expr_to_cudf(args[1], schema, outexpr, outscalar, ext_exprs);                     \
    auto &rhs = *(outexpr.back());                                                          \
    outexpr.push_back(                                                                      \
        std::make_shared<::cudf::ast::operation>(::cudf::ast::ast_operator::op, lhs, rhs)); \
    return

#define MAXIMUS_CUDF_EXPR_UNARY(args, schema, op, outexpr, outscalar)                         \
    arrow_expr_to_cudf(args[0], schema, outexpr, outscalar, ext_exprs);                       \
    outexpr.push_back(std::make_shared<::cudf::ast::operation>(::cudf::ast::ast_operator::op, \
                                                               *(outexpr.back())));           \
    return

#define MAXIMUS_CUDF_NUM_SCALAR(intype, inscalar, outexpr, outscalar)                     \
    auto num_scalar = std::static_pointer_cast<::cudf::numeric_scalar<intype>>(inscalar); \
    outexpr.push_back(std::make_shared<::cudf::ast::literal>(*num_scalar));               \
    return

#define MAXIMUS_CUDF_NUM_TIME(intype, inscalar, outexpr, outscalar)                        \
    auto ts_scalar = std::static_pointer_cast<::cudf::timestamp_scalar<intype>>(inscalar); \
    outexpr.push_back(std::make_shared<::cudf::ast::literal>(*ts_scalar));                 \
    return

#define MAXIMUS_CUDF_FIELD_REF(schema, ref, index)                                          \
    arrow::FieldVector fields                 = schema->fields();                           \
    std::vector<arrow::FieldPath> field_paths = ref->FindAll(fields);                       \
    assert(field_paths.size() == 1 && "Field path ambiguous");                              \
    arrow::Result<std::shared_ptr<arrow::Field>> field_result = field_paths[0].Get(fields); \
    assert(field_result.ok() && "Field not found");                                         \
    std::shared_ptr<arrow::Field> field = field_result.ValueOrDie();                        \
    index = std::find(fields.begin(), fields.end(), field) - fields.begin()

namespace maximus {
namespace gpu {

std::shared_ptr<::cudf::scalar> from_arrow_scalar(std::shared_ptr<arrow::Scalar> &arrow_scalar) {
    if (!arrow_scalar) {
        return nullptr;
    }

    auto maybe_array = arrow::MakeArrayFromScalar(*arrow_scalar, 1);
    if (!maybe_array.ok()) {
        CHECK_STATUS(maybe_array.status());
    }
    auto array = maybe_array.ValueOrDie();

    ArrowSchema schema;
    CHECK_STATUS(arrow::ExportType(*array->type(), &schema));

    ArrowArray arr;
    CHECK_STATUS(arrow::ExportArray(*array, &arr));

    auto col = cudf::from_arrow_column(&schema, &arr);
    if (!col) {
        arr.release(&arr);
        schema.release(&schema);
        return nullptr;
    }

    auto ret = cudf::get_element(col->view(), 0);
    arr.release(&arr);
    schema.release(&schema);

    return std::shared_ptr<cudf::scalar>(std::move(ret));
}

void arrow_expr_to_cudf(const arrow::compute::Expression &expr,
                        const std::shared_ptr<maximus::Schema> &_schema,
                        std::list<std::shared_ptr<::cudf::ast::expression>> &c_expr,
                        std::list<std::shared_ptr<::cudf::scalar>> &c_scalar,
                        std::list<ExtExpression> &ext_exprs) {
    assert(_schema);
    assert(_schema->size() > 0);

    std::shared_ptr<arrow::Schema> schema = _schema->get_schema();

    if (const arrow::FieldRef *ref = expr.field_ref()) {
        // case: column reference
        // std::cout << "schema inside project = " << schema->ToString() << std::endl;
        // std::cout << "ref = " << *ref->name() << std::endl;
        MAXIMUS_CUDF_FIELD_REF(schema, ref, int index);

        c_expr.push_back(std::make_shared<cudf::ast::column_reference>(index));
        return;

    } else if (const arrow::compute::Expression::Call *call = expr.call()) {
        // case: function call
        if (call->function_name == "add") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, ADD, c_expr, c_scalar);
        } else if (call->function_name == "subtract") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, SUB, c_expr, c_scalar);
        } else if (call->function_name == "multiply") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, MUL, c_expr, c_scalar);
        } else if (call->function_name == "divide") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, TRUE_DIV, c_expr, c_scalar);
        } else if (call->function_name == "power") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, POW, c_expr, c_scalar);
        } else if (call->function_name == "equal") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, NULL_EQUAL, c_expr, c_scalar);
        } else if (call->function_name == "not_equal") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, NOT_EQUAL, c_expr, c_scalar);
        } else if (call->function_name == "less") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, LESS, c_expr, c_scalar);
        } else if (call->function_name == "greater") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, GREATER, c_expr, c_scalar);
        } else if (call->function_name == "less_equal") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, LESS_EQUAL, c_expr, c_scalar);
        } else if (call->function_name == "greater_equal") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, GREATER_EQUAL, c_expr, c_scalar);
        } else if (call->function_name == "bit_wise_and") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, BITWISE_AND, c_expr, c_scalar);
        } else if (call->function_name == "bit_wise_or") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, BITWISE_OR, c_expr, c_scalar);
        } else if (call->function_name == "bit_wise_xor") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, BITWISE_XOR, c_expr, c_scalar);
        } else if (call->function_name == "and") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, LOGICAL_AND, c_expr, c_scalar);
        } else if (call->function_name == "and_kleene") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, NULL_LOGICAL_AND, c_expr, c_scalar);
        } else if (call->function_name == "or") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, LOGICAL_OR, c_expr, c_scalar);
        } else if (call->function_name == "or_kleene") {
            MAXIMUS_CUDF_EXPR_BINARY(call->arguments, _schema, NULL_LOGICAL_OR, c_expr, c_scalar);
        } else if (call->function_name == "is_null") {
            MAXIMUS_CUDF_EXPR_UNARY(call->arguments, _schema, IS_NULL, c_expr, c_scalar);
        } else if (call->function_name == "sin") {
            MAXIMUS_CUDF_EXPR_UNARY(call->arguments, _schema, SIN, c_expr, c_scalar);
        } else if (call->function_name == "cos") {
            MAXIMUS_CUDF_EXPR_UNARY(call->arguments, _schema, COS, c_expr, c_scalar);
        } else if (call->function_name == "tan") {
            MAXIMUS_CUDF_EXPR_UNARY(call->arguments, _schema, TAN, c_expr, c_scalar);
        } else if (call->function_name == "asin") {
            MAXIMUS_CUDF_EXPR_UNARY(call->arguments, _schema, ARCSIN, c_expr, c_scalar);
        } else if (call->function_name == "acos") {
            MAXIMUS_CUDF_EXPR_UNARY(call->arguments, _schema, ARCCOS, c_expr, c_scalar);
        } else if (call->function_name == "atan") {
            MAXIMUS_CUDF_EXPR_UNARY(call->arguments, _schema, ARCTAN, c_expr, c_scalar);
        } else if (call->function_name == "exp") {
            MAXIMUS_CUDF_EXPR_UNARY(call->arguments, _schema, EXP, c_expr, c_scalar);
        } else if (call->function_name == "ln") {
            MAXIMUS_CUDF_EXPR_UNARY(call->arguments, _schema, LOG, c_expr, c_scalar);
        } else if (call->function_name == "sqrt") {
            MAXIMUS_CUDF_EXPR_UNARY(call->arguments, _schema, SQRT, c_expr, c_scalar);
        } else if (call->function_name == "ceil") {
            MAXIMUS_CUDF_EXPR_UNARY(call->arguments, _schema, CEIL, c_expr, c_scalar);
        } else if (call->function_name == "floor") {
            MAXIMUS_CUDF_EXPR_UNARY(call->arguments, _schema, FLOOR, c_expr, c_scalar);
        } else if (call->function_name == "abs") {
            MAXIMUS_CUDF_EXPR_UNARY(call->arguments, _schema, ABS, c_expr, c_scalar);
        } else if (call->function_name == "round") {
            MAXIMUS_CUDF_EXPR_UNARY(call->arguments, _schema, RINT, c_expr, c_scalar);
        } else if (call->function_name == "bit_wise_not") {
            MAXIMUS_CUDF_EXPR_UNARY(call->arguments, _schema, BIT_INVERT, c_expr, c_scalar);
        } else if (call->function_name == "invert") {
            MAXIMUS_CUDF_EXPR_UNARY(call->arguments, _schema, NOT, c_expr, c_scalar);
        } else if (call->function_name == "cast") {
            // this one is tricky
            // we need to get the type of the expression
            assert(false && "Not implemented yet");
            return;
        } else if (call->function_name == "if_else") {
            // solve each subexpression separately

            arrow_expr_to_cudf(call->arguments[0], _schema, c_expr, c_scalar, ext_exprs);
            auto cond = c_expr.back();
            arrow_expr_to_cudf(call->arguments[1], _schema, c_expr, c_scalar, ext_exprs);
            auto lhs = c_expr.back();
            arrow_expr_to_cudf(call->arguments[2], _schema, c_expr, c_scalar, ext_exprs);
            auto rhs = c_expr.back();

            ext_exprs.push_back(ExtExpression(
                ExtExpression::ExtType::IF,
                {cond, lhs, rhs},
                ext_exprs.empty() ? schema->num_fields() : ext_exprs.back().result_col + 1));

            c_expr.push_back(
                std::make_shared<::cudf::ast::column_reference>(ext_exprs.back().result_col));

            return;
        } else if (call->function_name == "ends_with") {
            arrow_expr_to_cudf(call->arguments[0], _schema, c_expr, c_scalar, ext_exprs);
            auto col = c_expr.back();

            // create a string scalar
            std::shared_ptr<arrow::compute::MatchSubstringOptions> options =
                std::static_pointer_cast<arrow::compute::MatchSubstringOptions>(call->options);
            assert(options);
            std::shared_ptr<::cudf::scalar> str_scalar =
                std::make_shared<::cudf::string_scalar>(options->pattern);

            ext_exprs.push_back(ExtExpression(
                ExtExpression::ExtType::ENDS_WITH,
                {col},
                ext_exprs.empty() ? schema->num_fields() : ext_exprs.back().result_col + 1,
                str_scalar));

            c_expr.push_back(
                std::make_shared<::cudf::ast::column_reference>(ext_exprs.back().result_col));
            c_scalar.push_back(str_scalar);
            return;
        } else if (call->function_name == "starts_with") {
            arrow_expr_to_cudf(call->arguments[0], _schema, c_expr, c_scalar, ext_exprs);
            auto col = c_expr.back();

            // create a string scalar
            std::shared_ptr<arrow::compute::MatchSubstringOptions> options =
                std::static_pointer_cast<arrow::compute::MatchSubstringOptions>(call->options);
            assert(options);
            std::shared_ptr<::cudf::scalar> str_scalar =
                std::make_shared<::cudf::string_scalar>(options->pattern);

            ext_exprs.push_back(ExtExpression(
                ExtExpression::ExtType::STARTS_WITH,
                {col},
                ext_exprs.empty() ? schema->num_fields() : ext_exprs.back().result_col + 1,
                str_scalar));

            c_expr.push_back(
                std::make_shared<::cudf::ast::column_reference>(ext_exprs.back().result_col));
            c_scalar.push_back(str_scalar);
            return;
        } else if (call->function_name == "match_like") {
            arrow_expr_to_cudf(call->arguments[0], _schema, c_expr, c_scalar, ext_exprs);
            auto col = c_expr.back();

            // create a regex program
            std::shared_ptr<arrow::compute::MatchSubstringOptions> options =
                std::static_pointer_cast<arrow::compute::MatchSubstringOptions>(call->options);
            assert(options);

            std::regex percent("%");
            auto pattern = options->pattern;

            // Replace all occurrences of % with .*
            std::string cudf_pattern = std::regex_replace(pattern, percent, ".*");

            std::shared_ptr<::cudf::strings::regex_program> regex =
                std::move(::cudf::strings::regex_program::create(cudf_pattern));

            ext_exprs.push_back(ExtExpression(
                ExtExpression::ExtType::CONTAINS_RE,
                {col},
                ext_exprs.empty() ? schema->num_fields() : ext_exprs.back().result_col + 1,
                regex));

            c_expr.push_back(
                std::make_shared<::cudf::ast::column_reference>(ext_exprs.back().result_col));
            return;
        } else if (call->function_name == "utf8_slice_codeunits") {
            arrow_expr_to_cudf(call->arguments[0], _schema, c_expr, c_scalar, ext_exprs);
            auto col = c_expr.back();

            std::shared_ptr<arrow::compute::SliceOptions> options =
                std::static_pointer_cast<arrow::compute::SliceOptions>(call->options);
            assert(options);

            std::shared_ptr<::cudf::scalar> start =
                                                std::make_shared<::cudf::numeric_scalar<int32_t>>(
                                                    (int32_t) options->start),
                                            stop =
                                                std::make_shared<::cudf::numeric_scalar<int32_t>>(
                                                    (int32_t) options->stop);

            ext_exprs.push_back(ExtExpression(
                ExtExpression::ExtType::SUBSTR,
                {col},
                ext_exprs.empty() ? schema->num_fields() : ext_exprs.back().result_col + 1,
                start,
                stop));

            c_expr.push_back(
                std::make_shared<::cudf::ast::column_reference>(ext_exprs.back().result_col));
            return;
        } else if (call->function_name == "year") {
            arrow_expr_to_cudf(call->arguments[0], _schema, c_expr, c_scalar, ext_exprs);
            auto col = c_expr.back();

            ext_exprs.push_back(ExtExpression(
                ExtExpression::ExtType::DTYEAR,
                {col},
                ext_exprs.empty() ? schema->num_fields() : ext_exprs.back().result_col + 1));

            c_expr.push_back(
                std::make_shared<::cudf::ast::column_reference>(ext_exprs.back().result_col));
            return;
        }

        else {
            std::__throw_runtime_error("Unsupported function call");
            return;
        }
    } else if (const arrow::Datum *datum = expr.literal()) {
        // case: literal
        std::shared_ptr<arrow::DataType> type = datum->type();
        std::shared_ptr<arrow::Scalar> scalar = datum->scalar();

        assert(scalar && type && "Something is wrong with the expression");

        ::cudf::data_type cudf_type                = to_cudf_type(maximus::to_maximus_type(type));
        std::shared_ptr<::cudf::scalar> dev_scalar = std::move(from_arrow_scalar(scalar));

        c_scalar.push_back(dev_scalar);

        if (cudf_type.id() == ::cudf::type_id::STRING) {
            std::shared_ptr<::cudf::string_scalar> str_scalar =
                std::static_pointer_cast<::cudf::string_scalar>(dev_scalar);

            c_expr.push_back(std::make_shared<::cudf::ast::literal>(*str_scalar));
            return;
        } else if (cudf_type.id() == ::cudf::type_id::TIMESTAMP_DAYS) {
            MAXIMUS_CUDF_NUM_TIME(::cudf::timestamp_D, dev_scalar, c_expr, c_scalar);
        } else if (cudf_type.id() == ::cudf::type_id::TIMESTAMP_NANOSECONDS) {
            MAXIMUS_CUDF_NUM_TIME(::cudf::timestamp_ns, dev_scalar, c_expr, c_scalar);
        } else if (cudf_type.id() == ::cudf::type_id::TIMESTAMP_MICROSECONDS) {
            MAXIMUS_CUDF_NUM_TIME(::cudf::timestamp_us, dev_scalar, c_expr, c_scalar);
        } else if (cudf_type.id() == ::cudf::type_id::TIMESTAMP_MILLISECONDS) {
            MAXIMUS_CUDF_NUM_TIME(::cudf::timestamp_ms, dev_scalar, c_expr, c_scalar);
        } else if (cudf_type.id() == ::cudf::type_id::TIMESTAMP_SECONDS) {
            MAXIMUS_CUDF_NUM_TIME(::cudf::timestamp_s, dev_scalar, c_expr, c_scalar);
        } else if (cudf_type.id() == ::cudf::type_id::INT8) {
            MAXIMUS_CUDF_NUM_SCALAR(int8_t, dev_scalar, c_expr, c_scalar);
        } else if (cudf_type.id() == ::cudf::type_id::INT16) {
            MAXIMUS_CUDF_NUM_SCALAR(int16_t, dev_scalar, c_expr, c_scalar);
        } else if (cudf_type.id() == ::cudf::type_id::INT32) {
            MAXIMUS_CUDF_NUM_SCALAR(int32_t, dev_scalar, c_expr, c_scalar);
        } else if (cudf_type.id() == ::cudf::type_id::INT64) {
            MAXIMUS_CUDF_NUM_SCALAR(int64_t, dev_scalar, c_expr, c_scalar);
        } else if (cudf_type.id() == ::cudf::type_id::UINT8) {
            MAXIMUS_CUDF_NUM_SCALAR(uint8_t, dev_scalar, c_expr, c_scalar);
        } else if (cudf_type.id() == ::cudf::type_id::UINT16) {
            MAXIMUS_CUDF_NUM_SCALAR(uint16_t, dev_scalar, c_expr, c_scalar);
        } else if (cudf_type.id() == ::cudf::type_id::UINT32) {
            MAXIMUS_CUDF_NUM_SCALAR(uint32_t, dev_scalar, c_expr, c_scalar);
        } else if (cudf_type.id() == ::cudf::type_id::UINT64) {
            MAXIMUS_CUDF_NUM_SCALAR(uint64_t, dev_scalar, c_expr, c_scalar);
        } else if (cudf_type.id() == ::cudf::type_id::FLOAT32) {
            MAXIMUS_CUDF_NUM_SCALAR(float, dev_scalar, c_expr, c_scalar);
        } else if (cudf_type.id() == ::cudf::type_id::FLOAT64) {
            MAXIMUS_CUDF_NUM_SCALAR(double, dev_scalar, c_expr, c_scalar);
        } else if (cudf_type.id() == ::cudf::type_id::BOOL8) {
            MAXIMUS_CUDF_NUM_SCALAR(bool, dev_scalar, c_expr, c_scalar);
        } else {
            assert(false && "Unsupported type");
            return;
        }
    }
    assert(false && "Unsupported expression");
    return;
}

::cudf::data_type get_expr_type(::cudf::ast::expression &expr,
                                std::vector<::cudf::data_type> &types) {
    std::vector<::cudf::column_view> cols(types.size());
    std::transform(types.begin(), types.end(), cols.begin(), [](::cudf::data_type &type) {
        return ::cudf::column_view(type, 0, nullptr, nullptr, 0, 0);
    });
    ::cudf::table_view table = ::cudf::table_view(cols);

    return ::cudf::ast::detail::expression_parser{
        expr, table, true, rmm::cuda_stream_default, rmm::mr::get_current_device_resource()}
        .output_type();
}

}  // namespace gpu
}  // namespace maximus
