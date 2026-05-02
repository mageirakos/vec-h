#pragma once
#include <arrow/compute/api.h>

#include <cudf/ast/expressions.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/types.hpp>
#include <list>
#include <maximus/types/schema.hpp>

namespace maximus {

namespace gpu {

std::shared_ptr<::cudf::scalar> from_arrow_scalar(std::shared_ptr<arrow::Scalar> &arrow_scalar);

struct ExtExpression {
    enum class ExtType { IF, ENDS_WITH, STARTS_WITH, CONTAINS_RE, SUBSTR, DTYEAR };
    ExtType type;
    std::list<std::shared_ptr<::cudf::ast::expression>> args;
    int result_col;
    std::shared_ptr<::cudf::scalar> scalar = nullptr, scalar_ext = nullptr;
    std::shared_ptr<::cudf::strings::regex_program> regex = nullptr;

    ExtExpression(ExtType _type,
                  std::list<std::shared_ptr<::cudf::ast::expression>> _args,
                  int _result_col)
            : type(_type), args(_args), result_col(_result_col) {}

    ExtExpression(ExtType _type,
                  std::list<std::shared_ptr<::cudf::ast::expression>> _args,
                  int _result_col,
                  std::shared_ptr<::cudf::scalar> _scalar)
            : type(_type), args(_args), result_col(_result_col), scalar(_scalar) {}

    ExtExpression(ExtType _type,
                  std::list<std::shared_ptr<::cudf::ast::expression>> _args,
                  int _result_col,
                  std::shared_ptr<::cudf::strings::regex_program> _regex)
            : type(_type), args(_args), result_col(_result_col), regex(_regex) {}

    ExtExpression(ExtType _type,
                  std::list<std::shared_ptr<::cudf::ast::expression>> _args,
                  int _result_col,
                  std::shared_ptr<::cudf::scalar> _scalar,
                  std::shared_ptr<::cudf::scalar> _scalar_ext)
            : type(_type)
            , args(_args)
            , result_col(_result_col)
            , scalar(_scalar)
            , scalar_ext(_scalar_ext) {}
};

/**
 * To convert an arrow expression to a cudf expression.
 */
void arrow_expr_to_cudf(const arrow::compute::Expression &expr,
                        const std::shared_ptr<maximus::Schema> &_schema,
                        std::list<std::shared_ptr<::cudf::ast::expression>> &cudf_expr,
                        std::list<std::shared_ptr<::cudf::scalar>> &cudf_scalar,
                        std::list<ExtExpression> &ext_exprs);

/**
 * To get root data type of an expression
 */
::cudf::data_type get_expr_type(::cudf::ast::expression &expr,
                                std::vector<::cudf::data_type> &types);

}  // namespace gpu
}  // namespace maximus
