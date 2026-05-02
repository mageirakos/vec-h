#pragma once

#include <arrow/api.h>
#include <arrow/compute/api.h>

#include <maximus/types/aggregate.hpp>
#include <maximus/types/expression.hpp>
#include <string>

namespace maximus {
namespace cp = ::arrow::compute;

cp::Expression int32_literal(int32_t value);
cp::Expression int64_literal(int64_t value);
cp::Expression float32_literal(float value);
cp::Expression float64_literal(double value);

cp::Expression date_literal(const std::string& dateStr);

cp::Expression string_literal(const std::string& valueStr);

cp::Expression arrow_expr(cp::Expression left,
                          const std::string& op,
                          cp::Expression right,
                          const cp::FunctionOptions& options);

cp::Expression arrow_expr(cp::Expression left, const std::string& op, cp::Expression right);

cp::Expression arrow_cast(cp::Expression expr, arrow::TypeHolder to_type);

cp::Expression arrow_product(std::vector<cp::Expression> expressions);

cp::Expression arrow_starts_with(cp::Expression expr, const std::string& prefix);

cp::Expression arrow_field_starts_with(const std::string& field, const std::string& prefix);

cp::Expression arrow_ends_with(cp::Expression expr, const std::string& suffix);

cp::Expression arrow_field_ends_with(const std::string& field, const std::string& suffix);

cp::Expression arrow_like(cp::Expression expr, const std::string& regex);

cp::Expression arrow_field_like(const std::string& field, const std::string& regex);

std::shared_ptr<Expression> expr(cp::Expression arrow_expr);

std::shared_ptr<Aggregate> aggregate(const std::string& op,
                                     std::shared_ptr<cp::FunctionOptions> options,
                                     const std::string& target,
                                     std::string name = "");

std::shared_ptr<Aggregate> aggregate(const std::string& op,
                                     const std::string& target,
                                     const std::string& name = "");

cp::Expression arrow_if_else(cp::Expression cond, cp::Expression _true, cp::Expression _false);

std::shared_ptr<cp::CountOptions> count_all();

std::shared_ptr<cp::CountOptions> count_valid();

std::shared_ptr<cp::CountOptions> count_defaults();

std::shared_ptr<cp::ScalarAggregateOptions> sum_defaults();

std::shared_ptr<cp::ScalarAggregateOptions> median();

std::shared_ptr<cp::VarianceOptions> stddev();

std::shared_ptr<cp::ScalarAggregateOptions> sum_ignore_nulls();

std::vector<std::shared_ptr<Expression>> exprs(std::vector<std::string> column_refs);

cp::Expression arrow_in_range(const cp::Expression& expr,
                              const cp::Expression& lower,
                              const cp::Expression& upper);

cp::Expression arrow_between(const cp::Expression& expr,
                             const cp::Expression& lower,
                             const cp::Expression& upper);

cp::Expression arrow_all(const std::vector<cp::Expression>& exprs);

cp::Expression arrow_any(const std::vector<cp::Expression>& exprs);

cp::Expression arrow_in(const cp::Expression& expr, const std::vector<cp::Expression>& values);

cp::Expression arrow_not_in(const cp::Expression& expr, const std::vector<cp::Expression>& values);

cp::Expression arrow_not(const cp::Expression& expr);

cp::Expression arrow_equal(const cp::Expression& left, const cp::Expression& right);

cp::Expression arrow_abs(const cp::Expression& expr);

cp::Expression arrow_substring(const cp::Expression& expr, int start, int end);

cp::Expression year(const cp::Expression& date);

cp::Expression minute(const cp::Expression& timestamp);

cp::Expression arrow_is_null(const cp::Expression& expr);

cp::Expression arrow_is_not_null(const cp::Expression& expr);

cp::Expression arrow_len(const cp::Expression& str);

cp::Expression arrow_extract(const cp::Expression& str, const std::string& regex);
}  // namespace maximus
