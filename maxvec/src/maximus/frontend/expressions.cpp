#include <iostream>
#include <maximus/frontend/query_plan_api.hpp>

namespace maximus {

cp::Expression int32_literal(int32_t value) {
    auto maybe_scalar = arrow::MakeScalar(arrow::int32(), value);
    if (!maybe_scalar.ok()) {
        CHECK_STATUS(maybe_scalar.status());
    }
    auto scalar = maybe_scalar.ValueOrDie();
    return cp::literal(std::move(scalar));
}

cp::Expression int64_literal(int64_t value) {
    auto maybe_scalar = arrow::MakeScalar(arrow::int64(), value);
    if (!maybe_scalar.ok()) {
        CHECK_STATUS(maybe_scalar.status());
    }
    auto scalar = maybe_scalar.ValueOrDie();
    return cp::literal(std::move(scalar));
}

cp::Expression float32_literal(float value) {
    auto maybe_scalar = arrow::MakeScalar(arrow::float32(), value);
    if (!maybe_scalar.ok()) {
        CHECK_STATUS(maybe_scalar.status());
    }
    auto scalar = maybe_scalar.ValueOrDie();
    return cp::literal(std::move(scalar));
}

cp::Expression float64_literal(double value) {
    auto maybe_scalar = arrow::MakeScalar(arrow::float64(), value);
    if (!maybe_scalar.ok()) {
        CHECK_STATUS(maybe_scalar.status());
    }
    auto scalar = maybe_scalar.ValueOrDie();
    return cp::literal(std::move(scalar));
}

cp::Expression date_literal(const std::string &dateStr) {
    // Split the date string into year, month, and day
    std::istringstream iss(dateStr);
    std::string token;
    std::vector<std::string> tokens;
    while (std::getline(iss, token, '-')) {
        tokens.push_back(token);
    }

    if (tokens.size() != 3) {
        // Invalid date string format
        throw std::runtime_error("Invalid date string format: " + dateStr);
    }

    int year  = std::stoi(tokens[0]);
    int month = std::stoi(tokens[1]);
    int day   = std::stoi(tokens[2]);

    // Calculate the number of days since January 1, 1970
    int daysSinceEpoch = (year - 1970) * 365 + (year - 1969) / 4;  // Account for leap years
    static const int daysInMonth[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    for (int i = 1; i < month; ++i) {
        daysSinceEpoch += daysInMonth[i];
    }
    if (month > 2 && (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0))) {
        // Adjust for leap year if necessary
        daysSinceEpoch += 1;
    }
    daysSinceEpoch += day - 1;  // Subtract 1 since days are 0-indexed

    return cp::literal(arrow::Date32Scalar(daysSinceEpoch));
}

cp::Expression string_literal(const std::string &valueStr) {
    return cp::literal(arrow::StringScalar(valueStr));
}

const std::unordered_map<std::string, std::string> arrow_binary_op_map = {{"+", "add"},
                                                                          {"-", "subtract"},
                                                                          {"*", "multiply"},
                                                                          {"/", "divide"},
                                                                          {"==", "equal"},
                                                                          {"!=", "not_equal"},
                                                                          {"<", "less"},
                                                                          {"<=", "less_equal"},
                                                                          {">", "greater"},
                                                                          {">=", "greater_equal"},
                                                                          {"&&", "and"},
                                                                          {"||", "or"}};

cp::Expression arrow_expr(cp::Expression left,
                          const std::string &op,
                          cp::Expression right,
                          const cp::FunctionOptions &options) {
    auto it = arrow_binary_op_map.find(op);
    if (it == arrow_binary_op_map.end()) {
        throw std::runtime_error("Unsupported operator: " + op);
    }

    return cp::call(it->second, {std::move(left), std::move(right)}, options);
}

cp::Expression arrow_expr(cp::Expression left, const std::string &op, cp::Expression right) {
    auto it = arrow_binary_op_map.find(op);
    if (it == arrow_binary_op_map.end()) {
        throw std::runtime_error("Unsupported operator: " + op);
    }

    return cp::call(it->second, {std::move(left), std::move(right)});
}

cp::Expression arrow_cast(cp::Expression expr, arrow::TypeHolder to_type) {
    return cp::call("cast", {std::move(expr)}, cp::CastOptions::Safe(std::move(to_type)));
}

cp::Expression arrow_product(std::vector<cp::Expression> expressions) {
    assert(!expressions.empty());

    cp::Expression result = expressions[0];
    for (size_t i = 1; i < expressions.size(); ++i) {
        // result = arrow_expr(arrow_cast(result, arrow::decimal(precision, scale)), "*", std::move(expressions[i]), cp::CastOptions::Unsafe(arrow::decimal(precision, scale)));
        result = arrow_expr(result, "*", std::move(expressions[i]));
    }
    return result;
}

cp::Expression arrow_like(cp::Expression expr, const std::string &regex) {
    auto string_options = arrow::compute::MatchSubstringOptions(regex, /*ignore_case=*/false);
    return cp::call("match_like", {std::move(expr)}, std::move(string_options));
}

cp::Expression arrow_field_like(const std::string &field, const std::string &regex) {
    return arrow_like(cp::field_ref(field), regex);
}

cp::Expression arrow_starts_with(cp::Expression expr, const std::string &prefix) {
    auto string_options = arrow::compute::MatchSubstringOptions(prefix, /*ignore_case=*/false);
    return cp::call("starts_with", {std::move(expr)}, std::move(string_options));
}

cp::Expression arrow_field_starts_with(const std::string &field, const std::string &prefix) {
    return arrow_starts_with(cp::field_ref(field), prefix);
}

cp::Expression arrow_ends_with(cp::Expression expr, const std::string &suffix) {
    auto string_options = arrow::compute::MatchSubstringOptions(suffix, /*ignore_case=*/false);
    return cp::call("ends_with", {std::move(expr)}, std::move(string_options));
}

cp::Expression arrow_field_ends_with(const std::string &field, const std::string &suffix) {
    return arrow_ends_with(cp::field_ref(field), suffix);
}

std::shared_ptr<Expression> expr(cp::Expression arrow_expr) {
    auto expr = std::make_shared<cp::Expression>(std::move(arrow_expr));
    return std::make_shared<Expression>(std::move(expr));
}

std::shared_ptr<Aggregate> aggregate(const std::string &op,
                                     std::shared_ptr<cp::FunctionOptions> options,
                                     const std::string &target,
                                     std::string name) {
    auto aggr = std::make_shared<Aggregate>(
        op, std::move(options), arrow::FieldRef(target), std::move(name));
    return std::move(aggr);
}

std::shared_ptr<Aggregate> aggregate(const std::string &op,
                                     const std::string &target,
                                     const std::string &name) {
    auto aggr = std::make_shared<Aggregate>(op, arrow::FieldRef(target), name);
    return std::move(aggr);
}

cp::Expression arrow_if_else(cp::Expression cond, cp::Expression _true, cp::Expression _false) {
    return cp::call("if_else", {std::move(cond), std::move(_true), std::move(_false)});
}

std::shared_ptr<cp::CountOptions> count_all() {
    return std::make_shared<cp::CountOptions>(arrow::compute::CountOptions::CountMode::ALL);
}

std::shared_ptr<cp::CountOptions> count_valid() {
    return std::make_shared<cp::CountOptions>(arrow::compute::CountOptions::CountMode::ONLY_VALID);
}

std::shared_ptr<cp::CountOptions> count_defaults() {
    return std::make_shared<cp::CountOptions>(arrow::compute::CountOptions::Defaults());
}

std::shared_ptr<cp::ScalarAggregateOptions> sum_defaults() {
    return std::make_shared<cp::ScalarAggregateOptions>(
        arrow::compute::ScalarAggregateOptions::Defaults());
}

std::shared_ptr<cp::ScalarAggregateOptions> median() {
    return std::make_shared<cp::ScalarAggregateOptions>(
        arrow::compute::ScalarAggregateOptions::Defaults());
}

std::shared_ptr<cp::VarianceOptions> stddev() {
    return std::make_shared<cp::VarianceOptions>(arrow::compute::VarianceOptions::Defaults());
}


std::shared_ptr<cp::ScalarAggregateOptions> sum_ignore_nulls() {
    return std::make_shared<arrow::compute::ScalarAggregateOptions>(
        /*skip_nulls=*/true, /*min_count=*/1);
}

std::vector<std::shared_ptr<Expression>> exprs(std::vector<std::string> column_refs) {
    std::vector<std::shared_ptr<Expression>> result(column_refs.size());
    for (size_t i = 0; i < column_refs.size(); ++i) {
        result[i] = Expression::from_field_ref(column_refs[i]);
    }
    return result;
}

cp::Expression arrow_in_range(const cp::Expression &expr,
                              const cp::Expression &lower,
                              const cp::Expression &upper) {
    return arrow_expr(arrow_expr(expr, ">=", lower), "&&", arrow_expr(expr, "<", upper));
}

cp::Expression arrow_between(const cp::Expression &expr,
                             const cp::Expression &lower,
                             const cp::Expression &upper) {
    return arrow_expr(arrow_expr(expr, ">=", lower), "&&", arrow_expr(expr, "<=", upper));
}

cp::Expression arrow_all(const std::vector<cp::Expression> &exprs) {
    return cp::and_(exprs);
}

cp::Expression arrow_any(const std::vector<cp::Expression> &exprs) {
    return cp::or_(exprs);
}

cp::Expression arrow_in(const cp::Expression &expr, const std::vector<cp::Expression> &values) {
    std::vector<cp::Expression> in_exprs(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        in_exprs[i] = arrow_expr(expr, "==", values[i]);
    }
    return arrow_any(in_exprs);
}

cp::Expression arrow_not_in(const cp::Expression &expr, const std::vector<cp::Expression> &values) {
    std::vector<cp::Expression> in_exprs(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        in_exprs[i] = arrow_expr(expr, "!=", values[i]);
    }
    return arrow_all(in_exprs);
}

cp::Expression arrow_not(const cp::Expression &expr) {
    return cp::call("invert", {expr});
}

cp::Expression arrow_equal(const cp::Expression &left, const cp::Expression &right) {
    return arrow_expr(left, "==", right);
}

cp::Expression arrow_abs(const cp::Expression &expr) {
    return cp::call("abs", {expr});
}

cp::Expression arrow_substring(const cp::Expression &expr, int start, int end) {
    return cp::call("utf8_slice_codeunits", {expr}, cp::SliceOptions(start, end, /*step=*/1));
}

cp::Expression year(const cp::Expression &date) {
    return cp::call("year", {date});
}

cp::Expression minute(const cp::Expression &timestamp) {
    return cp::call("minute", {timestamp});
}

cp::Expression arrow_is_null(const cp::Expression &expr) {
    return cp::is_null(expr);
}

cp::Expression arrow_is_not_null(const cp::Expression &expr) {
    return arrow_not(arrow_is_null(expr));
}

cp::Expression arrow_len(const cp::Expression &str) {
    return cp::call("utf8_length", {str});
}

cp::Expression arrow_extract(const cp::Expression &str, const std::string &regex) {
    auto string_options = arrow::compute::MatchSubstringOptions(regex, /*ignore_case=*/false);
    return cp::call("extract_regex", {str}, std::move(string_options));
}

}  // namespace maximus