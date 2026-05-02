#include <arrow/compute/api.h>

#include <maximus/types/expression.hpp>
#include <sstream>

namespace maximus {

Expression::Expression(std::shared_ptr<arrow::compute::Expression> expr): expr_(std::move(expr)) {
}

bool Expression::operator==(const Expression& other) const {
    return expr_->Equals(*(other.get_expression()));
}

std::shared_ptr<arrow::compute::Expression> Expression::get_expression() {
    return expr_;
}

const std::shared_ptr<arrow::compute::Expression> Expression::get_expression() const {
    return expr_;
}

std::string Expression::to_string(int indent) const {
    std::string spaces(indent, ' ');
    std::stringstream ss;
    ss << "Expression(";
    if (!expr_) {
        ss << spaces << "Invalid Expression";
        ss << ")";
        return ss.str();
    }
    ss << spaces << expr_->ToString();
    ss << "\n" << spaces << ")";
    return ss.str();
}

std::shared_ptr<Expression> Expression::from_field_ref(const std::string field_name) {
    auto _expr =
        std::make_shared<arrow::compute::Expression>(arrow::compute::field_ref(field_name));
    auto expr = std::make_shared<Expression>(std::move(_expr));
    return std::move(expr);
}
}  // namespace maximus
