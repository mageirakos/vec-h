#pragma once

#include <arrow/compute/api.h>

#include <functional>

namespace maximus {

class Expression {
public:
    Expression(std::shared_ptr<arrow::compute::Expression> expr);

    static std::shared_ptr<Expression> from_field_ref(const std::string field_name);

    bool operator==(const Expression& other) const;

    std::shared_ptr<arrow::compute::Expression> get_expression();

    const std::shared_ptr<arrow::compute::Expression> get_expression() const;

    std::string to_string(int indent = 0) const;

protected:
    std::shared_ptr<arrow::compute::Expression> expr_;
};
}  // namespace maximus
