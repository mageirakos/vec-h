#pragma once

#include <arrow/compute/api.h>

#include <functional>

namespace maximus {

class Aggregate {
public:
    Aggregate(std::shared_ptr<arrow::compute::Aggregate> aggregate);
    Aggregate(std::string function,
              std::shared_ptr<arrow::compute::FunctionOptions> options,
              arrow::FieldRef target,
              std::string name = "");

    Aggregate(std::string function, arrow::FieldRef target, std::string name = "");

    std::shared_ptr<arrow::compute::Aggregate> get_aggregate();

    const std::shared_ptr<arrow::compute::Aggregate> get_aggregate() const;

    std::string to_string(int indent = 0) const;

protected:
    std::shared_ptr<arrow::compute::Aggregate> aggr_;
};
}  // namespace maximus
