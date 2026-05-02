#include <arrow/compute/api.h>

#include <maximus/types/aggregate.hpp>
#include <sstream>

namespace maximus {

Aggregate::Aggregate(std::shared_ptr<arrow::compute::Aggregate> aggr): aggr_(std::move(aggr)) {
}

Aggregate::Aggregate(std::string function,
                     std::shared_ptr<arrow::compute::FunctionOptions> options,
                     arrow::FieldRef target,
                     std::string name)
        : Aggregate(std::make_shared<arrow::compute::Aggregate>(function, options, target, name)) {
}

Aggregate::Aggregate(std::string function, arrow::FieldRef target, std::string name)
        : Aggregate(std::make_shared<arrow::compute::Aggregate>(function, target, name)) {
}

std::shared_ptr<arrow::compute::Aggregate> Aggregate::get_aggregate() {
    return aggr_;
}

const std::shared_ptr<arrow::compute::Aggregate> Aggregate::get_aggregate() const {
    return aggr_;
}

std::string Aggregate::to_string(int indent) const {
    std::string spaces(indent, ' ');
    std::stringstream ss;
    ss << "Aggregate(";
    if (!aggr_) {
        ss << spaces << "Invalid Aggregate";
        ss << ")";
        return ss.str();
    }
    assert(aggr_ && "Aggregate is null");
    assert(aggr_->target.size() > 0 && "Aggregate has no target fields");
    ss << spaces << "Target: [";
    for (auto& field : aggr_->target) {
        ss << field.ToString() << ", ";
    }
    ss << spaces << "]\n";
    ss << spaces << "function: " << aggr_->function << "\n";
    ss << spaces << "Name: " << aggr_->name << "\n";

    if (aggr_->options) {
        ss << "\n" << spaces << "Aggregate Options: ";
        ss << aggr_->options->ToString();
        ss << "\n" << spaces << ")";
    }

    return ss.str();
}

}  // namespace maximus
