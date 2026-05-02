#pragma once

#include <arrow/api.h>
#include <arrow/type_traits.h>

#include <maximus/types/types.hpp>
#include <vector>

namespace maximus {
class Schema {
public:
    Schema() = default;

    Schema(std::shared_ptr<arrow::Schema> schema);
    Schema(std::initializer_list<std::shared_ptr<arrow::Field>> fields);
    Schema(std::vector<std::shared_ptr<arrow::Field>> fields);

    bool operator==(const Schema& other) const;

    template<typename T>
    void append_column(const std::string& name, const T& value) {
        auto type  = arrow::TypeTraits<typename arrow::CTypeTraits<T>::ArrowType>::type_singleton();
        auto field = arrow::field(name, type);
        auto maybe_schema = schema_->AddField(schema_->num_fields(), field);
        if (!maybe_schema.ok()) {
            CHECK_ARROW_STATUS(maybe_schema.status());
        }
        schema_ = maybe_schema.ValueOrDie();
    }

    void pop_column();

    int column_index(const std::string& column_name) const;

    bool subset_of(const Schema& other) const;

    bool superset_of(const Schema& other) const;

    std::shared_ptr<arrow::Schema>& get_schema();

    std::vector<std::string> column_names() const;

    const std::shared_ptr<arrow::Schema>& get_schema() const;

    std::shared_ptr<Schema> subschema(std::vector<int32_t>& projected_columns) const;

    std::shared_ptr<Schema> subschema(const std::vector<std::string>& include_columns) const;

    std::size_t size() const;

    std::string to_string(int indent = 0) const;

    std::unordered_map<std::string, std::shared_ptr<arrow::DataType>> column_types() const;

private:
    std::shared_ptr<arrow::Schema> schema_;
};
}  // namespace maximus
