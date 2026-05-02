#include <arrow/api.h>

#include <algorithm>
#include <iostream>
#include <maximus/error_handling.hpp>
#include <maximus/types/schema.hpp>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace maximus {

Schema::Schema(std::shared_ptr<arrow::Schema> schema): schema_(std::move(schema)) {
}

Schema::Schema(std::initializer_list<std::shared_ptr<arrow::Field>> fields)
        : Schema(std::make_shared<arrow::Schema>(std::move(fields))) {
}

Schema::Schema(std::vector<std::shared_ptr<arrow::Field>> fields)
        : Schema(std::make_shared<arrow::Schema>(std::move(fields))) {
}

int Schema::column_index(const std::string &column_name) const {
    if (!schema_) {
        return -1;
    }

    int index = schema_->GetFieldIndex(column_name);
    if (index < 0) {
        std::stringstream ss;
        ss << "Column not found: " << column_name << ", in schema: " << schema_->ToString();
        std::cout << ss.str() << std::endl;
    }
    assert(index >= 0);
    return index;
}

void Schema::pop_column() {
    auto maybe_schema = schema_->RemoveField(size() - 1);
    if (!maybe_schema.ok()) {
        CHECK_STATUS(maybe_schema.status());
    }
    schema_ = maybe_schema.ValueOrDie();
}

// Define the equality operator
bool Schema::operator==(const Schema &other) const {
    if (schema_ && other.schema_) {
        // Use Arrow's Equals method to compare the schemas
        return schema_->Equals(*other.schema_);
    }
    // If one or both shared_ptrs are nullptr, consider them equal if both are nullptr
    return schema_ == other.schema_;
}

bool Schema::subset_of(const Schema &other) const {
    if (!schema_ || !other.schema_) {
        return false;  // If either schema is nullptr, it's not a subset
    }

    // Get the fields of both schemas
    const auto &this_fields  = schema_->fields();
    const auto &other_fields = other.schema_->fields();

    // Check if all fields in this schema exist in the other schema
    for (const auto &this_field : this_fields) {
        bool found = false;
        for (const auto &other_field : other_fields) {
            if (this_field->Equals(other_field)) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;  // Found a field in this schema that is not in the other schema
        }
    }

    return true;  // All fields in this schema are in the other schema
}

bool Schema::superset_of(const Schema &other) const {
    if (!schema_ || !other.schema_) {
        return false;  // If either schema is nullptr, it can't be a superset
    }

    // Get the fields of both schemas
    const auto &this_fields  = schema_->fields();
    const auto &other_fields = other.schema_->fields();

    // Create a set of other schema's field names
    std::unordered_set<std::string> other_field_names;
    for (const auto &other_field : other_fields) {
        other_field_names.insert(other_field->name());
    }

    // Check if all fields in the other schema are in this schema
    for (const auto &this_field : this_fields) {
        if (other_field_names.find(this_field->name()) == other_field_names.end()) {
            return false;  // Found a field in this schema not in the other schema
        }
    }

    return true;  // All fields in the other schema are in this schema
}

std::shared_ptr<arrow::Schema> &Schema::get_schema() {
    return schema_;
}

const std::shared_ptr<arrow::Schema> &Schema::get_schema() const {
    return schema_;
}

std::shared_ptr<Schema> Schema::subschema(std::vector<int32_t> &projected_columns) const {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (auto i : projected_columns) {
        fields.push_back(schema_->field(i));
    }

    auto schema = std::make_shared<arrow::Schema>(fields);

    return std::make_shared<Schema>(schema);
}

std::shared_ptr<Schema> Schema::subschema(const std::vector<std::string> &include_columns) const {
    assert(schema_);
    // Vector to hold the fields for the new schema
    std::vector<std::shared_ptr<arrow::Field>> fields;

    std::vector<std::string> columns = include_columns;
    // if include_columns is empty, include all the columns
    if (columns.size() == 0) {
        columns = column_names();
    }

    // Iterate over the column names we want to include
    for (const auto &col_name : columns) {
        // Find the field in the original schema
        auto field = schema_->GetFieldByName(col_name);
        if (field != nullptr) {
            // If the field is found, add it to the new fields vector
            fields.emplace_back(field);
        }
    }

    // Create and return a new schema with the filtered fields
    auto schema = std::make_shared<arrow::Schema>(fields);

    return std::make_shared<Schema>(schema);
}

std::size_t Schema::size() const {
    return schema_->num_fields();
}

std::vector<std::string> Schema::column_names() const {
    std::vector<std::string> names;
    if (schema_) {
        for (int i = 0; i < schema_->num_fields(); ++i) {
            names.push_back(schema_->field(i)->name());
        }
    }
    return names;
}

std::unordered_map<std::string, std::shared_ptr<arrow::DataType>> Schema::column_types() const {
    std::unordered_map<std::string, std::shared_ptr<arrow::DataType>> types;
    if (schema_) {
        for (int i = 0; i < schema_->num_fields(); ++i) {
            auto field           = schema_->field(i);
            types[field->name()] = field->type();
        }
    }
    return std::move(types);
}

std::string Schema::to_string(int indent) const {
    std::string spaces(indent, ' ');
    std::stringstream ss;
    ss << "Schema(";
    if (!schema_) {
        ss << spaces << "Invalid Schema";
        ss << ")";
        return ss.str();
    }
    for (int i = 0; i < schema_->num_fields(); ++i) {
        auto &field = schema_->field(i);
        ss << "\n    " << spaces << field->ToString();
    }
    ss << "\n" << spaces << ")";
    return ss.str();
}
}  // namespace maximus
