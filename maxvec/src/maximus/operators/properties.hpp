#pragma once

#include <maximus/indexes/index.hpp>
#include <maximus/io/csv.hpp>
#include <maximus/types/types.hpp>
#include <maximus/types/aggregate.hpp>
#include <maximus/types/device_table_ptr.hpp>
#include <maximus/types/expression.hpp>
#include <maximus/types/node_type.hpp>
#include <maximus/types/schema.hpp>
#include <sstream>

namespace maximus {

class NodeProperties {
public:
    virtual ~NodeProperties() = default;

    [[nodiscard]] virtual std::string to_string() const { return "NodeProperties"; }
};

class LocalBroadcastProperties : public NodeProperties {
public:
    explicit LocalBroadcastProperties() = default;

    LocalBroadcastProperties(int num_output_ports): num_output_ports(num_output_ports) {}

    int num_output_ports = 0;

    bool should_replicate = true;

    [[nodiscard]] std::string to_string() const override { return "LocalBroadcastProperties"; }
};

class TableSinkProperties : public NodeProperties {
public:
    explicit TableSinkProperties() = default;

    [[nodiscard]] std::string to_string() const override { return "TableSinkProperties"; }
};

class RandomTableSourceProperties : public NodeProperties {
public:
    explicit RandomTableSourceProperties(std::shared_ptr<Schema> output_schema,
                                         std::size_t output_batch_size,
                                         std::size_t total_num_rows,
                                         int seed)
            : output_schema(std::move(output_schema))
            , output_batch_size(output_batch_size)
            , total_num_rows(total_num_rows)
            , seed(seed) {}

    std::shared_ptr<Schema> output_schema;
    std::size_t output_batch_size = 100;
    std::size_t total_num_rows    = 1000;
    int seed                      = 0;

    [[nodiscard]] std::string to_string() const override {
        std::stringstream ss;
        ss << "RandomTableSourceProperties {"
           << "output_batch_size: " << output_batch_size << ", total_num_rows: " << total_num_rows
           << ", seed: " << seed << "}";
        return ss.str();
    }
};

class TableSourceProperties : public NodeProperties {
public:
    explicit TableSourceProperties(std::string path,
                                   std::shared_ptr<Schema> schema           = nullptr,
                                   std::vector<std::string> include_columns = {})
            : path(path), schema(std::move(schema)), include_columns(std::move(include_columns)) {
        assert(this->path != "");
    }

    explicit TableSourceProperties(DeviceTablePtr _table,
                                   std::vector<std::string> include_columns = {})
            : table(std::move(_table)), include_columns(std::move(include_columns)) {
        assert(table);
        assert(table.is_table() || table.is_gtable());
        if (table.is_table()) {
            assert(table.as_table());
            schema = table.as_table()->get_schema();
        }
        if (table.is_gtable()) {
#ifdef MAXIMUS_WITH_CUDA
            assert(table.as_gtable());
            schema = table.as_gtable()->get_schema();
#else
            throw std::runtime_error("Maximus must be built with the CUDA support to use GTable");
#endif
        }
        assert(schema && schema->size() > 0);
    }

    // if the table is not given to this operator
    // then it will be read from the filesystem's path
    std::string path                         = "";
    std::shared_ptr<Schema> schema           = nullptr;
    std::vector<std::string> include_columns = {};

    // if the table is already provided to this operator
    // then it will be used directly
    DeviceTablePtr table;

    // the output batch size
    int64_t output_batch_size = -1;

    // checks if the table source comes from an in-memory device table
    bool has_table() const { return !table.is_none() && path == ""; }

    [[nodiscard]] std::string to_string() const override {
        std::stringstream ss;
        ss << "TableSourceProperties {";
        ss << "path: " << path << ", include_columns: [";
        for (const auto& field : include_columns) {
            ss << field << ", ";
        }
        ss << "],\nSchema: " << (schema ? schema->to_string() : "nullptr") << "}";
        return ss.str();
    }
};

class TableSourceFilterProjectProperties : public NodeProperties {
public:
    explicit TableSourceFilterProjectProperties(std::vector<std::string> paths,
                                                std::shared_ptr<Schema> schema           = nullptr,
                                                std::vector<std::string> include_columns = {},
                                                std::shared_ptr<Expression> filter_expr  = {},
                                                std::vector<std::shared_ptr<Expression>> exprs = {},
                                                std::vector<std::string> column_names          = {})
            : paths(paths)
            , schema(std::move(schema))
            , include_columns(std::move(include_columns))
            , filter_expr(filter_expr)
            , exprs(exprs)
            , column_names(column_names) {
        assert(!this->paths.empty());
    }

    // This operator will read the table from the filesystem's path
    // and apply appropriate filtering and projection
    std::vector<std::string> paths;
    std::shared_ptr<Schema> schema;
    std::vector<std::string> include_columns;
    std::shared_ptr<Expression> filter_expr;
    std::vector<std::shared_ptr<Expression>> exprs;
    std::vector<std::string> column_names;

    bool is_filter_only() const { return exprs.empty() && column_names.empty(); }

    [[nodiscard]] std::string to_string() const override {
        std::stringstream ss;
        ss << "TableSourceFilterProjectProperties {";
        ss << "paths: [";
        for (const auto& path : paths) {
            ss << path << ", ";
        }
        ss << "],\n include_columns: [";
        for (const auto& field : include_columns) {
            ss << field << ", ";
        }
        ss << "],\nSchema: " << (schema ? schema->to_string() : "nullptr");
        ss << ", filter_expr: " << filter_expr << "\n";
        ss << ", exprs: [";
        for (const auto& expr : exprs) {
            ss << expr << ", ";
        }
        ss << "],\n column_names: [";
        for (const auto& cn : column_names) {
            ss << cn << ", ";
        }
        ss << "]}";
        return ss.str();
    }
};

class FilterProperties : public NodeProperties {
public:
    explicit FilterProperties(std::shared_ptr<Expression> filter_expression)
            : filter_expression(std::move(filter_expression)) {}

    std::shared_ptr<Expression> filter_expression;

    [[nodiscard]] std::string to_string() const override {
        return "FilterProperties { filter_expression: " + filter_expression->to_string() + " }";
    }
};

class ProjectProperties : public NodeProperties {
public:
    ProjectProperties(std::vector<std::shared_ptr<Expression>> project_expressions,
                      std::vector<std::string> column_names = {})
            : project_expressions(std::move(project_expressions))
            , column_names(std::move(column_names)) {}

    std::vector<std::shared_ptr<Expression>> project_expressions;
    std::vector<std::string> column_names;

    [[nodiscard]] std::string to_string() const override {
        std::stringstream ss;
        ss << "ProjectProperties {"
           << "project_expressions: [";
        for (const auto& expr : project_expressions) {
            ss << expr->to_string() << ", ";
        }
        ss << "], column_names: [";
        for (const auto& name : column_names) {
            ss << name << ", ";
        }
        ss << "]}";
        return ss.str();
    }
};

enum class JoinType {
    LEFT_SEMI,
    RIGHT_SEMI,
    LEFT_ANTI,
    RIGHT_ANTI,
    INNER,
    LEFT_OUTER,
    RIGHT_OUTER,
    FULL_OUTER,
    CROSS_JOIN
};

class JoinProperties : public NodeProperties {
public:
    JoinProperties() = default;

    explicit JoinProperties(
        JoinType join_type,
        std::vector<arrow::FieldRef> left_keys,
        std::vector<arrow::FieldRef> right_keys,
        std::shared_ptr<Expression> filter = std::make_shared<Expression>(
            std::make_shared<arrow::compute::Expression>(arrow::compute::literal(true))),
        std::string left_output_suffix  = "",
        std::string right_output_suffix = "")
            : join_type(join_type)
            , left_keys(std::move(left_keys))
            , right_keys(std::move(right_keys))
            , left_suffix(std::move(left_output_suffix))
            , right_suffix(std::move(right_output_suffix))
            , filter(std::move(filter)) {}

    JoinType join_type = JoinType::INNER;
    std::vector<arrow::FieldRef> left_keys;
    std::vector<arrow::FieldRef> right_keys;
    std::string left_suffix            = "";
    std::string right_suffix           = "";
    std::shared_ptr<Expression> filter = std::make_shared<Expression>(
        std::make_shared<arrow::compute::Expression>(arrow::compute::literal(true)));

    [[nodiscard]] std::string to_string() const override {
        std::stringstream ss;
        ss << "JoinProperties {"
           << "join_type: " << static_cast<int>(join_type) << ", left_keys: [";
        for (const auto& key : left_keys) {
            ss << key.ToString() << ", ";
        }
        ss << "], right_keys: [";
        for (const auto& key : right_keys) {
            ss << key.ToString() << ", ";
        }
        ss << "], left_suffix: " << left_suffix << ", right_suffix: " << right_suffix
           << ", filter: " << filter->to_string() << "}";
        return ss.str();
    }
};

enum class SortOrder { ASCENDING, DESCENDING };

enum class NullOrder { FIRST, LAST };

struct SortKey {
    SortKey(arrow::FieldRef field, SortOrder order = SortOrder::ASCENDING)
            : field(std::move(field)), order(order) {}
    arrow::FieldRef field;
    SortOrder order;
};

class OrderByProperties : public NodeProperties {
public:
    OrderByProperties(std::vector<SortKey> sort_keys, NullOrder null_order = NullOrder::FIRST)
            : sort_keys(std::move(sort_keys)), null_order(std::move(null_order)) {}

    std::vector<SortKey> sort_keys;
    NullOrder null_order;

    std::string to_string() const override {
        std::stringstream ss;
        ss << "OrderByProperties {"
           << "sort_keys: [";
        for (const auto& key : sort_keys) {
            ss << key.field.ToString() << " " << static_cast<int>(key.order) << ", ";
        }
        ss << "], null_order: " << static_cast<int>(null_order) << "}";
        return ss.str();
    }
};

class GroupByProperties : public NodeProperties {
public:
    explicit GroupByProperties(std::vector<arrow::FieldRef> group_keys,
                               std::vector<std::shared_ptr<Aggregate>> aggregates)
            : group_keys(std::move(group_keys)), aggregates(std::move(aggregates)) {}

    explicit GroupByProperties(std::vector<std::string> group_keys_string,
                               std::vector<std::shared_ptr<Aggregate>> aggregates)
            : aggregates(std::move(aggregates)) {
        std::vector<arrow::FieldRef> temp_group_keys;
        for (auto& gk : group_keys_string) {
            temp_group_keys.push_back(arrow::FieldRef(gk));
        }
        group_keys = temp_group_keys;
    }

    std::vector<arrow::FieldRef> group_keys;
    std::vector<std::shared_ptr<Aggregate>> aggregates;

    std::string to_string() const override {
        std::stringstream ss;
        ss << "GroupByProperties {"
           << "group_keys: [";
        for (const auto& key : group_keys) {
            ss << key.ToString() << ", ";
        }
        ss << "], aggregate_expressions: [";
        for (const auto& aggr : aggregates) {
            ss << aggr->to_string() << ", ";
        }
        ss << "]}";
        return ss.str();
    }
};

class LimitProperties : public NodeProperties {
public:
    LimitProperties(int64_t limit, int64_t offset): limit(limit), offset(offset) {}

    int64_t limit  = 0;
    int64_t offset = 0;

    std::string to_string() const override {
        std::stringstream ss;
        ss << "LimitProperties {"
           << "  limit: " << limit << "\n  offset: " << offset << "\n}";
        return ss.str();
    }
};

class DistinctProperties : public NodeProperties {
public:
    DistinctProperties(std::vector<arrow::FieldRef> distinct_keys = {})
            : distinct_keys(std::move(distinct_keys)) {}

    std::vector<arrow::FieldRef> distinct_keys;

    std::string to_string() const override {
        std::stringstream ss;
        ss << "DistinctProperties {"
           << "distinct_keys: [";
        for (const auto& key : distinct_keys) {
            ss << key.ToString() << ", ";
        }
        ss << "]}";
        return ss.str();
    }
};

class TakeProperties : public NodeProperties {
public:
    TakeProperties(std::string data_key, std::string index_key)
            : data_key(std::move(data_key)), index_key(std::move(index_key)) {}

    std::string data_key;
    std::string index_key;

    [[nodiscard]] std::string to_string() const override {
        std::stringstream ss;
        ss << "TakeProperties { data_key: " << data_key
           << ", index_key: " << index_key << " }";
        return ss.str();
    }
};

class FusedProperties : public NodeProperties {
public:
    FusedProperties(std::vector<std::shared_ptr<NodeProperties>> properties,
                    std::vector<NodeType> node_types)
            : properties(std::move(properties)), node_types(std::move(node_types)) {}

    std::vector<std::shared_ptr<NodeProperties>> properties;
    std::vector<NodeType> node_types;

    std::string to_string() const override {
        std::stringstream ss;
        ss << "FusedProperties (\n";
        for (std::size_t i = 0; i < node_types.size(); ++i) {
            ss << node_type_to_string(node_types[i]);
            if (i < node_types.size() - 1) {
                ss << " + ";
            }
        }
        ss << ")\n";
        for (std::size_t i = 0; i < properties.size(); ++i) {
            ss << "  - " << node_type_to_string(node_types[i]) << ": " << properties[i]->to_string()
               << "\n";
        }
        return ss.str();
    }
};

class VectorJoinProperties : public NodeProperties {
protected:
    VectorJoinProperties(arrow::FieldRef data_vector_column,
                         arrow::FieldRef query_vector_column,
                         std::optional<int64_t> K                     = std::nullopt,
                         std::optional<float> radius                  = std::nullopt,
                         bool keep_data_vector_column                = false,
                         bool keep_query_vector_column                 = false,
                         std::optional<std::string> distance_column   = std::nullopt,
                         std::optional<arrow::FieldRef> filter_bitmap = std::nullopt)
            : data_vector_column(std::move(data_vector_column))
            , query_vector_column(std::move(query_vector_column))
            , K(K)
            , radius(radius)
            , keep_data_vector_column(keep_data_vector_column)
            , keep_query_vector_column(keep_query_vector_column)
            , distance_column(distance_column)
            , filter_bitmap(filter_bitmap) {
        assert(K.has_value() ^ radius.has_value());
    }

public:
    std::optional<int64_t> K;
    std::optional<float> radius;
    arrow::FieldRef data_vector_column;
    arrow::FieldRef query_vector_column;
    bool keep_data_vector_column = false;
    bool keep_query_vector_column = false;
    std::optional<std::string> distance_column;
    std::optional<arrow::FieldRef> filter_bitmap;
};


class VectorJoinIndexedProperties : public VectorJoinProperties {
public:
    VectorJoinIndexedProperties(arrow::FieldRef data_vector_column,
                                arrow::FieldRef query_vector_column,
                                IndexPtr index,
                                std::optional<int64_t> K                          = std::nullopt,
                                std::optional<float> radius                       = std::nullopt,
                                std::shared_ptr<IndexParameters> index_parameters = nullptr,
                                bool keep_data_vector_column                     = false,
                                bool keep_query_vector_column                      = false,
                                std::optional<std::string> distance_column        = std::nullopt,
                                std::shared_ptr<Expression> filter_expr           = nullptr,
                                std::optional<arrow::FieldRef> filter_bitmap      = std::nullopt)

            : VectorJoinProperties(std::move(data_vector_column),
                   std::move(query_vector_column),
                                   K,
                                   radius,
                                   keep_data_vector_column,
                                   keep_query_vector_column,
                                   distance_column,
                                   filter_bitmap)
            , filter_expr(filter_expr)
            , index(std::move(index))
            , index_parameters(std::move(index_parameters)) {
        assert(!((filter_expr != nullptr) && filter_bitmap.has_value()));
    }
    std::shared_ptr<Expression> filter_expr;
    IndexPtr index;
    std::shared_ptr<IndexParameters> index_parameters;

    std::string to_string() const override {
        std::stringstream ss;
        ss << "VectorJoinIndexedProperties {"
           << "  K: " << K.value_or(-1) << "\n"
           << "  Radius: " << radius.value_or(-1) << "\n"
           << "  Data Column: " << data_vector_column.ToString()
           << "  Query Column: " << query_vector_column.ToString()
           << " (Keep data vector =" << keep_data_vector_column << ")\n"
           << " (Keep query vector =" << keep_query_vector_column << ")\n"
           << "  Filter Expression: " + (filter_expr ? filter_expr->to_string() : "None") << "\n"
           << "  Filter Bitmap Column: " + (filter_bitmap ? *filter_bitmap->name() : "None") << "\n"
           << "  Index: " + (index ? index->to_string() : "None") << "\n"
           << "  Dist. Column Projection: " + distance_column.value_or("None") << "\n"
           << ")\n}";
        return ss.str();
    }
};

class VectorJoinExhaustiveProperties : public VectorJoinProperties {
public:
    VectorJoinExhaustiveProperties(arrow::FieldRef data_vector_column,
                                   arrow::FieldRef query_vector_column,
                                   std::optional<int64_t> K      = std::nullopt,
                                   std::optional<float> radius   = std::nullopt,
                                   VectorDistanceMetric metric   = VectorDistanceMetric::L2,
                                   bool keep_data_vector_column = false,
                                   bool keep_query_vector_column  = false,
                                   std::optional<std::string> distance_column   = std::nullopt,
                                   std::optional<arrow::FieldRef> filter_bitmap = std::nullopt)
            : VectorJoinProperties(std::move(data_vector_column),
                    std::move(query_vector_column),
                                   K,
                                   radius,
                                   keep_data_vector_column,
                                   keep_query_vector_column,
                                   distance_column,
                                   filter_bitmap)
            , metric(metric) {}

    VectorDistanceMetric metric;

    std::string to_string() const override {
        std::stringstream ss;
        std::string metric_str =
            (metric == VectorDistanceMetric::INNER_PRODUCT) ? "INNER_PRODUCT" : "L2";
        ss << "VectorJoinExhaustiveProperties {"
           << "  K: " << K.value_or(-1) << "\n"
           << "  Radius: " << radius.value_or(-1) << "\n"
           << "  Data Column: " << data_vector_column.ToString()
           << "  Query Column: " << query_vector_column.ToString()
           << " (Keep data vector =" << keep_data_vector_column << ")\n"
           << " (Keep query vector =" << keep_query_vector_column << ")\n"
           << "  Filter Bitmap Column: " + (filter_bitmap ? *filter_bitmap->name() : "None") << "\n"
           << "  Dist. Metric: " + metric_str << "\n"
           << "  Dist. Column Projection: " + distance_column.value_or("None") << "\n"
           << ")\n}";
        return ss.str();
    }
};


class VectorProjectDistanceProperties : public NodeProperties {
public:
    VectorProjectDistanceProperties(std::string distance_column_name,
                                    arrow::FieldRef left_vector_column,
                                    arrow::FieldRef right_vector_column,
                                    bool keep_left_vector_column  = false,
                                    bool keep_right_vector_column = false)
            : distance_column_name(std::move(distance_column_name))
            , left_vector_column(std::move(left_vector_column))
            , right_vector_column(std::move(right_vector_column))
            , keep_left_vector_column(keep_left_vector_column)
            , keep_right_vector_column(keep_right_vector_column) {}
    std::string distance_column_name;
    arrow::FieldRef left_vector_column;
    arrow::FieldRef right_vector_column;
    bool keep_left_vector_column;
    bool keep_right_vector_column;

    std::string to_string() const override {
        std::stringstream ss;
        ss << "VectorProjectDistanceProperties {"
           << "  Distance Column Name: " << distance_column_name << "\n"
           << "  Left Vector Column: " << left_vector_column.ToString()
           << " (Keep left vector =" << keep_left_vector_column << ")\n"
           << "  Right Vector Column: " << right_vector_column.ToString()
           << " (Keep right vector =" << keep_right_vector_column << ")\n"
           << "  Metric: L2 \n}";
        return ss.str();
    }
};

class ScatterProperties : public NodeProperties {
public:
    ScatterProperties(std::vector<arrow::FieldRef> partition_keys, int num_partitions = 1)
            : partition_keys(std::move(partition_keys)), num_partitions(num_partitions) {}

    ScatterProperties(std::vector<std::string> partition_key_names, int num_partitions = 1)
            : num_partitions(num_partitions) {
        for (const auto& name : partition_key_names) {
            partition_keys.push_back(arrow::FieldRef(name));
        }
    }

    std::vector<arrow::FieldRef> partition_keys;
    int num_partitions = 1;  // Number of output ports (partitions)

    std::string to_string() const override {
        std::stringstream ss;
        ss << "ScatterProperties { partition_keys: [";
        for (const auto& key : partition_keys) {
            ss << key.ToString() << ", ";
        }
        ss << "], num_partitions: " << num_partitions << " }";
        return ss.str();
    }
};

class GatherProperties : public NodeProperties {
public:
    explicit GatherProperties(int num_inputs = 0) : num_inputs(num_inputs) {}

    int num_inputs = 0;

    std::string to_string() const override {
        std::stringstream ss;
        ss << "GatherProperties { num_inputs: " << num_inputs << " }";
        return ss.str();
    }
};

class LimitPerGroupProperties : public NodeProperties {
public:
    LimitPerGroupProperties(std::string group_key, int64_t limit_k)
            : group_key(std::move(group_key)), limit_k(limit_k) {}

    std::string group_key;
    int64_t limit_k = 0;

    std::string to_string() const override {
        std::stringstream ss;
        ss << "LimitPerGroupProperties { group_key: " << group_key
           << ", limit_k: " << limit_k << " }";
        return ss.str();
    }
};

}  // namespace maximus
