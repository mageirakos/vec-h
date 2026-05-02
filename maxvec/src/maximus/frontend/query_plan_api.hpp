#pragma once

#include <maximus/dag/query_node.hpp>
#include <maximus/database.hpp>
#include <maximus/indexes/index.hpp>
#include <maximus/operators/properties.hpp>
#include <maximus/types/device_table_ptr.hpp>
#include <maximus/types/expression.hpp>
#include <string>

namespace maximus {
namespace cp = ::arrow::compute;
namespace ac = ::arrow::acero;

std::shared_ptr<QueryNode> table_source(std::shared_ptr<Database>& db,
                                        const std::string& table_name,
                                        const std::shared_ptr<Schema> schema         = nullptr,
                                        const std::vector<std::string>& column_names = {},
                                        DeviceType device   = DeviceType::CPU,
                                        bool nocopy_variant = false);

#ifdef MAXIMUS_WITH_DATASET_API
std::shared_ptr<QueryNode> table_source_filter_project(
    std::shared_ptr<Database>& db,
    const std::string& table_name,
    const std::shared_ptr<Schema> schema,
    const std::vector<std::string>& column_names,
    const std::shared_ptr<Expression>& filter_expr,
    const std::vector<std::shared_ptr<Expression>> exprs,
    std::vector<std::string> project_column_names,
    DeviceType device);
#endif

std::shared_ptr<QueryNode> filter(const std::shared_ptr<QueryNode>& input_node,
                                  const std::shared_ptr<Expression>& filter_expr,
                                  DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryNode> distinct(const std::shared_ptr<QueryNode>& input_node,
                                    const std::vector<std::string>& column_names,
                                    DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryNode> project(const std::shared_ptr<QueryNode>& input_node,
                                   const std::vector<std::shared_ptr<Expression>> exprs,
                                   std::vector<std::string> column_names = {},
                                   DeviceType device                     = DeviceType::CPU);

std::shared_ptr<QueryNode> project(const std::shared_ptr<QueryNode>& input_node,
                                   std::vector<std::string> column_names,
                                   DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryNode> rename(const std::shared_ptr<QueryNode>& input_node,
                                  const std::vector<std::string>& old_column_names,
                                  const std::vector<std::string>& new_column_names,
                                  DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryNode> group_by(const std::shared_ptr<QueryNode>& input_node,
                                    const std::vector<std::string>& group_by_keys,
                                    const std::vector<std::shared_ptr<Aggregate>>& aggregates,
                                    DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryNode> order_by(const std::shared_ptr<QueryNode>& input_node,
                                    const std::vector<SortKey>& sort_keys,
                                    DeviceType device = DeviceType::CPU,
                                    NullOrder null_order = NullOrder::FIRST);

std::shared_ptr<QueryNode> join(const JoinType& join_type,
                                const std::shared_ptr<QueryNode>& left_node,
                                const std::shared_ptr<QueryNode>& right_node,
                                const std::vector<std::string>& left_keys,
                                const std::vector<std::string>& right_keys,
                                const std::string& left_suffix  = "",
                                const std::string& right_suffix = "",
                                DeviceType device               = DeviceType::CPU);

std::shared_ptr<QueryNode> inner_join(const std::shared_ptr<QueryNode>& left_node,
                                      const std::shared_ptr<QueryNode>& right_node,
                                      const std::vector<std::string>& left_keys,
                                      const std::vector<std::string>& right_keys,
                                      const std::string& left_suffix  = "",
                                      const std::string& right_suffix = "",
                                      DeviceType device               = DeviceType::CPU);

std::shared_ptr<QueryNode> cross_join(const std::shared_ptr<QueryNode>& left_node,
                                      const std::shared_ptr<QueryNode>& right_node,
                                      const std::vector<std::string>& left_columns,
                                      const std::vector<std::string>& right_columns,
                                      const std::string& left_suffix  = "",
                                      const std::string& right_suffix = "",
                                      DeviceType device               = DeviceType::CPU);

std::shared_ptr<QueryNode> left_semi_join(const std::shared_ptr<QueryNode>& left_node,
                                          const std::shared_ptr<QueryNode>& right_node,
                                          const std::vector<std::string>& left_keys,
                                          const std::vector<std::string>& right_keys,
                                          const std::string& left_suffix  = "",
                                          const std::string& right_suffix = "",
                                          DeviceType device               = DeviceType::CPU);

std::shared_ptr<QueryNode> left_anti_join(const std::shared_ptr<QueryNode>& left_node,
                                          const std::shared_ptr<QueryNode>& right_node,
                                          const std::vector<std::string>& left_keys,
                                          const std::vector<std::string>& right_keys,
                                          const std::string& left_suffix  = "",
                                          const std::string& right_suffix = "",
                                          DeviceType device               = DeviceType::CPU);

std::shared_ptr<QueryNode> left_outer_join(const std::shared_ptr<QueryNode>& left_node,
                                           const std::shared_ptr<QueryNode>& right_node,
                                           const std::vector<std::string>& left_keys,
                                           const std::vector<std::string>& right_keys,
                                           const std::string& left_suffix  = "",
                                           const std::string& right_suffix = "",
                                           DeviceType device               = DeviceType::CPU);

std::shared_ptr<QueryNode> table_sink(const std::shared_ptr<QueryNode>& input_node);

std::shared_ptr<QueryNode> take(const std::shared_ptr<QueryNode>& data_node,
                                const std::shared_ptr<QueryNode>& index_node,
                                const std::string& data_key,
                                const std::string& index_key,
                                DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryNode> limit(const std::shared_ptr<QueryNode>& input_node,
                                 int64_t limit,
                                 int64_t offset    = 0,
                                 DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryNode> limit_per_group(const std::shared_ptr<QueryNode>& input_node,
                                           const std::string& group_key,
                                           int64_t limit_k,
                                           DeviceType device = DeviceType::CPU);

#ifdef MAXIMUS_WITH_VS
std::shared_ptr<QueryNode> vector_project_distance(
    const std::shared_ptr<QueryNode>& data_node,
    const std::shared_ptr<QueryNode>& query_node,
    const std::string& data_vector_column,
    const std::string& query_vector_column,
    const bool keep_left_vector_column         = false,
    const bool keep_right_vector_column        = false,
    std::optional<std::string> distance_column = std::nullopt,
    DeviceType device                          = DeviceType::CPU);

std::shared_ptr<QueryNode> exhaustive_vector_join(
    const std::shared_ptr<QueryNode>& data_node,
    const std::shared_ptr<QueryNode>& query_node,
    const std::string& data_vector_column,
    const std::string& query_vector_column,
    VectorDistanceMetric metric                  = VectorDistanceMetric::L2,
    std::optional<int64_t> K                     = std::nullopt,
    std::optional<float> radius                  = std::nullopt,
    bool keep_data_vector_column                 = false,
    bool keep_query_vector_column                = false,
    std::optional<std::string> distance_column   = std::nullopt,
    DeviceType device                            = DeviceType::CPU,
    std::optional<arrow::FieldRef> filter_bitmap = std::nullopt);

std::shared_ptr<QueryNode> indexed_vector_join(
    const std::shared_ptr<QueryNode>& data_node,
    const std::shared_ptr<QueryNode>& query_node,
    const std::string& data_vector_column,
    const std::string& query_vector_column,
    IndexPtr index,
    std::optional<int64_t> K,
    std::optional<float> radius,
    std::shared_ptr<IndexParameters> index_parameters = nullptr,
    bool keep_data_vector_column                      = false,
    bool keep_query_vector_column                     = false,
    std::optional<std::string> distance_column        = std::nullopt,
    DeviceType device                                 = DeviceType::CPU,
    std::shared_ptr<Expression> filter_expr           = nullptr,
    std::optional<arrow::FieldRef> filter_bitmap      = std::nullopt);
#endif
std::shared_ptr<QueryPlan> query_plan(const std::shared_ptr<QueryNode>& sink_node);

// Scatter and Gather operators for scatter-gather pattern
std::shared_ptr<QueryNode> scatter(const std::shared_ptr<QueryNode>& input_node,
                                   const std::vector<std::string>& partition_keys,
                                   int num_partitions,
                                   DeviceType device = DeviceType::CPU);

std::shared_ptr<QueryNode> gather(const std::vector<std::shared_ptr<QueryNode>>& input_nodes,
                                  DeviceType device = DeviceType::CPU);

// Helper function to build scatter -> apply -> gather pattern for VSDS queries
// Creates: scatter(N) -> [order_by (optional) -> limit(k) x N] -> gather
std::shared_ptr<QueryNode> order_limit_per_partition(
    const std::shared_ptr<QueryNode>& input_node,
    const std::string& partition_key,
    int num_partitions,
    int64_t limit_k,
    DeviceType device = DeviceType::CPU,
    std::vector<SortKey> sort_keys = {});  // Optional: order by these keys before limiting



}  // namespace maximus
