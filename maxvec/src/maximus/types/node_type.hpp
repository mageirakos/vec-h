#pragma once

#include <cstdint>
#include <maximus/types/physical_operator_type.hpp>
#include <string>

namespace maximus {
enum class NodeType : uint8_t {
    FILTER,
    PROJECT,
    HASH_JOIN,
    ORDER_BY,
    GROUP_BY,
    RANDOM_TABLE_SOURCE,
    TABLE_SOURCE,
#ifdef MAXIMUS_WITH_DATASET_API
    TABLE_SOURCE_FILTER_PROJECT,
#endif
    TABLE_SINK,
    LIMIT,
    DISTINCT,
    FUSED,
#ifdef MAXIMUS_WITH_VS
    VECTOR_JOIN,
    VECTOR_PROJECT_DISTANCE,
#endif
    LOCAL_BROADCAST,
    // TODO: SCATTER requires knowledge of how many partitions the query will have so that the DAG can get scheduled correctly. Is that okay or do we want to do it differently?
    SCATTER,
    GATHER,
    LIMIT_PER_GROUP,
    TAKE,
    QUERY_PLAN_ROOT,
    UNDEFINED,
};

std::string node_type_to_string(NodeType type);
PhysicalOperatorType node_type_to_operator_type(NodeType type);

}  // namespace maximus
