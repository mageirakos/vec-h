#pragma once

#include <cstdint>
#include <string>

namespace maximus {
enum class PhysicalOperatorType : uint8_t {
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
    LOCAL_BROADCAST,
#ifdef MAXIMUS_WITH_VS
    VECTOR_JOIN,
    VECTOR_PROJECT_DISTANCE,
#endif
    SCATTER,
    GATHER,
    LIMIT_PER_GROUP,
    TAKE,
    UNDEFINED
};

std::string physical_operator_to_string(PhysicalOperatorType type);
}  // namespace maximus
