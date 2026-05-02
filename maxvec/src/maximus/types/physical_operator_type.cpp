#include <maximus/types/physical_operator_type.hpp>

std::string maximus::physical_operator_to_string(PhysicalOperatorType type) {
    switch (type) {
        case PhysicalOperatorType::FILTER:
            return "FILTER";
        case PhysicalOperatorType::PROJECT:
            return "PROJECT";
        case PhysicalOperatorType::HASH_JOIN:
            return "HASH_JOIN";
        case PhysicalOperatorType::ORDER_BY:
            return "ORDER_BY";
        case PhysicalOperatorType::GROUP_BY:
            return "GROUP_BY";
        case PhysicalOperatorType::RANDOM_TABLE_SOURCE:
            return "RANDOM_TABLE_SOURCE";
        case PhysicalOperatorType::TABLE_SOURCE:
            return "TABLE_SOURCE";
#ifdef MAXIMUS_WITH_DATASET_API
        case PhysicalOperatorType::TABLE_SOURCE_FILTER_PROJECT:
            return "TABLE_SOURCE_FILTER_PROJECT";
#endif
        case PhysicalOperatorType::TABLE_SINK:
            return "TABLE_SINK";
        case PhysicalOperatorType::LIMIT:
            return "LIMIT";
        case PhysicalOperatorType::DISTINCT:
            return "DISTINCT";
        case PhysicalOperatorType::FUSED:
            return "FUSED";
        case PhysicalOperatorType::LOCAL_BROADCAST:
            return "LOCAL_BROADCAST";
#ifdef MAXIMUS_WITH_VS
        case PhysicalOperatorType::VECTOR_JOIN:
            return "VECTOR_SEARCH";
        case PhysicalOperatorType::VECTOR_PROJECT_DISTANCE:
            return "VECTOR_PROJECT_DISTANCE";
#endif
        case PhysicalOperatorType::SCATTER:
            return "SCATTER";
        case PhysicalOperatorType::GATHER:
            return "GATHER";
        case PhysicalOperatorType::LIMIT_PER_GROUP:
            return "LIMIT_PER_GROUP";
        case PhysicalOperatorType::TAKE:
            return "TAKE";
        case PhysicalOperatorType::UNDEFINED:
            return "UNDEFINED";
        default:
            return "NOT SUPPORTED";
    }
}
