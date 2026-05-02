#include <maximus/types/node_type.hpp>

std::string maximus::node_type_to_string(NodeType type) {
    switch (type) {
        case NodeType::FILTER:
            return "FILTER";
        case NodeType::PROJECT:
            return "PROJECT";
        case NodeType::HASH_JOIN:
            return "HASH JOIN";
        case NodeType::ORDER_BY:
            return "ORDER BY";
        case NodeType::GROUP_BY:
            return "GROUP BY";
        case NodeType::RANDOM_TABLE_SOURCE:
            return "RANDOM TABLE SOURCE";
        case NodeType::TABLE_SOURCE:
            return "TABLE SOURCE";
#ifdef MAXIMUS_WITH_DATASET_API
        case NodeType::TABLE_SOURCE_FILTER_PROJECT:
            return "TABLE SOURCE FILTER PROJECT";
#endif
        case NodeType::TABLE_SINK:
            return "TABLE SINK";
        case NodeType::LIMIT:
            return "LIMIT";
        case NodeType::DISTINCT:
            return "DISTINCT";
        case NodeType::FUSED:
            return "FUSED";
#ifdef MAXIMUS_WITH_VS
        case NodeType::VECTOR_JOIN:
            return "VECTOR_JOIN";
        case NodeType::VECTOR_PROJECT_DISTANCE:
            return "VECTOR_PROJECT_DISTANCE";
#endif
        case NodeType::LOCAL_BROADCAST:
            return "LOCAL BROADCAST";
        case NodeType::SCATTER:
            return "SCATTER";
        case NodeType::GATHER:
            return "GATHER";
        case NodeType::LIMIT_PER_GROUP:
            return "LIMIT_PER_GROUP";
        case NodeType::TAKE:
            return "TAKE";
        case NodeType::QUERY_PLAN_ROOT:
            return "QUERY PLAN ROOT";
        default:
            return "NOT SUPPORTED";
    }
}

maximus::PhysicalOperatorType maximus::node_type_to_operator_type(NodeType type) {
    switch (type) {
        case NodeType::FILTER:
            return PhysicalOperatorType::FILTER;
        case NodeType::PROJECT:
            return PhysicalOperatorType::PROJECT;
        case NodeType::HASH_JOIN:
            return PhysicalOperatorType::HASH_JOIN;
        case NodeType::ORDER_BY:
            return PhysicalOperatorType::ORDER_BY;
        case NodeType::GROUP_BY:
            return PhysicalOperatorType::GROUP_BY;
        case NodeType::RANDOM_TABLE_SOURCE:
            return PhysicalOperatorType::RANDOM_TABLE_SOURCE;
        case NodeType::TABLE_SOURCE:
            return PhysicalOperatorType::TABLE_SOURCE;
#ifdef MAXIMUS_WITH_DATASET_API
        case NodeType::TABLE_SOURCE_FILTER_PROJECT:
            return PhysicalOperatorType::TABLE_SOURCE_FILTER_PROJECT;
#endif
        case NodeType::TABLE_SINK:
            return PhysicalOperatorType::TABLE_SINK;
        case NodeType::LIMIT:
            return PhysicalOperatorType::LIMIT;
        case NodeType::DISTINCT:
            return PhysicalOperatorType::DISTINCT;
        case NodeType::FUSED:
            return PhysicalOperatorType::FUSED;
#ifdef MAXIMUS_WITH_VS
        case NodeType::VECTOR_JOIN:
            return PhysicalOperatorType::VECTOR_JOIN;
        case NodeType::VECTOR_PROJECT_DISTANCE:
            return PhysicalOperatorType::VECTOR_PROJECT_DISTANCE;
#endif
        case NodeType::QUERY_PLAN_ROOT:
            return PhysicalOperatorType::UNDEFINED;
        case NodeType::SCATTER:
            return PhysicalOperatorType::SCATTER;
        case NodeType::GATHER:
            return PhysicalOperatorType::GATHER;
        case NodeType::LIMIT_PER_GROUP:
            return PhysicalOperatorType::LIMIT_PER_GROUP;
        case NodeType::TAKE:
            return PhysicalOperatorType::TAKE;
        default:
            return PhysicalOperatorType::UNDEFINED;
    }
}
