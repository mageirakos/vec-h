#include <maximus/operators/engine.hpp>
namespace maximus {

std::string engine_type_to_string(EngineType type) {
    // Convert EngineType to string representation if needed
    // Implement according to your specific EngineType enum
    // Return a string representation of the EngineType
    switch (type) {
        case EngineType::NATIVE:
            return "NATIVE";
        case EngineType::ACERO:
            return "ACERO";
        case EngineType::CUDF:
            return "CUDF";
        case EngineType::FAISS:
            return "FAISS";
        default:
            return "UNDEFINED";
    }
}

bool is_gpu_engine(const EngineType type) {
    return type == EngineType::CUDF || type == EngineType::FAISS;
}

bool is_cpu_engine(const EngineType type) {
    return type == EngineType::ACERO || type == EngineType::NATIVE || type == EngineType::FAISS;
}
}  // namespace maximus
