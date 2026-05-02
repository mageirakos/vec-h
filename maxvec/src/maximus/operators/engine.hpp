#pragma once

#include <maximus/context.hpp>

namespace maximus {

enum class EngineType : uint8_t { UNDEFINED, ACERO, CUDF, FAISS, NATIVE };

std::string engine_type_to_string(EngineType type);

bool is_gpu_engine(const EngineType type);

bool is_cpu_engine(const EngineType type);

class Engine {
public:
    Engine(EngineType type, std::shared_ptr<MaximusContext> ctx): type(type) {}

    EngineType kind() const { return type; }

    std::string to_string() const { return engine_type_to_string(type); }

    bool on_gpu() const { return type == EngineType::CUDF || type == EngineType::FAISS; }

    bool on_cpu() const {
        return type == EngineType::ACERO || type == EngineType::FAISS || type == EngineType::NATIVE;
    }

    std::shared_ptr<MaximusContext> ctx;
    EngineType type;
};


}  // namespace maximus
