#pragma once

#include <maximus/context.hpp>
#include <maximus/types/types.hpp>
#include <maximus/operators/engine.hpp>
#include <maximus/types/device_table_ptr.hpp>
#include <string>
#include <vector>

namespace maximus {

class Index {
public:
    Index(std::shared_ptr<MaximusContext>& ctx): ctx(ctx) {}

    virtual ~Index() = default;

    virtual std::string to_string() = 0;

    /// Returns true if the index has been trained and is ready for search.
    virtual bool is_trained() const = 0;
    virtual VectorDistanceMetric metric() const = 0;

public:
    DeviceType device_type = DeviceType::UNDEFINED;
    EngineType engine_type = EngineType::UNDEFINED;
    
    bool is_on_gpu() const { return device_type == DeviceType::GPU; }
    bool is_on_cpu() const { return device_type == DeviceType::CPU; }

protected:
    std::shared_ptr<MaximusContext> ctx;
    
public:
    std::string description = "";
};

class IndexParameters {
public:
    IndexParameters()          = default;
    virtual ~IndexParameters() = default;
};

using IndexPtr = std::shared_ptr<Index>;

}  // namespace maximus