#pragma once

#include <cstdint>
#include <maximus/types/types.hpp>
#include <memory>

namespace maximus {

namespace gpu {

// Abstract class for scalar
class GScalar {
public:
    GScalar() = default;

    /**
     * To check if the scalar is valid
     */
    virtual bool is_valid() = 0;

    /**
     * To get the data type of the scalar
     */
    virtual std::shared_ptr<DataType> get_data_type() = 0;
};

}  // namespace gpu
}  // namespace maximus
