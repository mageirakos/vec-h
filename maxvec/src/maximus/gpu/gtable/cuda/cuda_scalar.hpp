#pragma once

#include <cudf/scalar/scalar.hpp>
#include <maximus/gpu/gtable/gscalar.hpp>
#include <memory>

namespace maximus {

namespace gpu {

class CudaScalar : public GScalar {
public:
    CudaScalar(std::shared_ptr<cudf::scalar> &sc);

    /**
     * To get the scalar
     */
    std::shared_ptr<cudf::scalar> &get_scalar();

    /**
     * To get the data type of the scalar
     */
    std::shared_ptr<DataType> get_data_type();

private:
    std::shared_ptr<cudf::scalar> sc_;
};

}  // namespace gpu
}  // namespace maximus
