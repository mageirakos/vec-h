#include <maximus/gpu/cudf/cudf_types.hpp>
#include <maximus/gpu/gtable/cuda/cuda_scalar.hpp>

namespace maximus {

namespace gpu {

CudaScalar::CudaScalar(std::shared_ptr<cudf::scalar> &sc): sc_(std::move(sc)) {
}

std::shared_ptr<cudf::scalar> &CudaScalar::get_scalar() {
    return sc_;
}

std::shared_ptr<DataType> CudaScalar::get_data_type() {
    return to_maximus_type(sc_->type());
}

}  // namespace gpu
}  // namespace maximus
