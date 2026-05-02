#include <maximus/types/cuda_event.hpp>

namespace maximus::gpu {
void check_runtime_status(cudaError_t status) {
    if (status != cudaSuccess) {
        const char* error = cudaGetErrorString(status);
        std::cerr << "error: GPU API call : " << error << std::endl;
        throw std::runtime_error(std::string("GPU ERROR: ") + error);
    }
}
}  // namespace maximus::gpu
