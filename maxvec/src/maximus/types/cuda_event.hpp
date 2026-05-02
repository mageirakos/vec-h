#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

namespace maximus::gpu {
void check_runtime_status(cudaError_t status);

struct cuda_event {
    cuda_event() { check_runtime_status(cudaEventCreateWithFlags(&e_, cudaEventDisableTiming)); }
    virtual ~cuda_event() { check_runtime_status(cudaEventDestroy(e_)); }

    operator cudaEvent_t() { return e_; }

private:
    cudaEvent_t e_;
};
}  // namespace maximus::gpu
