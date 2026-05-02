#include <iostream>
#include <maximus/gpu/gtable/cuda/cuda_buffer.hpp>

namespace maximus {

namespace gpu {

template<typename offset_type>
__global__ void offset_buffer_by_value_kernel(offset_type *buf, offset_type offset, int32_t size) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        buf[idx] -= offset;
    }
}

void CudaBuffer::offset_buffer_by_value(int64_t offset, int offset_length) {
    int64_t el = sz_ / offset_length;
    assert(el * offset_length == sz_ && "not offset buffer");

    int64_t num_threads = std::min(el, (int64_t) 1024);
    int64_t num_blocks  = (el + num_threads - 1) / num_threads;

    if (offset_length == 4) {
        int32_t *ptr = (int32_t *) (buf_->data());
        offset_buffer_by_value_kernel<int32_t><<<num_blocks, num_threads>>>(ptr, offset, el);
    } else if (offset_length == 8) {
        int64_t *ptr = (int64_t *) (buf_->data());
        offset_buffer_by_value_kernel<int64_t><<<num_blocks, num_threads>>>(ptr, offset, el);
    } else {
        assert(false && "not support offset length");
    }
}

}  // namespace gpu

}  // namespace maximus