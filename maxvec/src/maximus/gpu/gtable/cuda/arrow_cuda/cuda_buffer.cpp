#include <maximus/arrow/arrow_types.hpp>
#include <maximus/gpu/gtable/cuda/cuda_buffer.hpp>

namespace maximus {

namespace gpu {

CudaBuffer::CudaBuffer(std::shared_ptr<arrow::cuda::CudaBuffer> buf, int64_t size): buf_(buf) {
    sz_ = size;
}

std::shared_ptr<arrow::cuda::CudaBuffer> &CudaBuffer::get_buffer() {
    return buf_;
}

uint64_t CudaBuffer::get_size() {
    return sz_;
}

void CudaBuffer::get_sliced_buffer(int64_t offset, int64_t length, std::shared_ptr<GBuffer> &out) {
    // Slice the buffer
    auto sliced_buf = std::dynamic_pointer_cast<arrow::cuda::CudaBuffer>(
        arrow::SliceBuffer(std::dynamic_pointer_cast<arrow::Buffer>(buf_), offset, length));
    out = std::make_shared<CudaBuffer>(sliced_buf, length);
}

}  // namespace gpu
}  // namespace maximus
