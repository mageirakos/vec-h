#include <maximus/gpu/gtable/cuda/cuda_buffer.hpp>

namespace maximus {

namespace gpu {

CudaBuffer::CudaBuffer(std::shared_ptr<rmm::device_buffer> buf, int64_t size)
        : buf_(std::move(buf)) {
    sz_ = size;
}

std::shared_ptr<rmm::device_buffer> &CudaBuffer::get_buffer() {
    return buf_;
}

uint64_t CudaBuffer::get_size() {
    return sz_;
}

std::shared_ptr<GBuffer> CudaBuffer::clone() {
    std::shared_ptr<rmm::device_buffer> cloned_buf_ =
        std::make_shared<rmm::device_buffer>(*buf_, rmm::cuda_stream_default);
    return std::make_shared<CudaBuffer>(cloned_buf_, sz_);
}

}  // namespace gpu
}  // namespace maximus
