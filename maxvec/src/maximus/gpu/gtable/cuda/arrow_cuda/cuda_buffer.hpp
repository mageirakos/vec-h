#pragma once
#include <arrow/gpu/cuda_api.h>

#include <maximus/gpu/gtable/gbuffer.hpp>

namespace maximus {

namespace gpu {

class CudaBuffer : public GBuffer {
public:
    CudaBuffer(std::shared_ptr<arrow::cuda::CudaBuffer> &buf, int64_t size);

    /**
     * To get the buffer
     */
    std::shared_ptr<arrow::cuda::CudaBuffer> &get_buffer();

    /**
     * To get a sliced portion of the buffer
     */
    void get_sliced_buffer(int64_t offset, int64_t length, std::shared_ptr<GBuffer> &out);

    /**
     * To get the size of the buffer
     */
    uint64_t get_size();

private:
    std::shared_ptr<arrow::cuda::CudaBuffer> buf_;
};

}  // namespace gpu
}  // namespace maximus
