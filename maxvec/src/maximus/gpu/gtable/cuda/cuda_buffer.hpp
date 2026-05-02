#pragma once

#include <cuda_runtime_api.h>

#include <maximus/gpu/gtable/gbuffer.hpp>
#include <rmm/device_buffer.hpp>

namespace maximus {

namespace gpu {

class CudaBuffer : public GBuffer {
public:
    CudaBuffer(std::shared_ptr<rmm::device_buffer> buf, int64_t size);

    /**
     * To get the buffer
     */
    std::shared_ptr<rmm::device_buffer> &get_buffer();

    /**
     * To get the size of the buffer
     */
    uint64_t get_size();

    /**
     * To get pointer to data
     */
    template<typename T>
    T *data() {
        return static_cast<T *>(get_untyped());
    }

    /**
     * To offset each element of the buffer
     */
    void offset_buffer_by_value(int64_t offset, int offset_length = 4);

    /**
     * To clone the buffer
     */
    std::shared_ptr<GBuffer> clone();

private:
    std::shared_ptr<rmm::device_buffer> buf_;

    void *get_untyped() override { return buf_->data(); }
};

}  // namespace gpu
}  // namespace maximus
