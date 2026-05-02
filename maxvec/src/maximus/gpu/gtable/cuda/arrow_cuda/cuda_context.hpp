#pragma once
#include <arrow/gpu/cuda_api.h>

#include <maximus/gpu/gtable/gcontext.hpp>

namespace maximus {

namespace gpu {

class MaximusCudaContext : public MaximusGContext {
public:
    MaximusCudaContext();

    /**
     * To set the device context
     */
    arrow::Status set_device_context(int device_id);

    /**
     * To copy memory from the host to the device
     */
    arrow::Status memcpy_host_to_device(const int64_t position,
                                        std::shared_ptr<arrow::Buffer> &host_buf,
                                        int64_t nbytes,
                                        std::shared_ptr<GBuffer> &device_buf);

    /**
     * To copy memory from the host to the device
     */
    arrow::Status memcpy_host_to_device(const int64_t position,
                                        uint8_t *host_buf,
                                        int64_t nbytes,
                                        std::shared_ptr<GBuffer> &device_buf);

    /**
     * To copy memory from the device to the host
     */
    arrow::Status memcpy_device_to_host(std::shared_ptr<GBuffer> &device_buf,
                                        const int64_t position,
                                        const int64_t nbytes,
                                        std::shared_ptr<arrow::Buffer> &host_buf,
                                        arrow::MemoryPool *pool = arrow::default_memory_pool());

    /**
     * To copy memory from the device to the host
     */
    arrow::Status memcpy_device_to_host(std::shared_ptr<GBuffer> &device_buf,
                                        const int64_t position,
                                        const int64_t nbytes,
                                        uint8_t **host_buf,
                                        arrow::MemoryPool *pool = arrow::default_memory_pool());

    std::shared_ptr<arrow::cuda::CudaContext> context = nullptr;
    std::shared_ptr<arrow::cuda::CudaDevice> device   = nullptr;

private:
    /**
     * To allocate memory on the device
     */
    arrow::Status allocate_memory(int bytes, std::shared_ptr<arrow::cuda::CudaBuffer> &out);

    /**
     * To free memory on the device
     */
    arrow::Status free_memory(std::shared_ptr<arrow::cuda::CudaBuffer> &out);
};

using GlobalCudaContext = MaximusCudaContext *;
using CudaContext       = std::unique_ptr<MaximusCudaContext>;

CudaContext make_cuda_context();

}  // namespace gpu
}  // namespace maximus
