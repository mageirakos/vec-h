#pragma once

#include <maximus/gpu/gtable/cuda/cuda_buffer.hpp>
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
    arrow::Status memcpy_host_to_device(const std::shared_ptr<arrow::Buffer> &host_buf,
                                        int64_t nbytes,
                                        int64_t offset,
                                        std::shared_ptr<GBuffer> &device_buf);

    /**
     * To copy memory from the host to the device
     */
    arrow::Status memcpy_host_to_device(uint8_t *host_buf,
                                        int64_t nbytes,
                                        std::shared_ptr<GBuffer> &device_buf);

    /**
     * To copy a bitmask from the host to the device
     */
    arrow::Status memcpy_masks_host_to_device(const std::shared_ptr<arrow::Buffer> &host_buf,
                                              int64_t nelements,
                                              int64_t offset,
                                              std::shared_ptr<GBuffer> &device_buf);

    /**
     * Transform buffer from masks to bools
     */
    arrow::Status transform_mask_to_bools(std::shared_ptr<GBuffer> &device_buf_masks,
                                          int64_t nbytes,
                                          std::shared_ptr<GBuffer> &device_buf_bools);

    /**
     * To copy memory from the device to the host
     */
    arrow::Status memcpy_device_to_host(std::shared_ptr<GBuffer> &device_buf,
                                        const int64_t nbytes,
                                        std::shared_ptr<arrow::Buffer> &host_buf,
                                        arrow::MemoryPool *pool = arrow::default_memory_pool());

    /**
     * To copy memory from the device to the host
     */
    arrow::Status memcpy_device_to_host(std::shared_ptr<GBuffer> &device_buf,
                                        const int64_t nbytes,
                                        uint8_t **host_buf,
                                        arrow::MemoryPool *pool = arrow::default_memory_pool());

    /**
     * Transform buffer from bools to masks
     */
    arrow::Status transform_bools_to_mask(std::shared_ptr<GBuffer> &device_buf_bools,
                                          int64_t nbytes,
                                          std::shared_ptr<GBuffer> &device_buf_masks);

    std::shared_ptr<rmm::cuda_device_id> device_id;

    /**
     * To get the current cuda device
     */
    int get_current_cuda_device();

    /**
     * To get the device id
     */
    std::shared_ptr<rmm::cuda_device_id> get_device_id();

    /**
     * To get the available and total device memory in bytes for the current
     * device
     */
    std::pair<uint64_t, uint64_t> get_available_device_memory();

private:
    /**
     * To get a C pointer to the memory resource
     */
    rmm::mr::device_memory_resource *get_memory_resource_ptr();

    // /**
    //  * To allocate memory on the device
    //  */
    // arrow::Status
    // allocate_memory(int bytes, std::shared_ptr<arrow::cuda::CudaBuffer>
    // &out);

    // /**
    //  * To free memory on the device
    //  */
    // arrow::Status free_memory(std::shared_ptr<arrow::cuda::CudaBuffer> &out);
};

using GlobalCudaContext = MaximusCudaContext *;
using CudaContext       = std::unique_ptr<MaximusCudaContext>;

CudaContext make_cuda_context();

}  // namespace gpu
}  // namespace maximus
