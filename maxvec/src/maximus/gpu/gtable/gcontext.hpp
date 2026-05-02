#pragma once

#include <maximus/gpu/gtable/gbuffer.hpp>
#include <maximus/types/types.hpp>

namespace maximus {

namespace gpu {

// Abstract class for context
class MaximusGContext {
public:
    MaximusGContext() = default;

    /**
     * To set the device context
     */
    virtual arrow::Status set_device_context(int device_id) = 0;

    /**
     * To copy memory from the host to the device
     */
    virtual arrow::Status memcpy_host_to_device(const std::shared_ptr<arrow::Buffer> &host_buf,
                                                int64_t nbytes,
                                                int64_t offset,
                                                std::shared_ptr<GBuffer> &device_buf) = 0;

    /**
     * To copy memory from the host to the device
     */
    virtual arrow::Status memcpy_host_to_device(uint8_t *host_buf,
                                                int64_t nbytes,
                                                std::shared_ptr<GBuffer> &device_buf) = 0;

    /**
     * To copy a bitmask from the host to the device
     */
    virtual arrow::Status memcpy_masks_host_to_device(
        const std::shared_ptr<arrow::Buffer> &host_buf,
        int64_t nelements,
        int64_t offset,
        std::shared_ptr<GBuffer> &device_buf) = 0;

    /**
     * Transform buffer from masks to bools
     */
    virtual arrow::Status transform_mask_to_bools(std::shared_ptr<GBuffer> &device_buf_masks,
                                                  int64_t nbytes,
                                                  std::shared_ptr<GBuffer> &device_buf_bools) = 0;

    /**
     * To copy memory from the device to the host
     */
    virtual arrow::Status memcpy_device_to_host(
        std::shared_ptr<GBuffer> &device_buf,
        const int64_t nbytes,
        std::shared_ptr<arrow::Buffer> &host_buf,
        arrow::MemoryPool *pool = arrow::default_memory_pool()) = 0;

    /**
     * To copy memory from the device to the host
     */
    virtual arrow::Status memcpy_device_to_host(
        std::shared_ptr<GBuffer> &device_buf,
        const int64_t nbytes,
        uint8_t **host_buf,
        arrow::MemoryPool *pool = arrow::default_memory_pool()) = 0;

    /**
     * Transform buffer from bools to masks
     */
    virtual arrow::Status transform_bools_to_mask(std::shared_ptr<GBuffer> &device_buf_bools,
                                                  int64_t nbytes,
                                                  std::shared_ptr<GBuffer> &device_buf_masks) = 0;
};

using GlobalGContext = MaximusGContext *;
using GContext       = std::unique_ptr<MaximusGContext>;

}  // namespace gpu
}  // namespace maximus
