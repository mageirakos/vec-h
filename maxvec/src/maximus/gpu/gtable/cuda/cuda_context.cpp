#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/transform.hpp>
#include <maximus/gpu/gtable/cuda/cuda_buffer.hpp>
#include <maximus/gpu/gtable/cuda/cuda_context.hpp>

namespace maximus {

namespace gpu {

MaximusCudaContext::MaximusCudaContext() {
    device_id = std::make_shared<rmm::cuda_device_id>(rmm::get_current_cuda_device());
}

arrow::Status MaximusCudaContext::set_device_context(int device_) {
    device_id = std::make_shared<rmm::cuda_device_id>(rmm::cuda_device_id(device_));
    return arrow::Status::OK();
}

arrow::Status MaximusCudaContext::memcpy_host_to_device(
    const std::shared_ptr<arrow::Buffer> &host_buf,
    int64_t nbytes,
    int64_t offset,
    std::shared_ptr<GBuffer> &device_buf) {
    // Transfer data from host to device
    std::shared_ptr<rmm::device_buffer> cuda_buf =
        std::make_shared<rmm::device_buffer>((const void *) (host_buf->data() + offset),
                                             nbytes,
                                             rmm::cuda_stream_default,
                                             get_memory_resource_ptr());
    device_buf = std::make_shared<CudaBuffer>(cuda_buf, nbytes);
    return arrow::Status::OK();
}

arrow::Status MaximusCudaContext::memcpy_masks_host_to_device(
    const std::shared_ptr<arrow::Buffer> &host_buf,
    int64_t nelements,
    int64_t offset,
    std::shared_ptr<GBuffer> &device_buf) {
    std::shared_ptr<rmm::device_buffer> mask =
        std::make_shared<rmm::device_buffer>((const void *) host_buf->data(),
                                             host_buf->size(),
                                             rmm::cuda_stream_default,
                                             get_memory_resource_ptr());

    assert((const uint32_t *) mask->data() != nullptr);

    std::shared_ptr<rmm::device_buffer> dev_buf =
        std::make_shared<rmm::device_buffer>(cudf::copy_bitmask((const uint32_t *) mask->data(),
                                                                offset,
                                                                nelements + offset,
                                                                rmm::cuda_stream_default,
                                                                get_memory_resource_ptr()));
    device_buf = std::make_shared<CudaBuffer>(dev_buf, dev_buf->size());
    return arrow::Status::OK();
}

arrow::Status MaximusCudaContext::transform_mask_to_bools(
    std::shared_ptr<GBuffer> &device_buf_masks,
    int64_t nbytes,
    std::shared_ptr<GBuffer> &device_buf_bools) {
    device_buf_bools = std::make_shared<CudaBuffer>(
        std::move(cudf::mask_to_bools(
                      (cudf::bitmask_type const *) device_buf_masks->data<void>(), 0, nbytes)
                      ->release()
                      .data),
        nbytes);
    return arrow::Status::OK();
}

arrow::Status MaximusCudaContext::memcpy_host_to_device(uint8_t *host_buf,
                                                        int64_t nbytes,
                                                        std::shared_ptr<GBuffer> &device_buf) {
    // Transfer data from host to device
    std::shared_ptr<rmm::device_buffer> cuda_buf = std::make_shared<rmm::device_buffer>(
        (const void *) host_buf, nbytes, rmm::cuda_stream_default, get_memory_resource_ptr());
    device_buf = std::make_shared<CudaBuffer>(cuda_buf, nbytes);
    return arrow::Status::OK();
}

arrow::Status MaximusCudaContext::memcpy_device_to_host(std::shared_ptr<GBuffer> &device_buf,
                                                        const int64_t nbytes,
                                                        std::shared_ptr<arrow::Buffer> &host_buf,
                                                        arrow::MemoryPool *pool) {
    std::shared_ptr<CudaBuffer> device_cuda_buf = std::static_pointer_cast<CudaBuffer>(device_buf);
    assert(device_cuda_buf != nullptr);
    uint8_t *out_ptr;

    // Allocate memory on the host
    arrow::Status status = pool->Allocate(nbytes, &out_ptr);
    if (!status.ok()) return status;

    // Copy from device to host
    cudaError_t copy_result =
        cudaMemcpyAsync(out_ptr, device_cuda_buf->data<void>(), nbytes, cudaMemcpyDeviceToHost);
    if (copy_result != cudaSuccess) {
        return arrow::Status::Invalid("Failed to copy data from device to host");
    }
    cudaStreamSynchronize(0);
    host_buf = std::make_shared<arrow::Buffer>((const uint8_t *) out_ptr, nbytes);
    return arrow::Status::OK();
}

arrow::Status MaximusCudaContext::memcpy_device_to_host(std::shared_ptr<GBuffer> &device_buf,
                                                        const int64_t nbytes,
                                                        uint8_t **host_buf,
                                                        arrow::MemoryPool *pool) {
    std::shared_ptr<CudaBuffer> device_cuda_buf = std::static_pointer_cast<CudaBuffer>(device_buf);
    assert(device_cuda_buf != nullptr);
    uint8_t *out_ptr;

    // Allocate memory on the host
    arrow::Status status = pool->Allocate(nbytes, host_buf);
    if (!status.ok()) return status;

    // Copy from device to host
    cudaError_t copy_result =
        cudaMemcpyAsync(*host_buf, device_cuda_buf->data<void>(), nbytes, cudaMemcpyDeviceToHost);
    if (copy_result != cudaSuccess) {
        return arrow::Status::Invalid("Failed to copy data from device to host");
    }
    cudaStreamSynchronize(0);
    return arrow::Status::OK();
}

arrow::Status MaximusCudaContext::transform_bools_to_mask(
    std::shared_ptr<GBuffer> &device_buf_bools,
    int64_t nbytes,
    std::shared_ptr<GBuffer> &device_buf_masks) {
    std::shared_ptr<rmm::device_buffer> new_buf = std::move(
        cudf::bools_to_mask(cudf::column_view(cudf::data_type(cudf::type_id::BOOL8),
                                              (int32_t) device_buf_bools->get_size(),
                                              (const void *) device_buf_bools->data<void>(),
                                              nullptr,
                                              0))
            .first);
    device_buf_masks = std::make_shared<CudaBuffer>(new_buf, new_buf->size());
    return arrow::Status::OK();
}

int MaximusCudaContext::get_current_cuda_device() {
    return device_id->value();
}

std::shared_ptr<rmm::cuda_device_id> MaximusCudaContext::get_device_id() {
    return device_id;
}

// std::shared_ptr<rmm::mr::device_memory_resource>
// MaximusCudaContext::get_memory_resource() {
//     return std::make_shared<rmm::mr::device_memory_resource>(
//         *rmm::mr::get_current_device_resource());
// }

std::pair<uint64_t, uint64_t> MaximusCudaContext::get_available_device_memory() {
    return rmm::available_device_memory();
}

rmm::mr::device_memory_resource *MaximusCudaContext::get_memory_resource_ptr() {
    return rmm::mr::get_current_device_resource();
}

CudaContext make_cuda_context() {
    return std::make_unique<MaximusCudaContext>();
}

}  // namespace gpu
}  // namespace maximus
