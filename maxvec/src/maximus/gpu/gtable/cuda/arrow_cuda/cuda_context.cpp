#include <maximus/gpu/gtable/cuda/cuda_buffer.hpp>
#include <maximus/gpu/gtable/cuda/cuda_context.hpp>
#include <typeinfo>

namespace maximus {

namespace gpu {

MaximusCudaContext::MaximusCudaContext() = default;

arrow::Status MaximusCudaContext::set_device_context(int device_id) {
    ARROW_ASSIGN_OR_RAISE(device, arrow::cuda::CudaDevice::Make(device_id));
    ARROW_ASSIGN_OR_RAISE(context, device->GetContext());
    return arrow::Status::OK();
}

arrow::Status MaximusCudaContext::allocate_memory(int bytes,
                                                  std::shared_ptr<arrow::cuda::CudaBuffer> &out) {
    ARROW_ASSIGN_OR_RAISE(auto buf, context->Allocate(bytes));
    out = std::shared_ptr<arrow::cuda::CudaBuffer>(std::move(buf));
    return arrow::Status::OK();
}

arrow::Status MaximusCudaContext::free_memory(std::shared_ptr<arrow::cuda::CudaBuffer> &out) {
    out.reset();
    return arrow::Status::OK();
}

arrow::Status MaximusCudaContext::memcpy_host_to_device(const int64_t position,
                                                        std::shared_ptr<arrow::Buffer> &host_buf,
                                                        int64_t nbytes,
                                                        std::shared_ptr<GBuffer> &device_buf) {
    std::shared_ptr<arrow::cuda::CudaBuffer> cuda_buf;
    // Allocate memory on the device
    auto status = allocate_memory(nbytes, cuda_buf);
    if (!status.ok()) return status;
    // Copy from host to device
    status = cuda_buf->CopyFromHost(position, host_buf->data(), nbytes);
    if (!status.ok()) return status;
    device_buf = std::make_shared<CudaBuffer>(cuda_buf, nbytes);
    return arrow::Status::OK();
}

arrow::Status MaximusCudaContext::memcpy_host_to_device(const int64_t position,
                                                        uint8_t *host_buf,
                                                        int64_t nbytes,
                                                        std::shared_ptr<GBuffer> &device_buf) {
    std::shared_ptr<arrow::cuda::CudaBuffer> cuda_buf;
    // Allocate memory on the device
    auto status = allocate_memory(nbytes, cuda_buf);
    if (!status.ok()) return status;
    // Copy from host to device
    status = cuda_buf->CopyFromHost(position, (const void *) host_buf, nbytes);
    if (!status.ok()) return status;
    device_buf = std::make_shared<CudaBuffer>(cuda_buf, nbytes);
    return arrow::Status::OK();
}

arrow::Status MaximusCudaContext::memcpy_device_to_host(std::shared_ptr<GBuffer> &device_buf,
                                                        const int64_t position,
                                                        const int64_t nbytes,
                                                        std::shared_ptr<arrow::Buffer> &host_buf,
                                                        arrow::MemoryPool *pool) {
    assert(typeid(*buf) == typeid(CudaBuffer));
    auto device_cuda_buf = std::static_pointer_cast<CudaBuffer>(device_buf);
    void *out_ptr;
    // Allocate memory on the host
    auto status = pool->Allocate(nbytes, (uint8_t **) (&out_ptr));
    if (!status.ok()) return status;
    // Copy from device to host
    status = device_cuda_buf->get_buffer()->CopyToHost(position, nbytes, out_ptr);
    if (!status.ok()) return status;
    host_buf = std::make_shared<arrow::Buffer>((const uint8_t *) out_ptr, nbytes);
    return arrow::Status::OK();
}

arrow::Status MaximusCudaContext::memcpy_device_to_host(std::shared_ptr<GBuffer> &device_buf,
                                                        const int64_t position,
                                                        const int64_t nbytes,
                                                        uint8_t **host_buf,
                                                        arrow::MemoryPool *pool) {
    assert(typeid(*buf) == typeid(CudaBuffer));
    auto device_cuda_buf = std::static_pointer_cast<CudaBuffer>(device_buf);
    // Allocate memory on the host
    auto status = pool->Allocate(nbytes, host_buf);
    if (!status.ok()) return status;
    // Copy from device to host
    status = device_cuda_buf->get_buffer()->CopyToHost(position, nbytes, (void *) (*host_buf));
    return arrow::Status::OK();
}

CudaContext make_cuda_context() {
    return std::make_unique<MaximusCudaContext>();
}

}  // namespace gpu
}  // namespace maximus
