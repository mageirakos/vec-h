#include "gpu_resources.hpp"
#include <faiss/gpu/utils/DeviceUtils.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>

#if defined(USE_NVIDIA_CUVS)
#include <raft/core/resource/cuda_stream.hpp>
#endif


namespace maximus {

MaximusFaissGpuResources::MaximusFaissGpuResources(
    std::shared_ptr<MaximusContext> context,
    size_t pinned_memory_size)
    : context_(std::move(context)),
      requestedPinnedMemorySize_(pinned_memory_size) {
    assert(context_ != nullptr && "MaximusContext cannot be null");
}

MaximusFaissGpuResources::~MaximusFaissGpuResources() {
    // Check for memory leaks before cleaning up
    if (!allocs_.empty()) {
        std::cerr << "WARNING: MaximusFaissGpuResources destroyed with "
                  << allocs_.size() << " memory allocations still outstanding." << std::endl;
        // Optionally, print the outstanding allocations
    }
    
    // Clean up resources this class created and owns
    for (auto const& [device, handle] : blasHandles_) {
        ::faiss::gpu::DeviceScope scope(device);
        cublasDestroy(handle);
    }

    // Free the pinned memory buffer we allocated from the Maximus pool
    if (pinnedBuffer_) {
        auto pool = context_->get_pinned_memory_pool();
        pool->Free(static_cast<uint8_t*>(pinnedBuffer_), pinnedBufferSize_);
    }

    // RAFT handles and user streams are wrappers and don't need destruction.
}

void MaximusFaissGpuResources::initializeForDevice(int device) {
    // Use a lock to ensure thread-safe initialization
    std::lock_guard<std::mutex> lock(allocsMutex_);

    if (blasHandles_.count(device)) {
        return; // Already initialized
    }
    
    ::faiss::gpu::DeviceScope scope(device);

    // Lazily allocate the pinned memory buffer on the first device initialization
    if (pinnedBuffer_ == nullptr && requestedPinnedMemorySize_ > 0) {
        auto pool = context_->get_pinned_memory_pool();
        FAISS_ASSERT(pool);
        arrow::Status status = pool->Allocate(requestedPinnedMemorySize_,
                                            reinterpret_cast<uint8_t**>(&pinnedBuffer_));
        if (status.ok()) {
             pinnedBufferSize_ = requestedPinnedMemorySize_;
        } else {
             std::cerr << "WARNING: Failed to allocate pinned memory from Maximus pool: "
                       << status.ToString() << std::endl;
             pinnedBuffer_ = nullptr;
             pinnedBufferSize_ = 0;
        }
    }

    // Create and store a cuBLAS handle
    cublasHandle_t handle;
    auto status = cublasCreate(&handle);
    assert(status == CUBLAS_STATUS_SUCCESS && "Failed to create cuBLAS handle for device");
    blasHandles_[device] = handle;
    
#if CUDA_VERSION >= 11000
    cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
#endif

#if defined(USE_NVIDIA_CUVS)
    // Create and store a RAFT handle, wrapping the default Maximus stream
    auto defaultStream = getDefaultStream(device); // gets the underlying stream
    raftHandles_.emplace(std::piecewise_construct,
                         std::forward_as_tuple(device),
                         std::forward_as_tuple());
    raft::resource::set_cuda_stream(raftHandles_.at(device), defaultStream);
#endif
}

bool MaximusFaissGpuResources::supportsBFloat16(int device) {
    auto& props = ::faiss::gpu::getDeviceProperties(device);
    return props.major >= 8;
}

cublasHandle_t MaximusFaissGpuResources::getBlasHandle(int device) {
    if (!blasHandles_.count(device)) {
        initializeForDevice(device);
    }
    return blasHandles_.at(device);
}

cudaStream_t MaximusFaissGpuResources::getDefaultStream(int device) {
    if (userDefaultStreams_.count(device)) {
        return userDefaultStreams_.at(device);
    }

    return cudf::get_default_stream().value();
}

void MaximusFaissGpuResources::setDefaultStream(int device, cudaStream_t stream) {
    userDefaultStreams_[device] = stream;

#if defined(USE_NVIDIA_CUVS)
    // If a RAFT handle exists for this device, we must update its stream
    if (raftHandles_.count(device)) {
        raft::resource::set_cuda_stream(raftHandles_.at(device), stream);
    }
#endif
}

std::vector<cudaStream_t> MaximusFaissGpuResources::getAlternateStreams(int device) {
    // not sure if this is using d2h stream by accident. 
    // TODO: check what FAISS wants and create separate streams if needed. For now using the ones we have.
    if (context_->stream_vector.size() <= 1) {
        return {};
    }
    return std::vector<cudaStream_t>(context_->stream_vector.begin() + 1,
                                     context_->stream_vector.end());
}

void* MaximusFaissGpuResources::allocMemory(const ::faiss::gpu::AllocRequest& req) {
    if (req.space != ::faiss::gpu::MemorySpace::Device && req.space != ::faiss::gpu::MemorySpace::Temporary) {
        throw std::runtime_error("MaximusFaissGpuResources only supports Device and Temporary memory allocations.");
    }
    if (req.size == 0) {
        return nullptr;
    }

    void* ptr = nullptr;
    try {
        ptr = context_->pool_mr.allocate_async(req.size, req.stream);
    } catch (const std::exception& e) {
        FAISS_THROW_FMT("RMM memory allocation of %zu bytes failed: %s", req.size, e.what());
    }
    
    FAISS_ASSERT(ptr);
    
    std::lock_guard<std::mutex> lock(allocsMutex_);
    allocs_[ptr] = req;
    
    return ptr;
}

void MaximusFaissGpuResources::deallocMemory(int device, void* p) {
    if (!p) return;
    
    std::lock_guard<std::mutex> lock(allocsMutex_);
    auto it = allocs_.find(p);
    assert(it != allocs_.end() && "Attempting to deallocate untracked memory at");
    
    const auto& req = it->second;
    
    // Use deallocate_async with the original request's stream for correct synchronization.
    context_->pool_mr.deallocate_async(p, req.size, req.stream);
    
    allocs_.erase(it);
}

size_t MaximusFaissGpuResources::getTempMemoryAvailable(int device) const {
    // Returning 0 is correct. It signals to Faiss that we don't have a separate,
    // pre-allocated temp buffer, so all temp requests go through allocMemory.
    return 0;
}

std::pair<void*, size_t> MaximusFaissGpuResources::getPinnedMemory() {
    // Lazily initialize on first request if not already done
    if (pinnedBuffer_ == nullptr) {
        // Assume device 0 if called before any other operation
        int device = 0;
        cudaGetDevice(&device);
        initializeForDevice(device);
    }
    return {pinnedBuffer_, pinnedBufferSize_};
}

cudaStream_t MaximusFaissGpuResources::getAsyncCopyStream(int device) {
    return context_->get_h2d_stream();
}

#if defined(USE_NVIDIA_CUVS)
raft::device_resources& MaximusFaissGpuResources::getRaftHandle(int device) {
    if (!raftHandles_.count(device)) {
        initializeForDevice(device);
    }
    return raftHandles_.at(device);
}
#endif

} // namespace maximus