#pragma once

#include <faiss/gpu/GpuResources.h>
#include <faiss/impl/FaissAssert.h>
#include <maximus/context.hpp> // Your provided context header
#include <cudf/utilities/default_stream.hpp>

#include <mutex>
#include <stdexcept>
#include <unordered_map>

// Conditionally include RAFT headers if Faiss is compiled with CUVS
#if defined(USE_NVIDIA_CUVS)
#include <raft/core/device_resources.hpp>
#endif

// Forward declaration from cublas_v2.h
struct cublasContext;
using cublasHandle_t = cublasContext*;


namespace maximus {

/**
 * @class MaximusFaissGpuResources (EXPERIMENTAL, NOT THOROUGLY TESTED)
 * @brief An implementation of faiss::gpu::GpuResources that bridges to a
 * maximus::MaximusContext.
 *
 * This class ensures that Faiss GPU indexes utilize the same CUDA resources
 * (RMM memory pool, CUDA streams) as the rest of the Maximus database.
 * It properly manages RAFT handles and integrates with the Maximus pinned
 * memory pool for efficient asynchronous data transfers.
 *
 */
class MaximusFaissGpuResources : public ::faiss::gpu::GpuResources {
public:
    /**
     * @brief Constructs the resource bridge.
     * @param context The shared pointer to the MaximusContext that manages
     * the underlying GPU resources.
     * @param pinned_memory_size The amount of pinned host memory to allocate
     * from the Maximus pool for use by Faiss. Defaults to 256 MiB.
     */
    explicit MaximusFaissGpuResources(
        std::shared_ptr<MaximusContext> context,
        size_t pinned_memory_size = 256 * 1024 * 1024);

    ~MaximusFaissGpuResources() override;

    // Delete copy and move constructors for safety
    MaximusFaissGpuResources(const MaximusFaissGpuResources&) = delete;
    MaximusFaissGpuResources& operator=(const MaximusFaissGpuResources&) = delete;

    //
    // faiss::gpu::GpuResources virtual method overrides
    //

    void initializeForDevice(int device) override;

    bool supportsBFloat16(int device) override;

    cublasHandle_t getBlasHandle(int device) override;

    cudaStream_t getDefaultStream(int device) override;

    void setDefaultStream(int device, cudaStream_t stream) override;

    std::vector<cudaStream_t> getAlternateStreams(int device) override;

    void* allocMemory(const ::faiss::gpu::AllocRequest& req) override;

    void deallocMemory(int device, void* p) override;

    size_t getTempMemoryAvailable(int device) const override;

    std::pair<void*, size_t> getPinnedMemory() override;

    cudaStream_t getAsyncCopyStream(int device) override;

#if defined(USE_NVIDIA_CUVS)
    raft::device_resources& getRaftHandle(int device) override;
#endif

private:
    /// The Maximus context that owns the actual GPU resources.
    std::shared_ptr<MaximusContext> context_;

    /// Faiss requires a cuBLAS handle; we manage it here since it's not in MaximusContext.
    std::unordered_map<int, cublasHandle_t> blasHandles_;

    /// Stores user-overridden default streams per device.
    std::unordered_map<int, cudaStream_t> userDefaultStreams_;

    /// Tracks the original allocation request to enable proper deallocation.
    std::unordered_map<void*, ::faiss::gpu::AllocRequest> allocs_;
    
    /// Mutex to protect the allocation map from concurrent access.
    std::mutex allocsMutex_;

    // --- Pinned Memory Management ---
    const size_t requestedPinnedMemorySize_;
    void* pinnedBuffer_ = nullptr;
    size_t pinnedBufferSize_ = 0;

#if defined(USE_NVIDIA_CUVS)
    /// RAFT handles, one per initialized device, wrapping the Maximus streams.
    std::unordered_map<int, raft::device_resources> raftHandles_;
#endif
};

} // namespace maximus