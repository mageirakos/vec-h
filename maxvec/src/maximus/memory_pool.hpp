#pragma once

#include <arrow/memory_pool.h>
#include <sys/mman.h>

#include <atomic>
#include <cassert>
#include <iostream>
#include <maximus/error_handling.hpp>
#include <mutex>

#ifdef MAXIMUS_WITH_CUDA
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/pinned_memory.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>
#endif


namespace maximus {

// The memory pool interface resembles the interface of the Apache Arrow memory pool.
class MemoryPool {
public:
    virtual ~MemoryPool() = default;

    /// Allocates a memory block of at least the specified size.
    ///
    /// The allocated memory will be aligned to a 64-byte boundary.
    /// @param size The minimum number of bytes to allocate.
    /// @param alignment The required alignment for the memory block.
    /// @param out Pointer to store the address of the allocated memory.
    /// @return Status indicating success or failure.
    virtual Status allocate(int64_t size, int64_t alignment, uint8_t** out) = 0;

    /// Resizes an existing allocated memory block.
    ///
    /// Since most platform allocators do not support aligned reallocation,
    /// this operation may involve copying the data to a new memory block.
    /// @param old_size The current size of the allocated memory block.
    /// @param new_size The desired new size of the memory block.
    /// @param alignment The alignment requirement of the memory block.
    /// @param ptr Pointer to the memory block to be resized. Updated on success.
    /// @return Status indicating success or failure.
    virtual Status reallocate(int64_t old_size,
                              int64_t new_size,
                              int64_t alignment,
                              uint8_t** ptr) = 0;

    /// Frees a previously allocated memory block.
    ///
    /// @param buffer Pointer to the start of the allocated memory block.
    /// @param size The size of the allocated memory block.
    ///        Some allocators may use this for tracking memory usage or
    ///        optimizing deallocation.
    /// @param alignment The alignment of the memory block.
    virtual void free(uint8_t* buffer, int64_t size, int64_t alignment) = 0;

    /// Retrieves the current amount of allocated memory that has not been freed.
    ///
    /// @return The number of bytes currently allocated.
    virtual int64_t bytes_allocated() const = 0;

    /// Retrieves the total amount of memory allocated since the pool's creation.
    ///
    /// @return The cumulative number of bytes allocated.
    virtual int64_t total_bytes_allocated() const = 0;

    /// Retrieves the total number of allocation and reallocation requests.
    ///
    /// @return The number of times memory has been allocated or reallocated.
    virtual int64_t num_allocations() const = 0;

    /// Retrieves the peak memory usage recorded by this memory pool.
    ///
    /// @return The highest number of bytes allocated at any point.
    ///         Returns -1 if tracking is not implemented.
    virtual int64_t max_memory() const = 0;

    /// Retrieves the name of the memory allocation backend in use.
    ///
    /// @return A string representing the backend (e.g., "system", "jemalloc").
    virtual std::string backend_name() const = 0;
};

// forward-declaration
class PinnedMemoryPool;

#ifdef MAXIMUS_WITH_CUDA
class PinnedMemoryPool : public arrow::MemoryPool {
public:
    PinnedMemoryPool(std::size_t pool_size) {
        cudf::pinned_mr_options options;
        options.pool_size = pool_size;
        bool success      = cudf::config_default_pinned_memory_resource(options);
        if (!success) {
            throw std::runtime_error("configuring the pinned memory pool not successful.");
        }

        // allocate and deallocate
        uint8_t* ptr;
        auto status = do_allocate(pool_size, 256, &ptr);
        CHECK_STATUS(status);
        do_deallocate(ptr, pool_size, 256);
    }

    ~PinnedMemoryPool() override = default;

    arrow::Status do_allocate(int64_t size, int64_t alignment, uint8_t** out) {
        // alignment = std::max((int64_t) 256, alignment);
        // std::cout << "Allocating size " << size << ", alignment = " << alignment << std::endl;
        if (size == 0) return arrow::Status::OK();
        if (static_cast<uint64_t>(size) >= std::numeric_limits<size_t>::max()) {
            return arrow::Status::OutOfMemory("allocate overflows size_t");
        }

        // std::scoped_lock lock{mutex_};
        void* buf = pool().allocate(size, alignment);
        if (!buf) {
            return arrow::Status::OutOfMemory("Cannot allocate pinned memory.");
        }

        assert(buf);
        *out = reinterpret_cast<uint8_t*>(buf);
        assert(*out);

        total_bytes_allocated_ += size;
        num_allocs_++;
        bytes_allocated_ += size;

        max_memory_mutex_.lock();
        max_memory_ = std::max(bytes_allocated_.load(), max_memory_);
        max_memory_mutex_.unlock();

        // std::cout << "max_memory = " << max_memory_ << std::endl;
        return arrow::Status::OK();
    }

    void do_deallocate(uint8_t* buffer, int64_t size, int64_t alignment) {
        // alignment = std::max((int64_t) 256, alignment);
        // std::cout << "Deallocating size " << size << ", alignment = " << alignment << std::endl;
        if (size == 0) return;
        if (static_cast<uint64_t>(size) >= std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("Arrow out of memory: free overflows size_t");
        }
        assert(buffer);
        void* p = reinterpret_cast<void*>(buffer);
        assert(p);
        pool().deallocate(p, size, alignment);

        if (p) {
            assert(bytes_allocated_.load() >= size);
            bytes_allocated_ -= size;
        }
    }

    arrow::Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override {
        // mutex_.lock();
        auto status = do_allocate(size, alignment, out);
        // mutex_.unlock();

        return status;
    }

    arrow::Status Reallocate(int64_t old_size,
                             int64_t new_size,
                             int64_t alignment,
                             uint8_t** ptr) override {
        do_deallocate(*ptr, old_size, alignment);
        auto status = do_allocate(new_size, alignment, ptr);
        return status;
    }

    void Free(uint8_t* buffer, int64_t size, int64_t alignment) override {
        do_deallocate(buffer, size, alignment);
    }

    int64_t bytes_allocated() const override { return bytes_allocated_.load(); }

    int64_t max_memory() const override {
        return max_memory_;
        // return std::numeric_limits<int64_t>::max();
        // return max_total_allocated;
    }

    int64_t total_bytes_allocated() const override { return total_bytes_allocated_.load(); }

    int64_t num_allocations() const override { return num_allocs_.load(); }

    std::string backend_name() const override { return "PinnedMemoryPool"; }

    rmm::host_device_async_resource_ref pool() { return cudf::get_pinned_memory_resource(); }

private:
    std::atomic_size_t total_bytes_allocated_{0};
    std::atomic_size_t bytes_allocated_{0};
    std::atomic_size_t num_allocs_{0};

    std::size_t max_memory_{0};
    std::mutex max_memory_mutex_;
};
#endif

}  // namespace maximus
