#pragma once
#include <arrow/memory_pool.h>

#include <cassert>
#include <maximus/memory_pool.hpp>
#include <maximus/profiler/profiler.hpp>

namespace maximus {

inline arrow::Status arrow_status(const maximus::Status &status) {
    return arrow::Status(static_cast<arrow::StatusCode>(status.code()), status.message());
}

class ProxyMemoryPool : public arrow::MemoryPool {
public:
    explicit ProxyMemoryPool(std::unique_ptr<maximus::MemoryPool> &&pool): pool_(std::move(pool)) {}

    arrow::Status Allocate(int64_t size, int64_t alignment, uint8_t **out) override {
        assert(pool_);
        return arrow_status(pool_->allocate(size, alignment, out));
    }

    arrow::Status Reallocate(int64_t old_size,
                             int64_t new_size,
                             int64_t alignment,
                             uint8_t **ptr) override {
        assert(pool_);
        return arrow_status(pool_->reallocate(old_size, new_size, alignment, ptr));
    };

    void Free(uint8_t *buffer, int64_t size, int64_t alignment) override {
        assert(pool_);
        pool_->free(buffer, size, alignment);
    }

    int64_t bytes_allocated() const override {
        assert(pool_);
        return this->pool_->bytes_allocated();
    }

    /// The number of bytes that were allocated.
    int64_t total_bytes_allocated() const override { return pool_->total_bytes_allocated(); }

    /// The number of allocations or reallocations that were requested.
    int64_t num_allocations() const override { return pool_->num_allocations(); }

    int64_t max_memory() const override {
        assert(pool_);
        return this->pool_->max_memory();
    }

    std::string backend_name() const override {
        assert(pool_);
        return this->pool_->backend_name();
    }

private:
    std::unique_ptr<maximus::MemoryPool> pool_;
};

class DefaultArrowMemoryPool : public arrow::MemoryPool {
public:
    DefaultArrowMemoryPool() = default;

    ~DefaultArrowMemoryPool() {
        // release the ownership to prevent freeing
        // since arrow will free the default pool
        if (pool_) {
            pool_.release();
        }
    }

    arrow::Status Allocate(int64_t size, int64_t alignment, uint8_t **out) override {
        // PE("DefaultArrowMemoryPool::allocate");
        assert(pool_);

        arrow::Status status;
        {
            status = pool_->Allocate(size, alignment, out);
        }

        // PL("DefaultArrowMemoryPool::allocate");
        return status;
    }

    arrow::Status Reallocate(int64_t old_size,
                             int64_t new_size,
                             int64_t alignment,
                             uint8_t **ptr) override {
        // PE("DefaultArrowMemoryPool::reallocate");
        assert(pool_);
        arrow::Status status;
        {
            status = pool_->Reallocate(old_size, new_size, alignment, ptr);
        }
        // PL("DefaultArrowMemoryPool::reallocate");
        return status;
    };

    void Free(uint8_t *buffer, int64_t size, int64_t alignment) override {
        // PE("DefaultArrowMemoryPool::free");
        assert(pool_);
        {
            pool_->Free(buffer, size, alignment);
        }
        // PL("DefaultArrowMemoryPool::free");
    }

    int64_t bytes_allocated() const override {
        assert(pool_);
        int64_t bytes = 0;
        {
            bytes = this->pool_->bytes_allocated();
        }
        return bytes;
    }

    /// The number of bytes that were allocated.
    int64_t total_bytes_allocated() const override {
        int64_t bytes = 0;
        {
            bytes = pool_->total_bytes_allocated();
        }
        return bytes;
    }

    /// The number of allocations or reallocations that were requested.
    int64_t num_allocations() const override {
        int64_t num = 0;
        {
            num = pool_->num_allocations();
        }
        return num;
    }

    int64_t max_memory() const override {
        assert(pool_);
        return this->pool_->max_memory();
    }

    std::string backend_name() const override {
        assert(pool_);
        return this->pool_->backend_name();
    }

private:
    std::unique_ptr<arrow::MemoryPool> pool_ =
        std::unique_ptr<arrow::MemoryPool>(arrow::default_memory_pool());
};
}  // namespace maximus
