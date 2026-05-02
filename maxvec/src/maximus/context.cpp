#include <arrow/compute/initialize.h>
#include <arrow/io/api.h>
#include <arrow/util/thread_pool.h>

#include <cassert>
#include <maximus/context.hpp>
#include <maximus/error_handling.hpp>
#include <maximus/memory_pool.hpp>
#include <maximus/profiler/profiler.hpp>

#ifdef MAXIMUS_WITH_CUDA
#include <maximus/gpu/gtable/cuda/cuda_context.hpp>
#endif

#ifdef MAXIMUS_WITH_FAISS
#include <omp.h>
#endif

namespace maximus {

MaximusContext::MaximusContext() {
    default_pool = std::make_unique<DefaultArrowMemoryPool>();
    auto pool    = get_memory_pool();
    assert(pool);

    n_outer_threads = get_num_outer_threads();
    n_inner_threads = get_num_inner_threads();
    n_io_threads    = get_num_io_threads();
    assert(n_outer_threads >= 1);
    assert(n_inner_threads >= 1);
    assert(n_io_threads >= 1);

    auto maybe_io_thread_pool = arrow::internal::ThreadPool::Make(n_io_threads);
    CHECK_STATUS(maybe_io_thread_pool.status());
    io_thread_pool_ = maybe_io_thread_pool.ValueOrDie();

    io_context = std::make_unique<arrow::io::IOContext>(pool, io_thread_pool_.get());

    max_pinned_pool_size = get_max_pinned_pool_size();
    assert(max_pinned_pool_size > 0);

#ifdef MAXIMUS_WITH_CUDA
    gcontext    = gpu::make_cuda_context();
    pinned_pool = std::make_unique<PinnedMemoryPool>(max_pinned_pool_size);

    stream_pool   = cudf::detail::create_global_cuda_stream_pool();
    stream_vector = stream_pool->get_streams(2);
    h2d_stream    = stream_vector[0];
    d2h_stream    = stream_vector[1];

    rmm::mr::set_current_device_resource(
        &pool_mr);  // Updates the current device resource pointer to `pool_mr`
#endif

#ifdef MAXIMUS_WITH_FAISS
    // For CPU operators internal threading
    omp_set_num_threads(n_inner_threads);
#endif

    CHECK_STATUS(arrow::SetCpuThreadPoolCapacity(n_inner_threads));
    CHECK_STATUS(arrow::io::SetIOThreadPoolCapacity(n_io_threads));
    exec_context = std::make_unique<arrow::compute::ExecContext>(
        get_memory_pool(), arrow::internal::GetCpuThreadPool());

    csv_batch_size = get_csv_batch_size();
    assert(csv_batch_size > 0);
    assert(csv_batch_size <= 1 << 30);  // 1GB (max block size, because of int32_t)

    // starting from apache arrow v21.0, arrow requires initializing the compute library
    // otherwise, only core arrow functions will be registered and available
    CHECK_STATUS(arrow::compute::Initialize());
}

MaximusContext::~MaximusContext() {
    // default_pool.release();
}

void MaximusContext::set_memory_pool(std::unique_ptr<MemoryPool> &&pool) {
    assert(!proxy_pool);
    assert(!default_pool);
    proxy_pool = std::make_unique<ProxyMemoryPool>(std::move(pool));
}

arrow::MemoryPool *MaximusContext::get_memory_pool() {
    // exactly one pool can be set
    if (proxy_pool) {
        assert(!default_pool);
        return proxy_pool.get();
    }
    return default_pool.get();
}

arrow::MemoryPool *MaximusContext::get_pinned_memory_pool() {
    assert(pinned_pool);
    // exactly one pool can be set
    return pinned_pool.get();
}

arrow::MemoryPool *MaximusContext::get_pinned_memory_pool_if_available() {
    if (pinned_pool) {
        return get_pinned_memory_pool();
    }
    return get_memory_pool();
}

arrow::compute::ExecContext *MaximusContext::get_exec_context() {
    assert(exec_context);
    return exec_context.get();
}

arrow::io::IOContext *MaximusContext::get_io_context() {
    return io_context.get();
}

std::shared_ptr<maximus::gpu::MaximusGContext> &MaximusContext::get_gpu_context() {
    return gcontext;
}

arrow::acero::QueryOptions MaximusContext::get_query_options() {
    arrow::acero::QueryOptions options;
    options.memory_pool       = get_exec_context()->memory_pool();
    options.function_registry = get_exec_context()->func_registry();
    return options;
}

std::shared_ptr<arrow::acero::ExecPlan> MaximusContext::get_mini_exec_plan() {
    auto maybe_plan = arrow::acero::ExecPlan::Make(get_exec_context());

    if (!maybe_plan.ok()) {
        CHECK_STATUS(maybe_plan.status());
    }

    return std::move(maybe_plan.ValueOrDie());
}

void MaximusContext::barrier() const {
#ifdef MAXIMUS_WITH_CUDA
    wait_d2h_copy();
#endif
}

#ifdef MAXIMUS_WITH_CUDA
rmm::cuda_stream_view MaximusContext::get_h2d_stream() {
    if (use_separate_copy_streams) {
        return h2d_stream;
    }
    return cudf::get_default_stream();
}

rmm::cuda_stream_view MaximusContext::get_d2h_stream() {
    if (use_separate_copy_streams) {
        return d2h_stream;
    }
    return cudf::get_default_stream();
}

rmm::cuda_stream_view MaximusContext::get_kernel_stream() {
    return cudf::get_default_stream();
}

void MaximusContext::wait_h2d_copy() const {
    if (use_separate_copy_streams) {
        h2d_stream.synchronize();
    }
}

void MaximusContext::wait_d2h_copy() const {
    if (use_separate_copy_streams) {
        d2h_stream.synchronize();
    } else {
        cudf::get_default_stream().synchronize();
    }
}
#endif

Context make_context() {
    static std::shared_ptr<MaximusContext> ctx = std::make_shared<MaximusContext>();
    return ctx;
}
}  // namespace maximus
