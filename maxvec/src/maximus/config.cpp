#include <algorithm>
#include <cassert>
#include <iostream>
#include <maximus/config.hpp>
#include <maximus/profiler/profiler.hpp>

namespace maximus {

bool is_power_of_two(std::size_t n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

int find_exponent(std::size_t n) {
    if (n == 0) return -1;  // Edge case: 0 is not a power of 2
    int exponent = 0;
    while (n > 1) {
        n >>= 1;
        exponent++;
    }
    return exponent;
}

int get_num_inner_threads() {
    auto num_inner_threads = get_env_var<int>(env_vars_names::MAXIMUS_NUM_INNER_THREADS,
                                              env_vars_defaults::MAXIMUS_NUM_INNER_THREADS);
    assert(num_inner_threads >= 1);

    return num_inner_threads;
}

int get_num_io_threads() {
    auto num_io_threads = get_env_var<int>(env_vars_names::MAXIMUS_NUM_IO_THREADS,
                                           env_vars_defaults::MAXIMUS_NUM_IO_THREADS);
    assert(num_io_threads >= 1);

    return num_io_threads;
}

int get_num_outer_threads() {
    auto num_outer_threads = get_env_var<int>(env_vars_names::MAXIMUS_NUM_OUTER_THREADS,
                                              env_vars_defaults::MAXIMUS_NUM_OUTER_THREADS);
    assert(num_outer_threads >= 1);

    // although the Caliper profiler seems to be thread-safe,
    // if the profiler is active, all threads must execute all the pipelines in the same order
    // so that the profiling regions are matching on all the threads
    // for this reason, we have to execute the pipelines serially
    if (profiler::is_active()) {
        std::cerr
            << "Warning: The number of outer threads is set to 1 because the profiler is active."
            << std::endl;
        num_outer_threads = 1;
    }

    return num_outer_threads;
}

bool get_operators_fusion() {
    return get_env_var<bool>(env_vars_names::MAXIMUS_OPERATORS_FUSION,
                             env_vars_defaults::MAXIMUS_OPERATORS_FUSION);
}

int32_t get_csv_batch_size() {
    return get_env_var<int32_t>(env_vars_names::MAXIMUS_CSV_BATCH_SIZE,
                                env_vars_defaults::MAXIMUS_CSV_BATCH_SIZE);
}

std::size_t get_max_pinned_pool_size() {
    auto size = get_env_var<std::size_t>(env_vars_names::MAXIMUS_MAX_PINNED_POOL_SIZE,
                                         env_vars_defaults::MAXIMUS_MAX_PINNED_POOL_SIZE);

    // cudf requires the pinned memory pool size to be a multiple of 256
    auto const aligned_size = (size + 255) & ~255;
    return aligned_size;
}

#ifdef MAXIMUS_WITH_DATASET_API
bool get_dataset_api() {
    return get_env_var<bool>(env_vars_names::MAXIMUS_DATASET_API,
                             env_vars_defaults::MAXIMUS_DATASET_API);
}
#endif

}  // namespace maximus
