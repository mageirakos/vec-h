#pragma once

#include <string>
#include <vector>

#ifdef MAXIMUS_WITH_CUDA
    #include <cuda_runtime.h>
#endif

namespace maximus {
// Define the helper functions to convert to const char*
inline const char* to_cstr(const std::string& str) {
    return str.c_str();
}

// Overload for C-style strings
inline const char* to_cstr(const char* str) {
    return str;
}

namespace profiler {
// Define a constexpr function to check if the profiler is active
constexpr bool is_active() {
#ifdef MAXIMUS_WITH_PROFILING
    return true;
#else
    return false;
#endif
}


// opens the regions for profiling in the same order as the regions vector
void open_regions(const std::vector<std::string>& regions);

// closes the regions for profiling in the reverse order as the regions vector
void close_regions(const std::vector<std::string>& regions);

}  // namespace profiler
}  // namespace maximus

#ifdef MAXIMUS_WITH_PROFILING
#include <caliper/cali-manager.h>
#include <caliper/cali.h>

// Define a macro for creating and initializing the ConfigManager object
// cali_config_set("CALI_CALIPER_ATTRIBUTE_DEFAULT_SCOPE", "process");
// mgr.add(maximus::to_cstr(config));
#define PROFILER_INIT(mgr, config)                                      \
    cali_config_set("CALI_CALIPER_ATTRIBUTE_DEFAULT_SCOPE", "process"); \
    cali::ConfigManager mgr;                                            \
    mgr.add(maximus::to_cstr(config));                                  \
    if (mgr.error()) std::cerr << "Caliper error: " << mgr.error_msg() << std::endl;

// Define macros for starting and flushing the manager
#define PROFILER_START(mgr) mgr.start();

#define PROFILER_FLUSH(mgr) mgr.flush();

// Define macros for opening and closing the Caliper regions
#ifdef MAXIMUS_WITH_CUDA
    #define PE(name) cudaDeviceSynchronize(); CALI_MARK_BEGIN(maximus::to_cstr(name))
    #define PL(name) cudaDeviceSynchronize(); CALI_MARK_END(maximus::to_cstr(name))
#else
    #define PE(name) CALI_MARK_BEGIN(maximus::to_cstr(name))
    #define PL(name) CALI_MARK_END(maximus::to_cstr(name))
#endif

#else
// Define empty macros when profiling is disabled
#define PROFILER_INIT(mgr, config)
#define PROFILER_START(mgr)
#define PROFILER_FLUSH(mgr)

// Define empty macros when profiling is disabled
#define PE(name)
#define PL(name)
#endif
