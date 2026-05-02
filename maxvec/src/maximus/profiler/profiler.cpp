#include <maximus/profiler/profiler.hpp>

namespace maximus::profiler {

// opens the regions for profiling in the same order as the regions vector
void open_regions(const std::vector<std::string>& regions) {
    for (const auto& region : regions) {
        PE(region);
    }
}

// closes the regions for profiling in the reverse order as the regions vector
void close_regions(const std::vector<std::string>& regions) {
    // iterate over regions in reverse order
    for (auto it = regions.rbegin(); it != regions.rend(); ++it) {
        PL(*it);
    }
}
}  // namespace maximus::profiler