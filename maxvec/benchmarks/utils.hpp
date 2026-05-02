#pragma once

#include <iostream>
#include <sstream>
#include <cstring>
#include <memory>
#include <optional>
#include <maximus/clickbench/clickbench_queries.hpp>
#include <maximus/context.hpp>
#include <maximus/database.hpp>
#include <maximus/h2o/h2o_queries.hpp>
#include <maximus/indexes/faiss/faiss_index.hpp>
#include <maximus/profiler/profiler.hpp>
#include <maximus/tpch/tpch_queries.hpp>
#include <maximus/utils/utils.hpp>
#include <maximus/vsds/vsds_queries.hpp>
#include <arrow/memory_pool.h>
#ifdef MAXIMUS_WITH_CUDA
#include <cuda_runtime.h>
#endif

// Remove specified keys from a comma-separated key=value params string.
// E.g. strip_keys("k=100,query_count=1,metric=IP", {"query_count", "query_start"})
//   -> "k=100,metric=IP"
inline std::string strip_keys(const std::string& params, const std::vector<std::string>& keys_to_remove) {
    std::istringstream stream(params);
    std::string token;
    std::string result;
    while (std::getline(stream, token, ',')) {
        auto eq = token.find('=');
        if (eq == std::string::npos) continue;
        std::string key = token.substr(0, eq);
        // trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        bool skip = false;
        for (const auto& k : keys_to_remove) {
            if (key == k) { skip = true; break; }
        }
        if (!skip) {
            if (!result.empty()) result += ",";
            result += token;
        }
    }
    return result;
}

#ifdef MAXIMUS_WITH_CUDA
inline void print_gpu_mem(const char* label) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    for (int dev = 0; dev < device_count; ++dev) {
        size_t free_bytes = 0, total_bytes = 0;
        cudaSetDevice(dev);
        cudaMemGetInfo(&free_bytes, &total_bytes);
        double total_gb = total_bytes / 1024.0 / 1024.0 / 1024.0;
        double free_gb  = free_bytes  / 1024.0 / 1024.0 / 1024.0;
        double used_gb  = total_gb - free_gb;
        printf("[GPU MEM] %s: free=%.2f GB, used=%.2f GB, total=%.2f GB\n",
               label, free_gb, used_gb, total_gb);
    }
}
#else
inline void print_gpu_mem(const char* /*label*/) {}
#endif

inline void print_mem(const char* label, std::shared_ptr<maximus::MaximusContext>& ctx) {
    print_gpu_mem(label);
    if (ctx) {
        auto* maximus_pool = ctx->get_memory_pool();
        auto* default_pool = arrow::default_memory_pool();
        printf("[MEM DETAIL] %s:\n"
               "  Arrow (maximus pool): allocated=%.2f GiB, total_allocated=%.2f GiB, num_allocs=%ld\n"
               "  Arrow (default pool): allocated=%.2f GiB, total_allocated=%.2f GiB\n",
               label,
               maximus_pool->bytes_allocated() / (1024.0*1024.0*1024.0),
               maximus_pool->total_bytes_allocated() / (1024.0*1024.0*1024.0),
               (long)maximus_pool->num_allocations(),
               default_pool->bytes_allocated() / (1024.0*1024.0*1024.0),
               default_pool->total_bytes_allocated() / (1024.0*1024.0*1024.0));
        FILE* f = fopen("/proc/self/status", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                if (strncmp(line, "VmRSS:", 6) == 0 || strncmp(line, "VmSize:", 7) == 0)
                    printf("  %s", line);
            }
            fclose(f);
        }
    }
}

// Flush CPU L2/L3 and (optionally) GPU L2 caches between benchmark repetitions.
// CPU: reads a 256 MB buffer (> typical L3 = 32-64 MB) with full cache-line stride.
// GPU: allocates + writes 96 MB on the device (covers H100 50 MB / A100 40 MB / GH200 60 MB L2).
// Call this immediately before context->barrier() + timing start in each rep.
inline void flush_caches(bool do_flush, maximus::DeviceType device) {
    if (!do_flush) return;

    // --- CPU L3 flush ---
    constexpr std::size_t cpu_flush_bytes = 256ULL * 1024 * 1024;  // 256 MB
    auto cpu_buf = std::make_unique<char[]>(cpu_flush_bytes);
    // write-then-read to force physical pages and evict existing cache lines
    std::memset(cpu_buf.get(), 1, cpu_flush_bytes);
    volatile char acc = 0;
    const char* ptr = cpu_buf.get();
    for (std::size_t i = 0; i < cpu_flush_bytes; i += 64) {
        acc += ptr[i];
    }
    (void)acc;

#ifdef MAXIMUS_WITH_CUDA
    if (device == maximus::DeviceType::GPU) {
        // GPU L2 flush: 96 MB covers the largest GPU L2 in this project's targets
        constexpr std::size_t gpu_flush_bytes = 96ULL * 1024 * 1024;
        void* gpu_buf = nullptr;
        cudaMalloc(&gpu_buf, gpu_flush_bytes);
        cudaMemset(gpu_buf, 0, gpu_flush_bytes);
        cudaDeviceSynchronize();
        cudaFree(gpu_buf);
        cudaDeviceSynchronize();  // settle page tables after cudaFree on ATS/unified memory (GH200)
    }
#endif
}

std::string csv_path() {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/csv-0.01";
    return path;
}

std::string parquet_path() {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/tpch/parquet";
    return path;
}

std::string to_string(const std::vector<std::string>& vec) {
    std::string str = "";
    for (unsigned i = 0u; i < vec.size(); ++i) {
        if (i == vec.size() - 1) {
            str += vec[i];
            continue;
        }
        str += vec[i] + ", ";
    }
    return str;
}

void print_output_table(const std::shared_ptr<maximus::MaximusContext>& ctx,
                        const std::string& engines,
                        const std::string& engine,
                        const maximus::TablePtr& table,
                        int limit) {
    if (maximus::contains(engines, engine)) {
        if (table) {
            int topn = std::min((int64_t)limit, (int64_t)table->num_rows());
            std::cout << engine + " RESULTS (top " << topn << " rows out of " << table->num_rows() << "):" << "\n";
            table->slice(0, topn)->print();
            std::cout << "\n\n";
        } else {
            std::cout << engine + " RESULTS (top 50 rows):" << "\n";
            std::cout << "---> The query result is empty."
                      << "\n\n";
        }
    }
}

void write_result_to_file(const std::shared_ptr<maximus::MaximusContext>& ctx,
                          const std::string& engines,
                          const std::string& engine,
                          const std::string& device_string,
                          const int query_id,
                          const maximus::TablePtr& table) {
    if (maximus::contains(engines, engine)) {
        if (!table) {
            std::cerr << "[write_result_to_file] WARNING: " << engine << " result table is null, skipping." << std::endl;
            return;
        }
        std::ostringstream oss;
        oss << engine << "_" << query_id << "." << device_string << ".csv";
        std::string target_name = oss.str();
        std::ofstream file;
        file.open(target_name);
        file << table->to_string();
        file.close();
        std::cout << "Query results saved to " << target_name << std::endl;
    }
}

// New improved version that takes full CSV file path
void write_result_to_csv(const std::shared_ptr<maximus::MaximusContext>& ctx,
                         const std::string& engines,
                         const std::string& engine,
                         const std::string& csv_filepath,
                         const maximus::TablePtr& table) {
    if (maximus::contains(engines, engine)) {
        if (!table) {
            std::cerr << "[write_result_to_csv] WARNING: " << engine << " result table is null, skipping CSV write." << std::endl;
            return;
        }
        // Construct filename: prepend engine name to the given path
        // e.g., csv_filepath = "dir/q2_cpu_cpu_0.01.csv" -> "dir/maximus_q2_cpu_cpu_0.01.csv"
        std::string filepath = csv_filepath;
        
        // Find the last directory separator
        size_t last_slash = filepath.find_last_of("/\\");
        std::string dir = "";
        std::string filename = filepath;
        
        if (last_slash != std::string::npos) {
            dir = filepath.substr(0, last_slash + 1);
            filename = filepath.substr(last_slash + 1);
        }
        
        // Prepend engine name to filename
        std::string target_name = dir + engine + "_" + filename;
        
        // Create directory if needed
        if (!dir.empty()) {
            std::string mkdir_cmd = "mkdir -p " + dir;
            if (system(mkdir_cmd.c_str()) != 0) {
                std::cerr << "ERROR: Failed to create directory: " << dir << "\n";
            }
        }
        
        std::ofstream file;
        file.open(target_name);
        if (file.is_open()) {
            file << table->to_string();
            file.close();
            std::cout << "Query results saved to " << target_name << std::endl;
        } else {
            std::cout << "Failed to create CSV file: " << target_name << std::endl;
        }
    }
}

struct timing_stats {
    timing_stats() = default;
    timing_stats(std::vector<std::vector<int64_t>>& timings,
                 std::vector<std::string> queries,
                 std::string engine,
                 std::string type,
                 maximus::DeviceType device) {
        std::string device_string = device == maximus::DeviceType::CPU ? "cpu" : "gpu";
        std::stringstream csv_results_stream;
        for (int i = 0; i < timings.size(); ++i) {
            csv_results_stream << device_string << "," << engine << "," << type << "," << queries[i]
                               << ",";
            min.push_back(*std::min_element(timings[i].begin(), timings[i].end()));
            max.push_back(*std::max_element(timings[i].begin(), timings[i].end()));
            avg.push_back(std::accumulate(timings[i].begin(), timings[i].end(), 0) /
                          timings[i].size());

            std::string timings_flattened = "\t";
            for (int j = 0; j < timings[i].size(); ++j) {
                timings_flattened += std::to_string(timings[i][j]);
                if (j != timings[i].size() - 1) {
                    timings_flattened += ", \t";
                }
                csv_results_stream << timings[i][j] << ",";
            }
            csv_results_stream << "\n";

            flattened.push_back(timings_flattened);
        }
        csv_results = csv_results_stream.str();
    }

    // maps queries to their min, max, and avg timings as well as a flattened string containing all the timings
    std::vector<int64_t> min;
    std::vector<int64_t> max;
    std::vector<int64_t> avg;
    std::vector<std::string> flattened;
    std::string csv_results;
};

void load_tables(const std::shared_ptr<maximus::Database>& db,
                 const std::vector<std::string>& tables,
                 const std::vector<std::shared_ptr<maximus::Schema>>& schemas = {},
                 const maximus::DeviceType& storage_device = maximus::DeviceType::CPU,
                 const std::vector<std::string>& force_cpu_tables = {}) {
    assert(schemas.empty() || schemas.size() == tables.size());
    for (unsigned i = 0u; i < tables.size(); ++i) {
        maximus::DeviceType table_device = storage_device;
        bool forced_cpu = false;
        if (std::find(force_cpu_tables.begin(), force_cpu_tables.end(), tables[i]) != force_cpu_tables.end()) {
            table_device = maximus::DeviceType::CPU;
            forced_cpu = true;
        }

        if (forced_cpu && storage_device == maximus::DeviceType::GPU) {
             std::cout << "[INFO] Forcing table '" << tables[i] << "' to load on CPU (should be moved to GPU later to avoid het/ous table locations).\n";
        }

        db->load_table(tables[i], schemas.empty() ? nullptr : schemas[i], {}, table_device);
        print_gpu_mem(("  loaded: " + tables[i]).c_str());
        auto* pool = db->get_context()->get_memory_pool();
        // DEBUG :
        //     printf("[TABLE LOAD] %s: arrow_allocated=%.2f GiB\n",
        //            tables[i].c_str(), pool->bytes_allocated() / (1024.0*1024.0*1024.0));
    }
}

void move_tables_to_gpu(const std::shared_ptr<maximus::Database>& db,
                        const std::vector<std::string>& tables,
                        std::shared_ptr<maximus::MaximusContext> context,
                        const std::map<std::string, std::string>& columns_to_drop = {}) {
    
    std::cout << "--------------------------------------------------\n";
    std::cout << "Moving tables to GPU: " << tables.size() << " tables\n";
    std::cout << "--------------------------------------------------\n";

    for (const auto& t_name : tables) {
        // Check if table exists in DB (might not be loaded if not required)
        auto device_table = db->get_table(t_name);
        if (device_table.is_none()) {
            continue;
        }

        std::string status_msg = "Table '" + t_name + "': ";

        // It should be on CPU (Arrow Table) if we intend to move it
        if (device_table.as_table()) { 
            auto cpu_table = device_table.as_table();
            
            // 1. Drop column if specified
            if (columns_to_drop.find(t_name) != columns_to_drop.end()) {
                std::string col_to_drop = columns_to_drop.at(t_name);
                auto schema = cpu_table->get_schema();
                int col_idx = schema->get_schema()->GetFieldIndex(col_to_drop);
                
                if (col_idx != -1) {
                    cpu_table->remove_column(col_idx);
                    status_msg += "Dropped '" + col_to_drop + "', ";
                }
            }

            // 2. Move to GPU
            auto new_schema = cpu_table->get_schema();
            device_table.to_gpu(context, new_schema);
            
            // Update the database with the new GPU table and schema
            db->set_table(t_name, device_table, new_schema);
            status_msg += "Moved to GPU.";
        } else if (device_table.is_gtable()) {
             status_msg += "Already on GPU.";
        } else {
             status_msg += "Unknown state (not CPU or GPU table?).";
        }
        std::cout << status_msg << "\n";
    }
    std::cout << "--------------------------------------------------\n";
}

bool verify_table_locations(const std::shared_ptr<maximus::Database>& db,
                            const std::vector<std::string>& tables,
                            const maximus::DeviceType& expected_device) {
    std::cout << "===================================" << std::endl;
    std::cout << "      STORAGE VERIFICATION         " << std::endl;
    std::cout << "===================================" << std::endl;
    std::cout << "Expected Device: " << maximus::device_type_to_string(expected_device) << "\n";
    
    bool all_match = true;
    for (const auto& t_name : tables) {
        auto tbl = db->get_table_nocopy(t_name);
        if (tbl.is_none()) continue; // Skip tables not loaded

        std::string loc = "Unknown";
        bool match = false;

        if (tbl.on_cpu()) {
            loc = "CPU";
            if (expected_device == maximus::DeviceType::CPU) match = true;
        } else if (tbl.on_gpu()) {
            loc = "GPU";
            if (expected_device == maximus::DeviceType::GPU) match = true;
        }

        if (!match) all_match = false;

        std::cout << "Table " << std::left << std::setw(20) << t_name << ": " << loc;
        if (!match) std::cout << " [MISMATCH]";

        // Print rows/cols for context
        if (tbl.is_table()) std::cout << " (Rows: " << tbl.as_table()->num_rows() << ", Cols: " << tbl.as_table()->num_columns() << ")";
        if (tbl.is_gtable()) std::cout << " (Rows: " << tbl.as_gtable()->get_num_rows() << ", Cols: " << tbl.as_gtable()->get_num_columns() << ")";
        std::cout << "\n";
    }
    
    std::cout << "Tables Status: " << (all_match ? "SUCCESS" : "WARNING: Location Mismatch") << "\n";
    if (!all_match) {
        std::cout << "[WARNING] Some tables are not on the expected storage device!\n";
    }
    return all_match;
}

bool verify_index_locations(const maximus::IndexMap& indexes,
                            const maximus::DeviceType& expected_device) {
    if (indexes.empty()) return true;
    std::cout << "Expected Device: " << maximus::device_type_to_string(expected_device) << "\n";
    
    bool all_match = true;
    for (const auto& [name, index] : indexes) {
        std::string loc = "Unknown";
        bool match = false;
        if (index->is_on_cpu()) {
            loc = "CPU";
            if (expected_device == maximus::DeviceType::CPU) match = true;
        } else if (index->is_on_gpu()) {
            loc = "GPU";
            if (expected_device == maximus::DeviceType::GPU) match = true;
        }

        if (!match) all_match = false;

        std::cout << "Index " << std::left << std::setw(20) << name << ": " << loc;
        if (!match) std::cout << " [MISMATCH]";
        std::cout << "\n";
    }
    
    std::cout << "Indexes Status: " << (all_match ? "SUCCESS" : "WARNING: Location Mismatch") << "\n";
    std::cout << "-----------------------------------\n";
    return all_match;
}

void print_timings(const std::string& csv_results, const std::string& filename) {
    std::ofstream file;
    file.open(filename);
    file << csv_results;
    file.close();
}

std::vector<std::string> get_table_names(const std::string& benchmark) {
    if (benchmark == "tpch") {
        return maximus::tpch::table_names();
    }
    if (benchmark == "clickbench") {
        return maximus::clickbench::table_names();
    }
    if (benchmark == "h2o") {
        return maximus::h2o::table_names();
    }
    if (benchmark == "vsds") {
        return maximus::vsds::table_names();
    }
    throw std::runtime_error("The benchmark argument not recognized. It can only take the values "
                             "{tpch, clickbench, h2o, vsds}");
}

std::vector<std::shared_ptr<maximus::Schema>> get_table_schemas(const std::string& benchmark) {
    if (benchmark == "tpch") {
        return maximus::tpch::schemas();
    }
    if (benchmark == "clickbench") {
        return maximus::clickbench::schemas();
    }
    if (benchmark == "h2o") {
        return maximus::h2o::schemas();
    }
    if (benchmark == "vsds") {
        return maximus::vsds::schemas();
    }
    throw std::runtime_error("The benchmark argument not recognized. It can only take the values "
                             "{tpch, clickbench, h2o, vsds}");
}

// Helper to get available queries and descriptions for a benchmark
std::vector<std::pair<std::string, std::string>> get_available_queries(const std::string& benchmark) {
    std::vector<std::pair<std::string, std::string>> queries;

    if (benchmark == "vsds") {
        return maximus::vsds::available_queries();
    } else {
        queries.push_back({"TBD", "No specific description"});
    }
    return queries;
}

// Unified get_query function for all benchmarks (including VSDS with indexes)
std::shared_ptr<maximus::QueryPlan> get_query(const std::string& query,
                                              std::shared_ptr<maximus::Database>& db,
                                              maximus::DeviceType device,
                                              const std::string& benchmark,
                                              const std::string& params = "",
                                              const maximus::IndexMap& indexes = {},
                                              const std::string& index_desc = "",
                                              std::optional<maximus::DeviceType> vs_device = std::nullopt) {
    if (benchmark == "tpch") {
        return maximus::tpch::query_plan(query, db, device);
    }
    if (benchmark == "clickbench") {
        return maximus::clickbench::query_plan(query, db, device);
    }
    if (benchmark == "h2o") {
        return maximus::h2o::query_plan(query, db, device);
    }
    if (benchmark == "vsds") {
        return maximus::vsds::query_plan(query, db, device, params, indexes, index_desc,
                                         vs_device.value_or(device));
    }
    throw std::runtime_error("The benchmark argument not recognized. It can only take the values "
                             "{tpch, clickbench, h2o, vsds}");
}

// Overload without indexes for non-VSDS benchmarks
std::shared_ptr<maximus::QueryPlan> get_query(const std::string& query,
                                              std::shared_ptr<maximus::Database>& db,
                                              const std::string& benchmark,
                                              const std::string& params = "") {
    return get_query(query, db, maximus::DeviceType::CPU, benchmark, params, {});
}

std::string uppercase(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
}

maximus::IndexMap build_indexes(const std::string& benchmark,
                               const std::string& index_desc,
                               const std::vector<std::string>& queries,
                               std::shared_ptr<maximus::Database> db,
                               std::shared_ptr<maximus::MaximusContext> context,
                               maximus::DeviceType storage_device,
                               maximus::DeviceType index_storage_device,
                               bool preload,
                               bool use_cache = false,
                               const std::string& cache_dir = "",
                               maximus::VectorDistanceMetric metric = maximus::VectorDistanceMetric::L2,
                               bool use_cuvs = true,
                               bool pin_index_on_cpu = false,
                               int index_data_on_gpu = -1,
                               int cagra_cache_graph = 0) {
    maximus::IndexMap indexes;

    if (benchmark == "vsds" && !index_desc.empty()) {
        // Collect required indexes from all queries
        std::set<std::string> needed_indexes;
        for (const auto& q : queries) {
            auto reqs = maximus::vsds::get_query_index_requirements(q);
            for (const auto& req : reqs) {
                needed_indexes.insert(req);
            }
        }

        if (!needed_indexes.empty()) {
            std::cout << "\n=== Building VSDS Indexes: " << index_desc << " ===\n";
            std::cout << "---> Required indexes: ";
            for (const auto& key : needed_indexes) {
                std::cout << key << " ";
            }
            std::cout << "\n";

            // Helper: split "table.column" -> (table, column)
            auto split_key = [](const std::string& key) -> std::pair<std::string, std::string> {
                auto dot = key.find('.');
                return {key.substr(0, dot), key.substr(dot + 1)};
            };

            auto build_and_prepare = [&](const std::string& table, const std::string& col) -> maximus::IndexPtr {
                if (!preload) {
                    db->load_table(table, maximus::vsds::schema(table), {}, storage_device);
                }
                auto data = db->get_table_nocopy(table);
                if (data.empty()) {
                    std::cout << "[WARNING] " << table << " table is empty, skipping index.\n";
                    return maximus::IndexPtr(nullptr);
                }

                // Resolve CAGRA dataset GPU copy during train().
                // Only copy when index_storage=gpu (data stays on GPU, no to_cpu() round-trip).
                // For index_storage=cpu: don't copy here (avoids OOM during to_cpu()).
                // The raft runtime flag in to_gpu() handles data=1 at query time instead. (from patch)
                bool do_gpu_copy = false;
                if (index_desc.find("GPU,Cagra") == 0) {
                    do_gpu_copy = (index_storage_device == maximus::DeviceType::GPU) &&
                                  (index_data_on_gpu != 0);
                }

                // print_mem("pre-index-build", context);

                auto idx = maximus::faiss::FaissIndex::build(
                    context, data, col, index_desc, metric, use_cache, cache_dir, use_cuvs,
                    do_gpu_copy, cagra_cache_graph);

                // print_mem("post-index-build", context);

                // Set index flags (survive to_cpu/to_gpu round-trip).
                // copy_data_to_gpu_: unified — controls data placement for CAGRA and IVF.
                //   CAGRA: controls raft ATS flag in to_gpu() for data=1.
                //   IVF: controls referenceFrom() vs copyInvertedListsFrom() in to_gpu().
                // cache_cagra_graph_: CAGRA-specific graph caching.
                auto* faiss_idx = dynamic_cast<maximus::faiss::FaissIndex*>(idx.get());
                if (faiss_idx) {
                    if (index_data_on_gpu == 1) faiss_idx->copy_data_to_gpu_ = true;
                    if (index_desc.find("GPU,Cagra") == 0 && cagra_cache_graph)
                        faiss_idx->cache_cagra_graph_ = true;
                }
                context->barrier();

                bool want_gpu_index = (index_storage_device == maximus::DeviceType::GPU);
                PE("setup_index_location");
                // print_mem("pre-index-move", context);
                if (idx->is_on_gpu() && !want_gpu_index) {
                    std::cout << "---> Setup Index: Moving " << table << " index from GPU to CPU\n";
                    idx = idx->to_cpu();
                } else if (!idx->is_on_gpu() && want_gpu_index) {
                    std::cout << "---> Setup Index: Moving " << table << " index from CPU to GPU\n";
                    idx = idx->to_gpu();
                }
                // print_mem("post-index-move", context);
                PL("setup_index_location");

                // Pin CPU index data if requested (for fast H2D transfer via DMA)
                if (pin_index_on_cpu && idx->is_on_cpu()) {
                    auto* faiss_idx = dynamic_cast<maximus::faiss::FaissIndex*>(idx.get());
                    if (faiss_idx) faiss_idx->pin();
                }
                std::cout << "---> " << table << " index now on: " << (idx->is_on_gpu() ? "GPU" : "CPU")
                          << "\n";
                return idx;
            };

            // Build only the required indexes
            for (const auto& key : needed_indexes) {
                auto [table, col] = split_key(key);
                auto start = std::chrono::high_resolution_clock::now();
                auto idx = build_and_prepare(table, col);
                // print_mem("post-build-and-prepare", context);
                auto end = std::chrono::high_resolution_clock::now();
                auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                std::cout << "---> " << key << " Index built in " << build_time << " ms\n";
                if (idx) {
                    indexes[key] = idx;
                }
            }
        } else {
            std::cout << "\n=== No indexes required for selected queries ===\n";
        }
    }
    return indexes;
}
