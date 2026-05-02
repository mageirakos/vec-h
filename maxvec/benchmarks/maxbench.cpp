#include <cxxopts.hpp>
#include <maximus/operators/acero/interop.hpp>
#include <maximus/tpch/tpch_queries.hpp>
#include <maximus/indexes/faiss/faiss_index.hpp>
#include <set>
#include <map>
#include <format>
#include <sstream>
#include <iomanip>

#include "utils.hpp"

int main(int argc, char** argv) {
    cxxopts::Options options("MAXIMUS BENCHMARKS (MAXBENCH)",
                             "Running benchmarks with Maximus and Apache Acero query engines.");
    auto adder = options.add_options()(
        "path", "Path to the CSV files", cxxopts::value<std::string>()->default_value(csv_path()))(
        "engines",
        "Engines to run the queries with. Options: maximus, acero and maximus,acero (to run both).",
        cxxopts::value<std::string>()->default_value({"maximus"}))(
        "r,n_reps", "Number of repetitions", cxxopts::value<int>()->default_value("1"))(
        "benchmark",
        "which benchmark to run? (tpch, h2o, clickbench, vsds)",
        cxxopts::value<std::string>()->default_value("tpch"))(
        "params",
        "Benchmark-specific parameters as key=value pairs separated by commas (e.g., "
        "k=10,hnsw_efsearch=100,ivf_nprobe=10)",
        cxxopts::value<std::string>()->default_value(""))(
        "q,queries",
        "name of the query in the benchmark",
        cxxopts::value<std::vector<std::string>>()->default_value({"q1"}))(
        "d,device",
        "Device to run the queries on. Options: cpu, gpu",
        cxxopts::value<std::string>()->default_value("cpu"))(
        "b,csv_batch_size",
        "Batch size (num. of rows) for reading CSV files. Options: e.g. 2^20, 2^30, max. If max "
        "chosen, all tables will be repackaged to a single chunk.",
        cxxopts::value<std::string>()->default_value("2^30"))(
        "s,storage_device",
        "Device where the tables are initially residing. Options: cpu, cpu-pinned, gpu",
        cxxopts::value<std::string>()->default_value("cpu"))(
        "is,index_storage_device",
        "Device where the indexes are stored. Options: cpu, cpu-pinned, gpu (defaults to storage_device)",
        cxxopts::value<std::string>()->default_value(""))(
        "vs_device",
        "Device to run vector-search operators on. Defaults to --device. Options: cpu, gpu",
        cxxopts::value<std::string>()->default_value(""))(
        "n_reps_storage",
        "How many repetitions to run for loading the tables. This is useful for benchmarking I/O.",
        cxxopts::value<int>()->default_value("1"))(
        "persist_results",
        "Whether to write the resulting table to a csv file. No = Do not write, Any other value = "
        "Do write.",
        cxxopts::value<std::string>()->default_value("no"))(
        "p,profile",
        "Profiling options: see Caliper -P command-line arguments.",
        // cxxopts::value<std::string>()->default_value("runtime-report,calc.inclusive,mem.highwatermark,sample-report,cuda-activity-report"))(
        // cxxopts::value<std::string>()->default_value("runtime-report,calc.inclusive"))(
        cxxopts::value<std::string>()->default_value(
            "runtime-report(calc.inclusive=true,output=stdout),event-trace"))(
        "i,index",
        "FAISS index description for VSDS benchmark (e.g., Flat, HNSW32, IVF256,Flat, GPU,Cagra). "
        "Leave empty for ENN queries that don't need an index.",
        cxxopts::value<std::string>()->default_value(""))(
        "preload",
        "Whether to preload tables into memory",
        cxxopts::value<bool>()->default_value("true"))(
        "list_queries",
        "List available queries for the selected benchmark",
        cxxopts::value<bool>()->default_value("false"))(
        "out_file",
        "Full path to output CSV result file (without extension, engine name will be prepended)",
        cxxopts::value<std::string>()->default_value(""))(
        "using_large_list",
        "Whether to use the CPU->GPU load workaround for large list columns (VSDS benchmark only, due to int32 limitation of cuDF)",
        cxxopts::value<bool>()->default_value("false"))(
        "use_index_cache_dir",
        "Directory path to store the FAISS index cache files. If provided, caching is enabled (VSDS benchmark only).",
        cxxopts::value<std::string>()->default_value(""))(
        "limit_rows",
        "Number of rows in the restult to print for each query",
        cxxopts::value<int>()->default_value("50"))(
        "vary_batch_sizes",
        "Comma-separated batch sizes for vary-batch mode (e.g., '1,10,100,1000'). "
        "When set, maxbench loops over batch sizes internally, avoiding repeated data loading. "
        "If total_queries=0 (default), each batch size runs once starting at query_start=0. "
        "If total_queries=N, all windows [0,N) are swept in steps of batch_size.",
        cxxopts::value<std::string>()->default_value(""))(
        "total_queries",
        "Total queries to cover in vary-batch mode. 0 (default) = run once at query_start=0 "
        "per batch size; N>0 = sweep all windows from 0 to N.",
        cxxopts::value<int>()->default_value("0"))(
        "flush_cache",
        "Flush CPU L3 and GPU L2 caches before each benchmark repetition to ensure clean cache state. "
        "Adds ~10-50ms overhead per rep but prevents cache warm-up from distorting latency measurements.",
        cxxopts::value<bool>()->default_value("false"))(
        "case6_persist_gpu_index",
        "Persist indexes on GPU across all vary_batch iterations (CASE 6). "
        "After build_indexes() and before the vary_batch loop, performs a one-time to_gpu() "
        "swap on every CPU-resident entry in the IndexMap. Wrapped in the 'setup_index_movement' "
        "Caliper region so it's unambiguously attributable vs. the per-query 'index_movement' region.",
        cxxopts::value<bool>()->default_value("false"))(
        "h,help", "Print help");

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    auto benchmark = result["benchmark"].as<std::string>();

    if (result.count("list_queries") && result["list_queries"].as<bool>()) {
        std::cout << "Available queries for benchmark '" << benchmark << "':\n";
        auto queries = get_available_queries(benchmark);
        for (const auto& q : queries) {
            std::cout << "  - " << std::left << std::setw(20) << q.first << ": " << q.second << "\n";
        }
        return 0;
    }

    // choose the base path where all the tables are stored
    auto path = result["path"].as<std::string>();
    // which engines to run the queries with
    auto engines = result["engines"].as<std::string>();
    // how many repetitions to run
    auto n_reps = result["n_reps"].as<int>();
    // which benchmark to run
    // auto benchmark = result["benchmark"].as<std::string>(); // Already declared above
    // benchmark-specific parameters
    auto params = result["params"].as<std::string>();
    // which queries to run
    auto queries = result["queries"].as<std::vector<std::string>>();
    // which device to run the queries on
    auto device_string = result["device"].as<std::string>();
    // the device where the tables are stored
    auto storage_device_string = result["storage_device"].as<std::string>();
    // the device where the indexes are stored
    auto index_storage_device_string = result["index_storage_device"].as<std::string>();
    if (index_storage_device_string.empty()) {
        index_storage_device_string = storage_device_string;
    }
    // the device to run vector-search operators on (defaults to --device)
    auto vs_device_string = result["vs_device"].as<std::string>();
    // how many times to load tables
    auto n_reps_storage = result["n_reps_storage"].as<int>();
    // whether to enable profiling
    auto profile = result["profile"].as<std::string>();
    // the batch size for reading CSV files
    auto csv_batch_size_string = result["csv_batch_size"].as<std::string>();
    // Whether to persist the data
    auto persist_results = result["persist_results"].as<std::string>();
    auto out_file = result["out_file"].as<std::string>();
    // Whether to preload tables
    auto preload = result["preload"].as<bool>();
    // Index description for VSDS benchmark (e.g., Flat, HNSW32, IVF256,Flat, GPU,Cagra)
    auto index_desc = result["index"].as<std::string>();
    // Whether to use large list GPU workaround
    auto using_large_list = result["using_large_list"].as<bool>();
    // FAISS Index Cache config
    auto use_index_cache_dir = result["use_index_cache_dir"].as<std::string>();
    bool use_index_cache = !use_index_cache_dir.empty();
    // Number of rows to print for each query
    auto result_limit_rows = result["limit_rows"].as<int>();
    // Vary-batch mode parameters
    auto case6_persist_gpu_index = result["case6_persist_gpu_index"].as<bool>();
    auto vary_batch_sizes_str = result["vary_batch_sizes"].as<std::string>();
    auto total_queries = result["total_queries"].as<int>();
    // Cache flush
    auto flush_cache = result["flush_cache"].as<bool>();

    // Parse batch sizes
    std::vector<int> vary_batch_sizes;
    if (!vary_batch_sizes_str.empty()) {
        std::istringstream bs_stream(vary_batch_sizes_str);
        std::string bs_token;
        while (std::getline(bs_stream, bs_token, ',')) {
            vary_batch_sizes.push_back(std::stoi(bs_token));
        }
    }
    // Vary-batch mode is active whenever batch sizes are provided.
    // total_queries=0 (default): run once per batch size at qstart=0.
    // total_queries=N: sweep all windows [0,N) in steps of batch_size.
    bool vary_batch_mode = !vary_batch_sizes.empty();

    // Parse params early to extract metric and GPU flags for index building (VSDS only)
    auto metric = maximus::VectorDistanceMetric::L2;  // default for non-VSDS
    bool use_cuvs = true;  // default: use cuVS for GPU indexes
    int index_data_on_gpu = -1;  // -1=auto, 0=host view, 1=GPU copy (CAGRA + IVF)
    int cagra_cache_graph = 0;  // 0=normal copyFrom_ex, 1=fast cached graph path
    if (benchmark == "vsds" && !params.empty()) {
        auto params_struct = maximus::vsds::parse_query_parameters(params);
        metric = params_struct.metric;
        use_cuvs = params_struct.use_cuvs;
        index_data_on_gpu = params_struct.index_data_on_gpu;
        cagra_cache_graph = params_struct.cagra_cache_graph;
    }

    // initialize the profiler if compiled with profiling enabled
    PROFILER_INIT(mgr, profile);
    PROFILER_START(mgr);
    PE("benchmark_setup");

    print_gpu_mem("script-start");

    // create a database catalogue and a database connection
    auto context = maximus::make_context();
    print_mem("post-context-init", context);

    auto device         = maximus::DeviceType::CPU;
    auto storage_device = maximus::DeviceType::CPU;
    auto index_storage_device = maximus::DeviceType::CPU;
    if (device_string == "gpu") {
        device = maximus::DeviceType::GPU;
    }
    if (storage_device_string == "gpu") {
        storage_device = maximus::DeviceType::GPU;
    }
    if (index_storage_device_string == "gpu") {
        index_storage_device = maximus::DeviceType::GPU;
    }
    auto vs_device = device;
    if (vs_device_string == "cpu") {
        vs_device = maximus::DeviceType::CPU;
    } else if (vs_device_string == "gpu") {
        vs_device = maximus::DeviceType::GPU;
    }
    bool pin_index_on_cpu = (index_storage_device_string == "cpu-pinned");
    if (storage_device_string == "cpu-pinned") {
        context->tables_initially_pinned = true;
    }

    if (csv_batch_size_string == "max") {
        context->csv_batch_size                   = 1 << 30;
        context->tables_initially_as_single_chunk = true;
    } else {
        context->csv_batch_size = maximus::get_value<int32_t>(csv_batch_size_string, 1 << 30);
    }

    auto db_catalogue = maximus::make_catalogue(path);
    auto db           = maximus::make_database(db_catalogue, context);

    auto acero_engine                                           = maximus::AceroExecutor(context->n_inner_threads);
    std::vector<std::string> tables                             = get_table_names(benchmark);
    std::vector<std::shared_ptr<maximus::Schema>> table_schemas = get_table_schemas(benchmark);

    // print_mem("pre-table-load", context);

    if (preload) {
        // preload all the tables, if you don't want to include I/O in the benchmarks
        std::vector<int64_t> timings_io(n_reps_storage, 0);
        for (int i = 0; i < n_reps_storage; ++i) {
            context->barrier();
            auto start = std::chrono::high_resolution_clock::now();
            
            // Manually force 'reviews' and 'images' to load on CPU for VSDS benchmark
            // This is to avoid cuDF list column size limits during Parquet read.
            std::vector<std::string> force_cpu_tables;
            if (benchmark == "vsds" && storage_device == maximus::DeviceType::GPU && using_large_list) {
                force_cpu_tables = {"reviews"}; // add images if you need large_list
            }

            load_tables(db, tables, table_schemas, storage_device, force_cpu_tables);

            context->barrier();
            auto end = std::chrono::high_resolution_clock::now();
            timings_io[i] =
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        }
        // print current time:
        auto now = std::chrono::system_clock::now();
        std::cout << "\nCurrent time: " << std::format("{:%F %T}", now) << "\n"; 
        std::cout << "===================================" << std::endl;
        std::cout << "          LOADING TABLES           " << std::endl;
        std::cout << "===================================" << std::endl;
        std::cout << "Loading tables to:                   " << storage_device_string << "\n";
        std::cout << "Loading times over repetitions [ms]: ";
        for (int i = 0; i < n_reps_storage; ++i) {
            std::cout << timings_io[i] << ",\t";
        }
        std::cout << "\n";
    }

    // print_mem("post-table-load", context);

    std::cout << "===================================" << std::endl;
    std::cout << "    MAXBENCH " << uppercase(benchmark) << " BENCHMARK:    " << std::endl;
    std::cout << "===================================" << std::endl;
    std::cout << "---> benchmark:                " << uppercase(benchmark) << "\n";
    std::cout << "---> queries:                  " << to_string(queries) << "\n";
    std::cout << "---> Tables path:              " << path << "\n";
    std::cout << "---> Engines:                  " << engines << "\n";
    std::cout << "---> Number of reps:           " << n_reps << "\n";
    std::cout << "---> Device:                   " << device_string << "\n";
    std::cout << "---> Storage Device:           " << storage_device_string << "\n";
    std::cout << "---> Index Storage Device:     " << index_storage_device_string << "\n";
    std::cout << "---> Num. outer threads:       " << context->n_outer_threads << "\n";
    std::cout << "---> Num. inner threads:       " << context->n_inner_threads << "\n";
    std::cout << "---> Num. IO threads:          " << context->n_io_threads << "\n";
    std::cout << "---> Operators Fusion:         " << (context->fusing_enabled ? "ON" : "OFF")
              << "\n";
    std::cout << "---> CSV Batch Size (string):  " << csv_batch_size_string << "\n";
    std::cout << "---> CSV Batch (number):       " << context->csv_batch_size << "\n";
    std::cout << "---> Tables initially pinned:  "
              << (context->tables_initially_pinned ? "YES" : "NO") << "\n";
    std::cout << "---> Tables as single chunk:   "
              << (context->tables_initially_as_single_chunk ? "YES" : "NO") << "\n";
    std::cout << "---> Flush cache between reps: " << (flush_cache ? "YES" : "NO") << "\n";
    if (benchmark == "vsds") {

        std::cout << "===================================" << std::endl;
        std::cout << "          INDEX  " << std::endl;
        std::cout << "===================================" << std::endl;
        std::cout << "---> VSDS Index:               " << index_desc << "\n";
        if (!params.empty()) {
            // parse and print parameters
            auto params_struct = maximus::vsds::parse_query_parameters(params);
            std::cout << "[VSDS Parameters]\n";
            std::cout << "---> faiss_index:              " << index_desc << "\n";
            std::cout << "---> k:                        " << params_struct.k << "\n";
            std::cout << "---> hnsw_efsearch:            " << params_struct.hnsw_efsearch << "\n";
            std::cout << "---> cagra_itopksize:          " << params_struct.cagra_itopksize << "\n";
            std::cout << "---> cagra_searchwidth:        " << params_struct.cagra_searchwidth << "\n";
            std::cout << "---> ivf_nprobe:               " << params_struct.ivf_nprobe << "\n";
            std::cout << "---> postfilter_ksearch:       " << params_struct.postfilter_ksearch << "\n";
            std::cout << "---> metric:                   " << (params_struct.metric == maximus::VectorDistanceMetric::INNER_PRODUCT ? "IP" : "L2") << "\n";
            std::cout << "---> query_count:              " << params_struct.query_count << "\n";
            std::cout << "---> query_start (if qcnt>0):  " << params_struct.query_start << "\n";
            std::cout << "---> use_cuvs:                 " << (params_struct.use_cuvs ? "true" : "false") << "\n";
            std::cout << "---> use_post:                 " << (params_struct.use_post ? "true" : "false") << "\n";
            std::cout << "---> use_limit_per_group:      " << (params_struct.use_limit_per_group ? "true" : "false") << "\n";
            bool is_cagra = index_desc.find("Cagra") != std::string::npos;
            bool is_ivf_flat = index_desc.find("IVF") != std::string::npos
                            && index_desc.find("Flat") != std::string::npos;
            if (is_cagra || is_ivf_flat) {
                std::cout << "---> OPT-H (host view):        " << (params_struct.index_data_on_gpu == 0 ? "ON" : "OFF") << "\n";
            }
            if (is_cagra) {
                std::cout << "---> OPT-C (cache graph):      " << (params_struct.cagra_cache_graph == 1 ? "ON" : "OFF") << "\n";
            }
            std::cout << "---> use_index_cache:          " << (use_index_cache ? "true" : "false") << "\n";
            if (params_struct.incr_step > 0) {
                std::cout << "---> incr_step:                " << params_struct.incr_step << "\n";
                std::cout << "[WARNING] incr_step is active:\n"
                          << "          - Rep 0 (warmup) and rep 1 (first hot) both use query_start="
                          << params_struct.query_start << "\n"
                          << "          - Rep 2+ advance by " << params_struct.incr_step << " per rep.\n"
                          << "          - Printed/saved result table is from the LAST rep's query.\n";
            }
        }
    }


    std::vector<std::vector<int64_t>> schedule_timings_maximus(queries.size(),
                                                               std::vector<int64_t>(n_reps, 0));
    std::vector<std::vector<int64_t>> query_timings_maximus(queries.size(),
                                                            std::vector<int64_t>(n_reps, 0));
    std::vector<std::vector<int64_t>> schedule_timings_acero(queries.size(),
                                                             std::vector<int64_t>(n_reps, 0));
    std::vector<std::vector<int64_t>> query_timings_acero(queries.size(),
                                                          std::vector<int64_t>(n_reps, 0));

    std::vector<maximus::TablePtr> maximus_result_tables(queries.size());
    std::vector<maximus::TablePtr> acero_result_tables(queries.size());

    int query_idx = 0;

    // Build VSDS indexes based on query requirements
    maximus::IndexMap indexes = build_indexes(benchmark, index_desc, queries, db, context, storage_device, index_storage_device, preload, use_index_cache, use_index_cache_dir, metric, use_cuvs, pin_index_on_cpu, index_data_on_gpu, cagra_cache_graph);

    // VSDS Post-Index Cleanup:
    // As workaround to the current cuDF int32 limitation, if running on GPU, we loaded 'reviews'/'images' on CPU to build the index.
    // Now we must:
    // 1. Drop the large embedding column (index owns the data now). So that cuDF doesn't overflow.
    // 2. Move the remaining table to GPU. So that it is available for the query, in the location it is expected (storage_device)
    if (benchmark == "vsds" && storage_device == maximus::DeviceType::GPU && using_large_list) {
        // Tables that were forced to CPU
        // NOTE: depending on SF we might need to add "images" to large list --> requires chancing schema in vsds_queries
        std::vector<std::string> vsds_large_tables = {"reviews"};
        
        // Columns to drop before moving to GPU
        std::map<std::string, std::string> columns_to_drop = {
            {"reviews", "rv_embedding"},
            // {"images", "i_embedding"}
        };

        move_tables_to_gpu(db, vsds_large_tables, context, columns_to_drop);
    }

    // Final Verification
    auto time_start = std::chrono::high_resolution_clock::now();
    verify_table_locations(db, tables, storage_device);
    verify_index_locations(indexes, index_storage_device);
    auto time_end = std::chrono::high_resolution_clock::now();
    auto verification_time = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    std::cout << "Storage verification time: " << verification_time << " ms\n";

    // NOTE: use $nvidia-smi dmon -s u -d 0.1 -i 0 to monitor GPU utilization during execution (but 0.1 is 100ms so might miss short queries, cant poll too much either)
    print_mem("post-setup", context);
    PL("benchmark_setup");

    // CASE 6: persist every CPU-resident index on the GPU for the entire vary_batch loop.
    // Distinct region name so Caliper aggregation cannot conflate this one-time setup move
    // with the per-query 'index_movement' region emitted by join_indexed_operator.
    if (case6_persist_gpu_index) {
        PE("setup_index_movement");
        for (auto& kv : indexes) {
            if (!kv.second || !kv.second->is_on_cpu()) continue;
            auto* faiss_idx = dynamic_cast<maximus::faiss::FaissIndex*>(kv.second.get());
            if (!faiss_idx) {
                std::cout << "[CASE 6] Skipping non-Faiss index: " << kv.first << "\n";
                continue;
            }
            std::cout << "[CASE 6] Persisting index to GPU: " << kv.first << "\n";
            kv.second = faiss_idx->to_gpu();
        }
        PL("setup_index_movement");
    }

    if (vary_batch_mode && benchmark == "vsds") {
        // =====================================================================
        // VARY-BATCH MODE: Load data/build index once, loop over batches (VSDS only for now)
        // =====================================================================
        std::cout << "\n=== VARY-BATCH MODE ===" << std::endl;
        std::cout << "---> Batch sizes:    " << vary_batch_sizes_str << std::endl;
        std::cout << "---> Total queries:  " << (total_queries > 0 ? std::to_string(total_queries) : "0 (single run at qstart=0)") << std::endl;
        std::cout << "[NOTE] chrono 'timings [ms]' below include Executor::schedule + final "
                     "barrier (~1-2ms framework overhead). For pure operator pipeline time "
                     "use Caliper 'Executor::execute' inclusive." << std::endl;

        std::string base_params = strip_keys(params, {"query_count", "query_start"});

        // Global timing accumulators: batch_size -> total ms
        std::map<int, double> batch_size_totals;
        std::map<int, int> batch_size_counts; // number of (qstart x query x rep) executions
        std::vector<std::vector<int64_t>> query_timings_varbatch(queries.size());

        PE("vary_batch");

        for (int batch_size : vary_batch_sizes) {
            if (total_queries > 0 && batch_size > total_queries) {
                std::cout << "Skipping batch_size=" << batch_size
                          << " (> total_queries=" << total_queries << ")\n";
                continue;
            }

            PE("batch_size_" + std::to_string(batch_size));
            auto bs_start = std::chrono::high_resolution_clock::now();

            // total_queries=0: one iteration at qstart=0; total_queries=N: full sweep
            int qstart_limit = total_queries > 0 ? total_queries : batch_size;
            for (int qstart = 0; qstart < qstart_limit; qstart += batch_size) {
                std::string batch_params = base_params
                    + ",query_count=" + std::to_string(batch_size)
                    + ",query_start=" + std::to_string(qstart);

                for (int qi = 0; qi < (int)queries.size(); qi++) {
                    const auto& query = queries[qi];
                    std::string batch_tag = query + "_batch_" + std::to_string(batch_size)
                                          + "_qstart_" + std::to_string(qstart);

                    PE(batch_tag);

                    std::vector<double> rep_timings(n_reps, 0.0);

                    for (int i = 0; i < n_reps; ++i) {
                        auto plan = get_query(query, db, device, benchmark, batch_params, indexes, index_desc, vs_device);

                        PE("Repetition_" + std::to_string(i));
                        PE("flush_cache");
                        flush_caches(flush_cache, device);
                        PL("flush_cache");
                        context->barrier();
                        auto start = std::chrono::high_resolution_clock::now();
                        db->schedule(plan);
                        auto exec_result = db->execute();
                        context->barrier();
                        auto end = std::chrono::high_resolution_clock::now();
                        rep_timings[i] = std::chrono::duration<double, std::milli>(end - start).count();
                        PL("Repetition_" + std::to_string(i));

                        // Accumulate
                        batch_size_totals[batch_size] += rep_timings[i];
                        batch_size_counts[batch_size]++;
                        query_timings_varbatch[qi].push_back(
                            static_cast<int64_t>(rep_timings[i] + 0.5));

                        // Persist CSV for last rep if requested
                        if (i == n_reps - 1 && persist_results != "no" && !out_file.empty()) {
                            auto batch_out = out_file;
                            auto dot = batch_out.rfind('.');
                            if (dot != std::string::npos) {
                                batch_out = batch_out.substr(0, dot)
                                          + "_batch_" + std::to_string(batch_size)
                                          + "_qstart_" + std::to_string(qstart)
                                          + batch_out.substr(dot);
                            }
                            write_result_to_csv(context, engines, "maximus", batch_out, exec_result[0]);
                        }
                    }

                    PL(batch_tag);

                    // Print per-batch timing
                    std::cout << batch_tag << " timings [ms]: ";
                    std::cout << std::fixed << std::setprecision(3);
                    for (auto t : rep_timings) std::cout << t << ", ";
                    std::cout << std::defaultfloat;
                    std::cout << "\n";
                }
            }

            auto bs_end = std::chrono::high_resolution_clock::now();
            auto bs_wall = std::chrono::duration_cast<std::chrono::milliseconds>(bs_end - bs_start).count();
            PL("batch_size_" + std::to_string(batch_size));
            std::cout << "\n--- batch_size=" << batch_size << " wall clock: " << bs_wall << " ms ---\n";
        }

        PL("vary_batch");

        // Print global summary
        std::cout << "\n===================================\n";
        std::cout << "      VARY-BATCH SUMMARY\n";
        std::cout << "===================================\n";
        for (auto& [bs, total] : batch_size_totals) {
            int n_batches = total_queries / bs;
            std::cout << "batch_size=" << bs
                      << "\t: total=" << std::fixed << std::setprecision(1) << total << " ms"
                      << " (" << n_batches << " batches x "
                      << queries.size() << " queries x "
                      << n_reps << " reps = "
                      << batch_size_counts[bs] << " executions)\n"
                      << "    avg per execution: "
                      << std::fixed << std::setprecision(1)
                      << (double)total / batch_size_counts[bs] << " ms\n";
        }

        // bs1_fullsweep only: single bs → one flat stats line per query.
        if (case6_persist_gpu_index && vary_batch_sizes.size() == 1) {
            auto vb_stats = timing_stats(query_timings_varbatch, queries, "maximus", "query", device);
            std::cout << "\n===================================\n";
            std::cout << "              TIMINGS              \n";
            std::cout << "===================================\n";
            for (size_t qi = 0; qi < queries.size(); ++qi) {
                if (query_timings_varbatch[qi].empty()) continue;
                std::cout << "Query: " << queries[qi] << "\n";
                std::cout << "- MAXIMUS TIMINGS [ms]: " << vb_stats.flattened[qi] << "\n";
                std::cout << "- MAXIMUS STATS: MIN = " << vb_stats.min[qi]
                          << " ms; \tMAX = " << vb_stats.max[qi]
                          << " ms; \tAVG = " << vb_stats.avg[qi] << " ms\n";
            }
        }
    } else {
    // =====================================================================
    //                          NORMAL MODE
    // =====================================================================

    // incr_step: VSDS normal mode only.
    // Reps 0 and 1 use initial query_start; rep i>=2 uses query_start + (i-1)*step.
    int     incr_step_val   = 0;
    int64_t incr_base_start = 0;
    std::string base_params_incr = params;
    if (benchmark == "vsds" && !params.empty()) {
        auto pq = maximus::vsds::parse_query_parameters(params);
        if (pq.incr_step > 0) {
            incr_step_val   = pq.incr_step;
            incr_base_start = pq.query_start;
            base_params_incr = strip_keys(params, {"query_start", "incr_step"});
        }
    }

    for (const auto& query : queries) {
        // PE(query);
        std::shared_ptr<maximus::QueryPlan> query_plan_acero   = nullptr;
        std::shared_ptr<maximus::QueryPlan> query_plan_maximus = nullptr;

        if (maximus::contains(engines, "acero")) {
            query_plan_acero = get_query(query, db, benchmark, params);
        }
        if (maximus::contains(engines, "maximus")) {
            // Pass index_desc directly to avoid comma parsing issues
            query_plan_maximus = get_query(query, db, device, benchmark, params, indexes, index_desc, vs_device);
        }

        std::cout << "===================================" << std::endl;
        std::cout << "            QUERY " << query << std::endl;
        std::cout << "===================================" << std::endl;
        std::cout << "---> query: " << query << "\n";
        std::cout << "---> Query Plan: \n";
        if (query_plan_maximus) {
            std::cout << query_plan_maximus->to_string() << "\n";
        } else if (query_plan_acero) {
            std::cout << query_plan_acero->to_string() << "\n";
        }

        PE(query);
        for (int i = 0; i < n_reps; ++i) {
            // Per-rep params: advance query_start for hot reps when incr_step is set.
            // i=0: warmup (initial); i=1: first hot (initial); i>=2: initial + (i-1)*step.
            std::string iter_params = params;
            if (incr_step_val > 0 && benchmark == "vsds") {
                int64_t qs = (i <= 1) ? incr_base_start
                                      : incr_base_start + static_cast<int64_t>(i - 1) * incr_step_val;
                iter_params = base_params_incr + ",query_start=" + std::to_string(qs);
                std::cout << "[incr_step] Rep " << i << ": query_start=" << qs << "\n";
            }

            // recreate the query plans as table sources might contain dangling pointers
            // since the tables have been exported out of the source operators
            if (maximus::contains(engines, "acero")) {
                query_plan_acero = get_query(query, db, benchmark, iter_params);
            }
            if (maximus::contains(engines, "maximus")) {
                // Pass index_desc directly to avoid comma parsing issues
                query_plan_maximus = get_query(query, db, device, benchmark, iter_params, indexes, index_desc, vs_device);
            }

            PE("Repetition_" + std::to_string(i));
            PE("flush_cache");
            flush_caches(flush_cache, device);
            PL("flush_cache");
            // define the variables
            auto start_time = std::chrono::high_resolution_clock::now();
            auto end_time   = std::chrono::high_resolution_clock::now();

            std::vector<maximus::TablePtr> result;

            // std::cout << "Running acero" << std::endl;
            if (maximus::contains(engines, "acero")) {
                if (benchmark == "tpch" && (query == "q2" || query == "q20")) {
                    // if (i > 0) {
                    query_timings_acero[query_idx][i] = -1;
                    // }
                    if (i == n_reps - 1) {
                        acero_result_tables[query_idx] = nullptr;
                    }
                } else {
                    start_time = std::chrono::high_resolution_clock::now();
                    acero_engine.schedule(query_plan_acero);
                    end_time = std::chrono::high_resolution_clock::now();
                    schedule_timings_acero[query_idx][i] =
                        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time)
                            .count();

                    context->barrier();
                    start_time = std::chrono::high_resolution_clock::now();
                    acero_engine.execute();
                    context->barrier();
                    result   = acero_engine.results();
                    end_time = std::chrono::high_resolution_clock::now();

                    // discard the first run as it is a warm-up
                    // if (i > 0) {
                    query_timings_acero[query_idx][i] =
                        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time)
                            .count();
                    // }

                    // if this is the last repetition, store the result
                    if (i == n_reps - 1) {
                        acero_result_tables[query_idx] = result[0];
                    }
                }
            }

            // std::cout << "Running maximus" << std::endl;
            if (maximus::contains(engines, "maximus")) {
                start_time = std::chrono::high_resolution_clock::now();
                db->schedule(query_plan_maximus);
                end_time = std::chrono::high_resolution_clock::now();
                schedule_timings_maximus[query_idx][i] =
                    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time)
                        .count();
                if (i == 0) {
                    std::cout << "Query Plan after scheduling: \n"
                              << query_plan_maximus->to_string() << "\n";
                }
                context->barrier();
                start_time = std::chrono::high_resolution_clock::now();
                result     = db->execute();
                context->barrier();
                end_time = std::chrono::high_resolution_clock::now();
                // discard the first run as it is a warm-up
                // if (i > 0) {
                query_timings_maximus[query_idx][i] =
                    std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time)
                        .count();
                // }
                // if this is the last repetition, store the result
                if (i == n_reps - 1) {
                    maximus_result_tables[query_idx] = result[0];
                }
            }
            PL("Repetition_" + std::to_string(i));
        }
        PL(query);

        std::cout << "===================================" << std::endl;
        std::cout << "              RESULTS              " << std::endl;
        std::cout << "===================================" << std::endl;
        if (incr_step_val > 0) {
            int64_t last_qs = (n_reps <= 1) ? incr_base_start
                                            : incr_base_start + static_cast<int64_t>(n_reps - 2) * incr_step_val;
            std::cout << "[WARNING] incr_step is active: result table below is from the LAST rep"
                      << " (query_start=" << last_qs << ").\n";
        }

        print_output_table(context, engines, "maximus", maximus_result_tables[query_idx], result_limit_rows);
        print_output_table(context, engines, "acero", acero_result_tables[query_idx], result_limit_rows);
        std::cout.flush();  // Flush stdout so output survives if persist crashes

        // Persist result of last iteration if requested
        if (persist_results != "no") {
            if (incr_step_val > 0) {
                int64_t last_qs = (n_reps <= 1) ? incr_base_start
                                                : incr_base_start + static_cast<int64_t>(n_reps - 2) * incr_step_val;
                std::cerr << "[incr_step] Saved CSV is from last rep (query_start=" << last_qs << ").\n";
            }
            if (!out_file.empty()) {
                std::cerr << "[persist] Writing results to CSV: " << out_file << std::endl;
                write_result_to_csv(context, engines, "maximus", out_file, maximus_result_tables[query_idx]);
                write_result_to_csv(context, engines, "acero", out_file, acero_result_tables[query_idx]);
            } else {
                // Fallback to old naming if no out_file specified
                write_result_to_file(context, engines, "maximus", device_string, query_idx, maximus_result_tables[query_idx]);
                write_result_to_file(context, engines, "acero", device_string, query_idx, acero_result_tables[query_idx]);
            }
        }

        context->barrier();

        std::cout << "===================================" << std::endl;
        std::cout << "              TIMINGS              " << std::endl;
        std::cout << "===================================" << std::endl;
        std::cout << "Execution times [ms]: \n";
        timing_stats maximus_stats;
        timing_stats acero_stats;

        if (maximus::contains(engines, "maximus")) {
            maximus_stats =
                timing_stats(query_timings_maximus, queries, "maximus", "query", device);
            std::cout << "- MAXIMUS TIMINGS [ms]: " << maximus_stats.flattened[query_idx]
                      << std::endl;
        }
        if (maximus::contains(engines, "acero")) {
            acero_stats = timing_stats(
                query_timings_acero, queries, "acero", "query", maximus::DeviceType::CPU);
            std::cout << "- ACERO TIMINGS   [ms]: " << acero_stats.flattened[query_idx] << std::endl
                      << std::endl;
        }
        std::cout << "Execution stats (min, max, avg): \n";

        if (maximus::contains(engines, "maximus")) {
            std::cout << "- MAXIMUS STATS: MIN = " << maximus_stats.min[query_idx]
                      << " ms; \tMAX = " << maximus_stats.max[query_idx]
                      << " ms; \tAVG = " << maximus_stats.avg[query_idx] << " ms" << "\n";
        }

        if (maximus::contains(engines, "acero")) {
            std::cout << "- ACERO STATS  : MIN = " << acero_stats.min[query_idx]
                      << " ms; \tMAX = " << acero_stats.max[query_idx]
                      << " ms; \tAVG = " << acero_stats.avg[query_idx] << " ms" << "\n\n";
        }

        // PL(query);
        ++query_idx;
        context->barrier();
    }

    std::cout << "===================================" << std::endl;
    std::cout << "        SUMMARIZED TIMINGS         " << std::endl;
    std::cout << "===================================" << std::endl;
    std::string filename = "./results.csv";
    std::stringstream csv_results_stream;

    if (maximus::contains(engines, "maximus")) {
        auto query_maximus_stats =
            timing_stats(query_timings_maximus, queries, "maximus", "query", device);
        auto schedule_maximus_stats =
            timing_stats(schedule_timings_maximus, queries, "maximus", "schedule", device);
        csv_results_stream << query_maximus_stats.csv_results;
        csv_results_stream << schedule_maximus_stats.csv_results;
    }

    if (maximus::contains(engines, "acero")) {
        auto query_acero_stats =
            timing_stats(query_timings_acero, queries, "acero", "query", maximus::DeviceType::CPU);
        auto schedule_acero_stats =
            timing_stats(schedule_timings_acero, queries, "acero", "schedule", device);
        csv_results_stream << query_acero_stats.csv_results;
        csv_results_stream << schedule_acero_stats.csv_results;
    }

    print_timings(csv_results_stream.str(), "results.csv");
    std::cout << "--->Results saved to " << filename << std::endl;
    std::cout << csv_results_stream.str() << std::endl;
    }

    print_mem("post-queries", context);

    PROFILER_FLUSH(mgr);

    return 0;
}
