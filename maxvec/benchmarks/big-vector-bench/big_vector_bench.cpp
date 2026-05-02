#include <cxxopts.hpp>

#include "../utils.hpp"
#include "queries.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
    cxxopts::Options options("Big Vector Bench Benchmarks");
    options.add_options()(
        "path",
        "Path to data",
        cxxopts::value<std::string>()->default_value(big_vector_bench_parquet_path()))(
        "r,n_reps", "Number of repetitions", cxxopts::value<int>()->default_value("1"))(
        "benchmark",
        "which benchmark to run? (ag_news-384-euclidean-filter, cc_news-384-euclidean-filter, "
        "app_reviews-384-euclidean-filter, amazon-384-euclidean-5filter, ag_news-384-euclidean)",
        cxxopts::value<std::string>()->default_value("ag_news-384-euclidean-filter"))(
        // Query parameters
        "impl, method",
        "Which approach/queryplan/implementation tu use for the workload",
        cxxopts::value<std::string>()->default_value("pre"))(
        "index", "Which index (if applicable)", cxxopts::value<std::string>()->default_value(""))(
        "hnsw_efsearch", "(if applicable)", cxxopts::value<int>()->default_value("0"))(
        "cagra_itopksize", "(if applicable)", cxxopts::value<int>()->default_value("0"))(
        "cagra_searchwidth", "(if applicable)", cxxopts::value<int>()->default_value("0"))(
        "ivf_nprobe", "(if applicable)", cxxopts::value<int>()->default_value("0"))(
        "postfilter_ksearch", "(if applicable)", cxxopts::value<int>()->default_value("100"))(
        // Auxiliary parameters
        "limit",
        "Limit the number of queries to execute (useful for debugging). 0 = Run All.",
        cxxopts::value<int>()->default_value("0"))(
        "save",
        "Whether to save benchmarking results. No = Do not write, Any other value = filename ",
        cxxopts::value<std::string>()->default_value("no"))(
        "title",
        "Title of the benchmark, used for saving results",
        cxxopts::value<std::string>()->default_value("benchmark"))(
        "persist_results",
        "Whether to write the resulting table to a csv file. No = Do not write, Any other value = "
        "Do write.",
        cxxopts::value<std::string>()->default_value("no"))(
        "p,profile",
        "Profiling options: see Caliper -P command-line arguments.",
        cxxopts::value<std::string>()->default_value(
            "runtime-report(calc.inclusive=true,output=stdout),event-trace"))("h,help",
                                                                              "Print help");

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    // choose the base path where all the tables are stored
    auto path = result["path"].as<std::string>();
    // how many repetitions to run
    auto n_reps = result["n_reps"].as<int>();
    // which benchmark to run
    auto benchmark = result["benchmark"].as<std::string>();
    // whether to enable profiling
    auto profile = result["profile"].as<std::string>();
    // Whether to persist the data
    auto persist_results = result["persist_results"].as<std::string>();
    // Whether to save results to file
    auto save_benchmarking_results = result["save"].as<std::string>();
    // Whether to save results to file
    auto title = result["title"].as<std::string>();
    // Limit queries
    auto limit_queries = result["limit"].as<int>();

    big_vector_bench::QueryParameters query_parameters;
    query_parameters.method             = result["method"].as<std::string>();
    query_parameters.faiss_index        = result["index"].as<std::string>();
    query_parameters.hnsw_efsearch      = result["hnsw_efsearch"].as<int>();
    query_parameters.postfilter_ksearch = result["postfilter_ksearch"].as<int>();
    query_parameters.cagra_itopksize    = result["cagra_itopksize"].as<int>();
    query_parameters.cagra_searchwidth  = result["cagra_searchwidth"].as<int>();
    query_parameters.ivf_nprobe         = result["ivf_nprobe"].as<int>();

    // initialize the profiler if compiled with profiling enabled
    PROFILER_INIT(mgr, profile);
    PROFILER_START(mgr);

    // create a database catalogue and a database connection
    auto context                              = maximus::make_context();
    context->tables_initially_pinned          = false;
    context->csv_batch_size                   = 1 << 30;
    context->tables_initially_as_single_chunk = false;
    context->fusing_enabled                   = false;
    auto db_catalogue                         = maximus::make_catalogue(path);
    auto db                                   = maximus::make_database(db_catalogue, context);

    std::cout << "===================================" << std::endl;
    std::cout << "    MAXBENCH " << benchmark << " BENCHMARK    " << std::endl;
    std::cout << "===================================" << std::endl;
    std::cout << "---> benchmark:                " << benchmark << "\n";
    std::cout << "---> implementation:           " << query_parameters.method << "\n";
    std::cout << "---> Tables path:              " << path << "\n";
    std::cout << "---> Number of reps:           " << n_reps << "\n";
    std::cout << "---> Num. outer threads:       " << context->n_outer_threads << "\n";
    std::cout << "---> Num. inner threads:       " << context->n_inner_threads << "\n";
    std::cout << "---> Operators Fusion:         " << (context->fusing_enabled ? "ON" : "OFF")
              << "\n";
    std::cout << "---> CSV Batch (number):       " << context->csv_batch_size << "\n";
    std::cout << "---> Tables initially pinned:  "
              << (context->tables_initially_pinned ? "YES" : "NO") << "\n";
    std::cout << "---> Tables as single chunk:   "
              << (context->tables_initially_as_single_chunk ? "YES" : "NO") << "\n";
    std::cout << "---> Inner Threads:   " << context->n_inner_threads << "\n";
    std::cout << "---> Outer Threads:   " << context->n_outer_threads << "\n";
    std::cout << "---> Query Limit:     "
              << (limit_queries > 0 ? std::to_string(limit_queries) : "ALL")
              << "\n";  // <--- ADDED LOG
#ifdef MAXIMUS_RELEASE_BUILD
    std::cout << "---> Build: RELEASE \n";
#else
    std::cout << "---> Build: DEBUG \n";
#endif

    std::cout << "===================================" << std::endl;
    std::cout << "          LOADING TABLES           " << std::endl;
    std::cout << "===================================" << std::endl;

    auto workload                   = big_vector_bench::get_workload(benchmark, db);
    std::vector<std::string> tables = workload->table_names();
    std::vector<std::shared_ptr<maximus::Schema>> table_schemas = workload->table_schemas();
    // Load tables using zero-copy variants for this benchmark
    assert(table_schemas.empty() || table_schemas.size() == tables.size());
    for (unsigned i = 0u; i < tables.size(); ++i) {
        db->load_table_nocopy(tables[i],
                              table_schemas.empty() ? nullptr : table_schemas[i],
                              {},
                              maximus::DeviceType::CPU);
    }

    // Print loaded tables
    std::vector<std::string> loaded_tables = db->get_table_names();
    for (const auto& table : loaded_tables) {
        std::cout << "Loaded table: " << table << "\n";
        db->get_table_nocopy(table).as_table()->slice(0, 3)->print();
    }

    std::cout << "===================================" << std::endl;
    std::cout << "            QUERY " << std::endl;
    std::cout << "===================================" << std::endl;

    std::vector<std::vector<int64_t>> timings_maximus;
    std::vector<maximus::TablePtr> result_tables;

    for (int i = 0; i < n_reps; ++i) {
        PE("Repetition_" + std::to_string(i));
        std::cout << "Repetition " << i << " of " << n_reps << std::endl;

        // Create query plans for the workload
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<std::shared_ptr<maximus::QueryPlan>> query_plans =
            workload->query_plans(query_parameters);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::cout
            << "---> Created " << query_plans.size() << " QueryPlans in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
            << "ms\n";
        if (i == 0) {
            std::cout << "---> Query Plan[0]: \n";
            std::cout << query_plans[0]->to_string() << "\n";
        }
        timings_maximus.push_back(std::vector<int64_t>(query_plans.size(), 0));
        result_tables.resize(query_plans.size());

        // Execute
        std::cout << "---> Executing Query Plans..." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();

        int exec_count = static_cast<int>(query_plans.size());
        if (limit_queries > 0 && limit_queries < exec_count) {
            exec_count = limit_queries;
            std::cout << "LIMITING execution to " << exec_count << " queries (via --limit)"
                      << std::endl;
        }

        for (int j = 0; j < exec_count; j++) {
            // --- CHANGED END ---
            db->schedule(query_plans[j]);
            context->barrier();
            auto start_time_q = std::chrono::high_resolution_clock::now();
            result_tables[j]  = (db->execute())[0];
            context->barrier();
            auto end_time_q = std::chrono::high_resolution_clock::now();
            query_plans[j]  = nullptr;
            timings_maximus[i][j] =
                std::chrono::duration_cast<std::chrono::microseconds>(end_time_q - start_time_q)
                    .count();
        }

        end_time = std::chrono::high_resolution_clock::now();
        std::cout
            << "---> Processed queries in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
            << "ms \n";
        PL("Repetition_" + std::to_string(i));
    }

    std::cout << "===================================" << std::endl;
    std::cout << "              TIMINGS              " << std::endl;
    std::cout << "===================================" << std::endl;
    std::cout << "Execution times [ms]:\n";

    std::vector<int64_t> timings_maximus_flat;
    for (int j = 0; j < timings_maximus.size(); ++j) {
        Stats stats(timings_maximus[j]);
        std::cout << "  Rep. " << j << ": " << "\tSUM: " << stats.total << "\tMIN: " << stats.min
                  << "\tMAX: " << stats.max << "\tAVG: " << stats.avg
                  << "\tQPS: " << stats.rate_per_second << std::endl;
        timings_maximus_flat.insert(
            timings_maximus_flat.end(), timings_maximus[j].begin(), timings_maximus[j].end());
    }
    Stats stats_total(timings_maximus_flat);
    std::cout << "  Total: " << "\tSUM: " << stats_total.total << "\tMIN: " << stats_total.min
              << "\tMAX: " << stats_total.max << "\tAVG: " << stats_total.avg
              << "\tQPS: " << stats_total.rate_per_second << std::endl;
    std::cout << std::endl;

    std::cout << "===================================" << std::endl;
    std::cout << "              RESULTS              " << std::endl;
    std::cout << "===================================" << std::endl;

    // Persist result of last iteration if requested
    if (persist_results != "no") {
        big_vector_bench::write_results_to_file(context, result_tables);
    } else {
        std::cout << "Results not persisted" << std::endl;
    }
    result_tables[0]->slice(0, 5)->print();
    //result_tables[1]->print();
    //result_tables[2]->print();


    context->barrier();

    std::cout << "===================================" << std::endl;
    std::cout << "              QUALITY              " << std::endl;
    std::cout << "===================================" << std::endl;

    maximus::QualityMetrics metrics = workload->evaluate(result_tables);
    metrics.print();
    std::cout << std::endl;

    if (save_benchmarking_results != "no") {
        std::string filename = save_benchmarking_results;
        write_benchmarking_results_to_file(
            filename, title, benchmark, query_parameters, stats_total, metrics);
    } else {
        std::cout << "Benchmarking results not saved to file." << std::endl;
    }

    // PL(query);
    context->barrier();
    PROFILER_FLUSH(mgr);
    return 0;
}
