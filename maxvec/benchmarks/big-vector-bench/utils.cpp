#include "utils.hpp"

std::string big_vector_bench_parquet_path() {
    std::string path = PROJECT_SOURCE_DIR;
    path += "/tests/big-vector-bench/parquet";
    return path;
}

void write_benchmarking_results_to_file(std::string filename,
                                        std::string title,
                                        const std::string& benchmark,
                                        const big_vector_bench::QueryParameters& query_parameters,
                                        const Stats& stats_totals,
                                        const maximus::QualityMetrics& metrics) {
    std::ofstream outfile(filename, std::ios::app);
    if (!outfile) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    std::string description =
        benchmark + " - " + query_parameters.method + " - index: " + query_parameters.faiss_index +
        " - hnsw_efsearch: " + std::to_string(query_parameters.hnsw_efsearch) +
        " - postfilter_ksearch: " + std::to_string(query_parameters.postfilter_ksearch) +
        " - cagra_itopksize: " + std::to_string(query_parameters.cagra_itopksize) +
        " - cagra_searchwidth: " + std::to_string(query_parameters.cagra_searchwidth) +
        " - ivf_nprobe: " + std::to_string(query_parameters.ivf_nprobe) +
        " - precision: " + std::to_string(metrics.precision) +
        " - num_retrieved: " + std::to_string(metrics.num_retrieved);
    outfile << "[\n";
    outfile << "  \"" << title << "\",\n";
    outfile << "  \"" << description << "\",\n";
    outfile << "  " << metrics.recall << ",\n";
    outfile << "  " << stats_totals.rate_per_second << "\n";
    outfile << "]," << std::endl;
    outfile.close();
}

std::vector<std::string> remove_value(const std::vector<std::string>& vec,
                                      const std::string& value) {
    std::vector<std::string> result;
    result.reserve(vec.size());  // Reserve enough space
    for (const auto& s : vec) {
        if (s != value) {
            result.push_back(s);
        }
    }
    return result;
}