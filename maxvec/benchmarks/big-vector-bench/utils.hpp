#pragma once
#include <iostream>
#include <maximus/utils/evaluation_helpers.hpp>
#include <vector>

#include "queries.hpp"

std::string big_vector_bench_parquet_path();

struct Stats {
    Stats(std::vector<int64_t>& timings) {  // input in microseconds
        min   = *std::min_element(timings.begin(), timings.end()) / 1000.0;
        max   = *std::max_element(timings.begin(), timings.end()) / 1000.0;
        total = 0;
        for (const auto& t : timings) {
            total += t;
        }
        total /= 1000.0;  // convert to milliseconds
        avg = total / timings.size();
        rate_per_second =
            (total != 0) ? (1000 * timings.size() / total) : std::numeric_limits<double>::max();
    }

    double min;  // in milliseconds
    double max;
    double avg;
    double total;
    double rate_per_second;
};

void write_benchmarking_results_to_file(std::string filename,
                                        std::string title,
                                        const std::string& benchmark,
                                        const big_vector_bench::QueryParameters& query_parameters,
                                        const Stats& stats_totals,
                                        const maximus::QualityMetrics& metrics);

std::vector<std::string> remove_value(const std::vector<std::string>& vec,
                                      const std::string& value);