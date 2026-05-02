#pragma once

#include <maximus/dag/query_node.hpp>
#include <maximus/dag/query_plan.hpp>
#include <maximus/database.hpp>
#include <maximus/types/device_table_ptr.hpp>
#include <maximus/utils/evaluation_helpers.hpp>

namespace big_vector_bench {

struct QueryParameters {
    std::string method;
    std::string faiss_index;
    int hnsw_efsearch;       // efSearch for HNSW
    int cagra_itopksize;     // itopkSize for CAGRA
    int cagra_searchwidth;   // searchWidth for CAGRA
    int ivf_nprobe;          // nProbe for IVF
    int postfilter_ksearch;  // kSearch for convserative post-filtering
};

class AbstractWorkload {
public:
    virtual std::vector<std::shared_ptr<maximus::QueryPlan>> query_plans(
        const QueryParameters& query_parameters)                                     = 0;
    virtual std::vector<std::string> table_names() const                             = 0;
    virtual std::vector<std::shared_ptr<maximus::Schema>> table_schemas() const      = 0;
    virtual maximus::QualityMetrics evaluate(std::vector<maximus::TablePtr> results) = 0;
};

std::shared_ptr<AbstractWorkload> get_workload(const std::string& benchmark,
                                               std::shared_ptr<maximus::Database>& db);

void write_results_to_file(maximus::Context& context, std::vector<maximus::TablePtr> results);

}  // namespace big_vector_bench
