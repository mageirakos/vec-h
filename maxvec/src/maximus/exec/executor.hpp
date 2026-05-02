#pragma once

#include <maximus/dag/query_plan.hpp>
#include <maximus/exec/pipeline.hpp>
#include <maximus/operators/acero/interop.hpp>
#include <taskflow/taskflow.hpp>
#include <vector>

namespace maximus {

// Executor class
class Executor {
public:
    Executor(std::shared_ptr<MaximusContext>& ctx);

    void schedule(std::shared_ptr<QueryPlan>& query_plan);

    void execute();

    void set_num_outer_threads(int num_outer_threads);

    void set_num_inner_threads(int num_inner_threads);

private:
    void cleanup();

    void add_pipelines(const std::vector<std::shared_ptr<Pipeline>> pipelines);

    std::unordered_map<int, std::shared_ptr<Pipeline>> pipelines_;

    std::vector<tf::Taskflow> taskflows_;
    std::unique_ptr<tf::Executor> tf_executor_;

    std::unordered_map<int, int> indegree;

    bool should_cleanup_ = false;

    int pending_queries_ = 0;

    std::shared_ptr<MaximusContext> ctx_;

    std::queue<int> queue;
};

class AceroExecutor {
public:
    AceroExecutor(int num_threads = get_num_inner_threads());

    void schedule(std::shared_ptr<QueryPlan>& query_plan);

    void execute();

    std::vector<TablePtr> results();

private:
    std::vector<std::shared_ptr<arrow::acero::Declaration>> acero_plans_;
    std::vector<TablePtr> tables_;

    int num_threads_ = 1;
};

}  // namespace maximus
