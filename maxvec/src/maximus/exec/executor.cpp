#include <algorithm>
#include <cassert>
#include <maximus/exec/executor.hpp>

namespace maximus {

Executor::Executor(std::shared_ptr<MaximusContext> &ctx): ctx_(ctx) {
    if (ctx_->n_outer_threads > 1) {
        tf_executor_ = std::make_unique<tf::Executor>(ctx_->n_outer_threads);
    }
    CHECK_STATUS(arrow::SetCpuThreadPoolCapacity(ctx_->n_inner_threads));
    // CHECK_STATUS(arrow::io::SetIOThreadPoolCapacity(2));
    // std::cout << "IO THREADS = " << arrow::io::GetIOThreadPoolCapacity() << std::endl;
    // std::cout << "COMPUTE THREADS = " << arrow::GetCpuThreadPoolCapacity() << std::endl;
}

// Executor class
void Executor::add_pipelines(const std::vector<std::shared_ptr<Pipeline>> pipelines) {
    cleanup();
    // assert(pipelines_.empty() && "add_pipelines can only be called once.");

    if (pipelines.size() == 0) return;

    int num_pipelines = pipelines_.size();

    std::unordered_map<int, int> task_id;
    std::vector<tf::Task> pipeline_tasks;
    int idx = 0;

    if (ctx_->n_outer_threads > 1) {
        pipeline_tasks.reserve(num_pipelines);
        taskflows_.push_back(tf::Taskflow());
    }

    // convert the pipelines to tasks
    for (auto &pipeline : pipelines) {
        assert(pipeline && "The pipeline is nullptr.");
        auto pipeline_id = pipeline->get_id();

        pipelines_[pipeline_id] = pipeline;

        assert(pipelines_[pipeline_id]->get_id() == pipeline_id &&
               "The pipeline id is not correct.");
        assert(pipelines_[pipeline_id] == pipeline);

        if (ctx_->n_outer_threads > 1) {
            auto &current_pipeline = pipelines_[pipeline_id];
            pipeline_tasks.emplace_back(taskflows_.back().emplace([&]() {
                current_pipeline->execute();
            }));

            task_id[pipeline_id] = idx;
        } else {
            assert(ctx_->n_outer_threads == 1);
            indegree[pipeline_id] = pipelines_[pipeline_id]->num_predecessors();
            if (indegree[pipeline_id] == 0) {
                queue.push(pipeline_id);
            }
        }
        ++idx;
    }

    if (ctx_->n_outer_threads > 1) {
        // add the tasks dependencies
        for (auto &pipeline : pipelines) {
            auto parent_task_id = task_id[pipeline->get_id()];
            for (const auto &child : pipeline->successor_pipelines()) {
                int child_task_id = task_id[child->get_id()];
                pipeline_tasks[parent_task_id].precede(pipeline_tasks[child_task_id]);
            }
        }
    }
}

void Executor::schedule(std::shared_ptr<QueryPlan> &query_plan) {
    PE("Executor::schedule");
    assert(query_plan && "Executor::schedule: query_plan is nullptr.");
    auto pipelines = query_plan->to_pipelines();
    add_pipelines(std::move(pipelines));
    ++pending_queries_;
    // if there are already pending queries, we turn on the outer parallelism
    PL("Executor::schedule");
}

void Executor::execute() {
    PE("Executor::execute");
    int num_pipelines = this->pipelines_.size();

    if (num_pipelines == 0) return;

    // Run the taskflow with multiple workers
    // auto start= std::chrono::high_resolution_clock::now();

    if (ctx_->n_outer_threads > 1) {
        // std::cout << "Running the executor with " << ctx_->n_outer_threads << " threads.\n";
        std::vector<tf::Future<void>> futures;
        futures.reserve(taskflows_.size());
        for (auto &taskflow : taskflows_) {
            futures.emplace_back(tf_executor_->run(taskflow));
        }

        /*
        for (auto &future : futures) {
            future.wait();
        }
        */
        tf_executor_->wait_for_all();
    } else {
        // std::cout << "Running the executor with a single thread.\n";
        assert(ctx_->n_outer_threads == 1);
        while (!queue.empty()) {
            auto pipeline_id = queue.front();
            queue.pop();
            auto &pipeline = pipelines_[pipeline_id];
            pipeline->execute();
            for (const auto &child : pipeline->successor_pipelines()) {
                int child_id = child->get_id();
                --indegree[child_id];
                assert(indegree[child_id] >= 0);
                if (indegree[child_id] == 0) {
                    queue.push(child_id);
                }
            }
        }
    }

    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout << "Execution time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    should_cleanup_ = true;
    PL("Executor::execute");
}

void Executor::cleanup() {
    if (should_cleanup_) {
        this->pipelines_.clear();
        this->taskflows_.clear();
        this->indegree.clear();
        pending_queries_ = 0;
        indegree.clear();
    }
    should_cleanup_ = false;
}

void Executor::set_num_inner_threads(int num_inner_threads) {
    assert(num_inner_threads >= 1);
    CHECK_STATUS(arrow::SetCpuThreadPoolCapacity(num_inner_threads));
}

void Executor::set_num_outer_threads(int num_outer_threads) {
    assert(num_outer_threads >= 1);
    if (num_outer_threads > 1) {
        tf_executor_ = std::move(std::make_unique<tf::Executor>(num_outer_threads));
    }
}

AceroExecutor::AceroExecutor(int num_threads): num_threads_(num_threads) {
    CHECK_STATUS(arrow::SetCpuThreadPoolCapacity(num_threads));
}

void AceroExecutor::schedule(std::shared_ptr<QueryPlan> &query_plan) {
    assert(query_plan && "AceroExecutor::schedule: query_plan is nullptr.");
    acero_plans_.emplace_back(
        std::make_shared<arrow::acero::Declaration>(to_acero_declaration(query_plan)));
}

void AceroExecutor::execute() {
    auto ctx         = make_context();
    bool use_threads = num_threads_ > 1;

    for (const auto &acero_plan : acero_plans_) {
        auto maybe_arrow_table = arrow::acero::DeclarationToTable(*acero_plan, use_threads);
        if (!maybe_arrow_table.ok()) {
            CHECK_STATUS(maybe_arrow_table.status());
        }
        auto arrow_table = maybe_arrow_table.ValueOrDie();
        tables_.emplace_back(std::make_shared<Table>(ctx, arrow_table));
    }
    acero_plans_.clear();
}

std::vector<TablePtr> AceroExecutor::results() {
    auto result = std::move(tables_);
    tables_.clear();
    return std::move(result);
}

}  // namespace maximus
