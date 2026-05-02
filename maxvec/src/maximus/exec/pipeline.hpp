#pragma once

#include <maximus/dag/edge.hpp>
#include <maximus/operators/abstract_operator.hpp>
#include <vector>

namespace maximus {

class Pipeline : public std::enable_shared_from_this<Pipeline> {
public:
    Pipeline() {
        static int id_counter_ = 0;
        id_                    = id_counter_++;
    }

    Pipeline(Pipeline&&)            = default;
    Pipeline& operator=(Pipeline&&) = default;

    Pipeline(Pipeline& other) = delete;
    Pipeline(const Pipeline&) = delete;

    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline& operator=(Pipeline&)       = delete;

    ~Pipeline() {
        // std::cout << "Pipeline " << get_id() << " is being destroyed." << std::endl;
    }

    void add_child_pipeline(std::shared_ptr<Pipeline> pipeline);

    void add_operator(std::shared_ptr<AbstractOperator> op);
    void add_operator(std::shared_ptr<AbstractOperator> op, Edge edge);

    std::shared_ptr<AbstractOperator>& get_source();
    std::shared_ptr<AbstractOperator>& get_sink();

    std::vector<std::shared_ptr<AbstractOperator>>& get_operators();
    std::vector<std::shared_ptr<AbstractOperator>> get_operators() const;

    std::vector<std::shared_ptr<Pipeline>> predecessor_pipelines();
    std::vector<std::shared_ptr<Pipeline>> successor_pipelines();

    std::size_t num_predecessors() const;
    std::size_t num_successors() const;

    void push(std::size_t from_op);

    void execute();

    int get_id() const;

    std::string to_string() const;

    std::size_t size() const;

private:
    int id_ = -1;
    std::vector<std::weak_ptr<Pipeline>> parents_;
    std::vector<std::shared_ptr<Pipeline>> children_;

    std::vector<std::shared_ptr<AbstractOperator>> operators_;
    std::vector<Edge> edges_;
};
}  // namespace maximus
