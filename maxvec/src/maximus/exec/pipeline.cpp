#include <fstream>
#include <iostream>
#include <maximus/exec/pipeline.hpp>
#include <maximus/operators/native/local_broadcast_operator.hpp>
#include <maximus/operators/native/scatter_operator.hpp>
#ifdef MAXIMUS_WITH_CUDA
#include <maximus/operators/gpu/cudf/local_broadcast_operator.hpp>
#include <maximus/operators/gpu/cudf/scatter_operator.hpp>
#endif
#include <sstream>

namespace maximus {

void Pipeline::add_child_pipeline(std::shared_ptr<Pipeline> pipeline) {
    children_.push_back(std::move(pipeline));
    auto shared_this = shared_from_this();
    assert(shared_this);
    children_.back()->parents_.emplace_back(std::move(shared_this));
}

void Pipeline::add_operator(std::shared_ptr<AbstractOperator> op) {
    assert(operators_.empty() && "Only the source operator (the first one in a "
                                 "pipeline) can be added without an edge.");
    operators_.emplace_back(std::move(op));
}

void Pipeline::add_operator(std::shared_ptr<AbstractOperator> op, Edge edge) {
    // assert(!is_frozen() && "Pipeline is frozen, cannot add more operators.");
    operators_.emplace_back(std::move(op));
    edges_.emplace_back(std::move(edge));
}

std::shared_ptr<AbstractOperator> &Pipeline::get_source() {
    // std::cout << "Getting the source for pipeline " << get_id() << "..." <<
    // std::endl;
    assert(!operators_.empty());
    return operators_[0];
}

std::shared_ptr<AbstractOperator> &Pipeline::get_sink() {
    assert(!operators_.empty());
    return operators_.back();
}

std::vector<std::shared_ptr<AbstractOperator>> &Pipeline::get_operators() {
    return operators_;
}

std::vector<std::shared_ptr<AbstractOperator>> Pipeline::get_operators() const {
    return operators_;
}

void Pipeline::push(std::size_t from_op) {
    if (from_op >= operators_.size()) {
        return;
    }

    assert(from_op < operators_.size());

    auto &op = operators_[from_op];

    // if not sink
    if (from_op < operators_.size() - 1) {
        auto &next_op  = operators_[from_op + 1];
        auto &edge     = edges_[from_op];
        auto next_port = edge.target_port;

        assert(next_op);

        if (from_op == 0) {
            assert(!op->needs_input(0));
        }
        bool blocking = from_op == 0;

        // std::cout << "Checking whether " <<
        // physical_operator_to_string(op->type) << " has more batches with
        // blocking = " << (blocking ? "TRUE" : "FALSE") << std::endl;

        if (op->type == PhysicalOperatorType::LOCAL_BROADCAST) {
            std::shared_ptr<AbstractLocalBroadcastOperator> bcast_op;
            if (is_gpu_engine(op->engine_type)) {
#ifdef MAXIMUS_WITH_CUDA
                bcast_op = std::dynamic_pointer_cast<cudf::LocalBroadcastOperator>(op);
#else
                throw std::runtime_error("Maximus must be compiled with the GPU support.");
#endif
            } else {
                assert(is_cpu_engine(op->engine_type));
                bcast_op = std::dynamic_pointer_cast<native::LocalBroadcastOperator>(op);
            }
            assert(bcast_op);
            auto bcast_port = edge.source_port;

            while (bcast_op->has_more_batches(true, bcast_port)) {
                auto batch = bcast_op->export_next_batch(bcast_port);
                assert(batch);
                if (next_op->needs_input(next_port)) {
                    next_op->add_input(std::move(batch), next_port);
                    // push(from_op + 1);
                }
            }
        } else if (op->type == PhysicalOperatorType::SCATTER) {
            // ScatterOperator is a multi-output operator like LocalBroadcast
            std::shared_ptr<AbstractScatterOperator> scatter_op;
            if (is_gpu_engine(op->engine_type)) {
#ifdef MAXIMUS_WITH_CUDA
                scatter_op = std::dynamic_pointer_cast<cudf::ScatterOperator>(op);
#else
                throw std::runtime_error("Maximus must be compiled with the GPU support.");
#endif
            } else {
                assert(is_cpu_engine(op->engine_type));
                scatter_op = std::dynamic_pointer_cast<native::ScatterOperator>(op);
            }
            assert(scatter_op);
            auto scatter_port = edge.source_port;

            while (scatter_op->has_more_batches(true, scatter_port)) {
                auto batch = scatter_op->export_next_batch(scatter_port);
                assert(batch);
                if (next_op->needs_input(next_port)) {
                    next_op->add_input(std::move(batch), next_port);
                }
            }
        } else {
            assert(op->type != PhysicalOperatorType::LOCAL_BROADCAST);
            assert(op->type != PhysicalOperatorType::SCATTER);
            while (op->has_more_batches(blocking)) {
                // std::cout << "Yes, " << physical_operator_to_string(op->type)
                //           << " has more batches..." << std::endl;
                // we always export the next batch, regardless of the next
                // operator because we don't want to leave any batches behind
                auto batch = op->export_next_batch();
                assert(batch);

                if (next_op->needs_input(next_port)) {
                    // std::cout << "Pipeline: " << get_id() << ", adding input
                    // " << physical_operator_to_string(op->type) << ", to " <<
                    // physical_operator_to_string(next_op->type) << "..." <<
                    // std::endl;
                    next_op->add_input(std::move(batch), next_port);
                    // push(from_op + 1);
                } else {
                    // std::cout <<"Pipeline: " << get_id() << ", next_op = " <<
                    // physical_operator_to_string(next_op->type) << ", doesn't
                    // need input..." << std::endl; auto finished =
                    // next_op->is_finished();
                }
            }
        }

        // std::cout << "Pipeline: " << get_id() << ", invoking no_more_input from " << physical_operator_to_string(op->type) << ", to " <<
        // physical_operator_to_string(next_op->type) << "..." << std::endl;
        next_op->no_more_input(next_port);
        push(from_op + 1);
        // assert(!next_op->has_more_batches(true));
    }
}

void Pipeline::execute() {
    // PE("Pipeline::execute");
    // std::cout << "Starting the execution of the pipeline \n" << to_string()
    // << std::endl;
    assert(get_source() && "The source operator not set in the pipeline.");
    auto source = get_source();
    assert(source && "The source operator not set in the pipeline.");

    for (std::size_t i = 0; i < operators_.size() - 1; ++i) {
        auto &op      = operators_[i];
        auto &next_op = operators_[i + 1];
        assert(op && "Operator is null.");
        assert(next_op && "Next operator is null.");
        assert(op->engine_type != EngineType::UNDEFINED);
        assert(next_op->engine_type != EngineType::UNDEFINED);
        op->next_engine_type = next_op->engine_type;
        op->next_op_type     = next_op->type;
    }
    if (operators_.size() > 0) {
        auto &op = operators_.back();
        assert(op && "Operator is null.");
        op->next_engine_type = EngineType::NATIVE;
        op->next_op_type     = PhysicalOperatorType::TABLE_SINK;
    }

    push(0);

    // PL("Pipeline::execute");
    // std::cout << "Finished Propagating no_more_input for id = " << get_id()
    // << std::endl;
}

std::size_t Pipeline::num_predecessors() const {
    return parents_.size();
}

std::size_t Pipeline::num_successors() const {
    return children_.size();
}

std::vector<std::shared_ptr<Pipeline>> Pipeline::predecessor_pipelines() {
    std::vector<std::shared_ptr<Pipeline>> shared_parents(parents_.size());
    for (unsigned i = 0u; i < parents_.size(); ++i) {
        assert(!parents_[i].expired());

        shared_parents[i] = parents_[i].lock();

        assert(shared_parents[i]);
    }
    assert(shared_parents.size() == parents_.size());
    return shared_parents;
}

std::vector<std::shared_ptr<Pipeline>> Pipeline::successor_pipelines() {
    return children_;
}

int Pipeline::get_id() const {
    assert(id_ >= 0 && "Pipeline id is not set.");
    // assert(id_ < 10000 && "Pipeline id is too large.");
    return id_;
}

std::string Pipeline::to_string() const {
    std::stringstream ss;
    ss << "Pipeline " << get_id() << ":\n";
    for (auto &op : operators_) {
        ss << "    Operator: " << (op ? op->to_string(14) : "null") << "\n";
    }
    return ss.str();
}

std::size_t Pipeline::size() const {
    return operators_.size();
}

}  // namespace maximus
