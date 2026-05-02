#include <iostream>
#include <maximus/operators/acero/dummy_node.hpp>

namespace maximus::acero {

DummyNode::DummyNode(arrow::acero::ExecPlan *plan,
                     arrow::acero::ExecPlan::NodeVector inputs,
                     std::shared_ptr<arrow::Schema> output_schema,
                     bool is_sink,
                     std::string prefix)
        : arrow::acero::ExecNode(
              plan,
              std::move(inputs),
              {} /* input labels */,
              is_sink ? nullptr : (output_schema ? std::move(output_schema) : dummy_schema()))
        , prefix(std::move(prefix)) {
    input_labels_.resize(inputs_.size());
    for (size_t i = 0; i < input_labels_.size(); ++i) {
        input_labels_[i] = std::to_string(i);
    }
}

const char *DummyNode::kind_name() const {
    return "maximus::acero::DummyNode";
}
arrow::Status DummyNode::InputReceived(arrow::acero::ExecNode *input,
                                       arrow::compute::ExecBatch batch) {
    std::cout << "DummyNode::InputReceived batch = " << batch.ToString() << std::endl;
    if (!is_sink()) {
        std::cout << (prefix + ": forwarding the input...") << std::endl;
        return output_->InputReceived(this, std::move(batch));
    }
    std::cout << (prefix + ": caching the input...") << std::endl;
    output_batches.push_back(std::move(batch));
    return arrow::Status::OK();
}

arrow::Status DummyNode::InputFinished(arrow::acero::ExecNode *input, int total_batches) {
    if (!is_sink()) {
        std::cout << (prefix + ": forwarding the input finished signal...") << std::endl;
        return output_->InputFinished(this, total_batches);
    }
    std::cout << (prefix + ": input finished received...") << std::endl;
    return arrow::Status::OK();
}

arrow::Status DummyNode::StartProducing() {
    started = true;
    return arrow::Status::OK();
}

void DummyNode::PauseProducing(arrow::acero::ExecNode *output, int32_t counter) {
}

void DummyNode::ResumeProducing(arrow::acero::ExecNode *output, int32_t counter) {
}

arrow::Status DummyNode::StopProducingImpl() {
    return arrow::Status::OK();
}

std::shared_ptr<arrow::Schema> DummyNode::dummy_schema() {
    return arrow::schema({arrow::field("dummy", arrow::null())});
}

arrow::acero::ExecNode *MakeDummyNode(arrow::acero::ExecPlan *plan,
                                      arrow::acero::ExecPlan::NodeVector inputs,
                                      std::shared_ptr<arrow::Schema> output_schema,
                                      bool is_sink,
                                      std::string prefix) {
    /*
    return plan->EmplaceNode<DummyNode>(plan,
                                        std::move(inputs),
                                        std::move(output_schema),
                                        is_sink);
                                        */
    auto node = std::make_unique<DummyNode>(
        plan, std::move(inputs), std::move(output_schema), is_sink, prefix);
    return plan->AddNode(std::move(node));
}

DummyNode *MakeDummySource(arrow::acero::ExecPlan *plan,
                           std::shared_ptr<arrow::Schema> output_schema,
                           std::string prefix) {
    return dynamic_cast<DummyNode *>(
        MakeDummyNode(plan, {} /*inputs*/, std::move(output_schema), false, prefix));
}

DummyNode *MakeDummySink(arrow::acero::ExecPlan *plan,
                         arrow::acero::ExecPlan::NodeVector inputs,
                         std::string prefix) {
    return dynamic_cast<DummyNode *>(MakeDummyNode(plan, std::move(inputs), nullptr, true, prefix));
}
}  // namespace maximus::acero