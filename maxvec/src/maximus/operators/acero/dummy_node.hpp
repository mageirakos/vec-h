#pragma once
#include <arrow/acero/exec_plan.h>
#include <arrow/api.h>
#include <arrow/compute/exec.h>

namespace maximus {
namespace acero {

class DummyNode : public arrow::acero::ExecNode {
public:
    // using NodeVector = arrow::acero::ExecPlan::NodeVector;

    ~DummyNode() override = default;

    explicit DummyNode(arrow::acero::ExecPlan* plan,
                       arrow::acero::ExecPlan::NodeVector inputs    = {},  // can be empty
                       std::shared_ptr<arrow::Schema> output_schema = nullptr,
                       bool is_sink                                 = false,
                       std::string prefix                           = "");

    [[nodiscard]] const char* kind_name() const override;

    arrow::Status InputReceived(arrow::acero::ExecNode* input,
                                arrow::compute::ExecBatch batch) override;

    arrow::Status InputFinished(arrow::acero::ExecNode* input, int total_batches) override;

    arrow::Status StartProducing() override;

    void PauseProducing(arrow::acero::ExecNode* output, int32_t counter) override;

    void ResumeProducing(arrow::acero::ExecNode* output, int32_t counter) override;

    arrow::Status StopProducingImpl() override;

    std::shared_ptr<arrow::Schema> dummy_schema();

    bool started = false;

    std::vector<arrow::compute::ExecBatch> output_batches;

    std::string prefix;
};

arrow::acero::ExecNode* MakeDummyNode(arrow::acero::ExecPlan* plan,
                                      arrow::acero::ExecPlan::NodeVector inputs,
                                      std::shared_ptr<arrow::Schema> output_schema,
                                      bool is_sink,
                                      std::string prefix);

DummyNode* MakeDummySource(arrow::acero::ExecPlan* plan,
                           std::shared_ptr<arrow::Schema> output_schema,
                           std::string prefix);

DummyNode* MakeDummySink(arrow::acero::ExecPlan* plan,
                         arrow::acero::ExecPlan::NodeVector inputs,
                         std::string prefix);
}  // namespace acero
}  // namespace maximus
