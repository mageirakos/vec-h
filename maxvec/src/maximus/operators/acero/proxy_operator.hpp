#pragma once

#include <arrow/acero/options.h>
#include <arrow/util/async_generator.h>

#include <maximus/context.hpp>
#include <maximus/operators/abstract_operator.hpp>
#include <maximus/operators/acero/dummy_node.hpp>
#include <maximus/types/table_batch.hpp>
#include <mutex>

namespace cp = ::arrow::compute;
namespace ac = ::arrow::acero;

namespace maximus::acero {

struct CustomSinkNodeConsumer : public ac::SinkNodeConsumer {
    CustomSinkNodeConsumer(std::vector<DeviceTablePtr>& output_batches,
                           uint32_t* batches_received,
                           arrow::Future<> finish,
                           EngineType& next_engine_type,
                           std::shared_ptr<MaximusContext>& ctx)
            : output_batches(output_batches)
            , batches_received(batches_received)
            , finish(std::move(finish))
            , next_engine_type(next_engine_type)
            , ctx(ctx) {}

    arrow::Status Init(const std::shared_ptr<arrow::Schema>& schema,
                       ac::BackpressureControl* backpressure_control,
                       ac::ExecPlan* plan) override {
        // This will be called as the plan is started (before the first call to Consume)
        // and provides the schema of the data coming into the node, controls for pausing /
        // resuming input, and a pointer to the plan itself which can be used to access
        // other utilities such as the thread indexer or async task scheduler.
        // auto arrow_schema = arrow::schema(schema->fields(), schema->metadata());
        output_schema = std::make_shared<Schema>(schema);
        return arrow::Status::OK();
    }

    arrow::Status Consume(cp::ExecBatch batch) override {
        if (batch.length > 0) {
            AceroTableBatchPtr batch_ptr = std::make_shared<cp::ExecBatch>(std::move(batch));
            assert(batch_ptr);
            if (should_copy_to_gpu()) {
                local_lock.lock();
                local_batches.push_back(DeviceTablePtr(std::move(batch_ptr)));
                local_lock.unlock();
            } else {
                DeviceTablePtr device_batch = DeviceTablePtr(std::move(batch_ptr));
                assert(device_batch && !device_batch.empty());
                output_lock.lock();
                output_batches.push_back(std::move(device_batch));
                (*batches_received)++;
                output_lock.unlock();
            }
        }
        return arrow::Status::OK();
    }

    arrow::Future<> Finish() override {
        if (should_copy_to_gpu()) {
            // profiler::open_regions({"DataTransformation", "cpu::concatenate"});
            std::vector<ArrowTableBatchPtr> rbs;
            for (const auto batch : local_batches) {
                auto lb = batch.as_acero_table_batch();
                auto maybe_rb =
                    lb->ToRecordBatch(output_schema->get_schema(), ctx->get_memory_pool());
                auto rb = std::move(maybe_rb.ValueOrDie());
                assert(rb);
                rbs.push_back(std::move(rb));
            }


            auto rb = to_record_batch(rbs, ctx->get_pinned_memory_pool());
            // profiler::close_regions({"DataTransformation", "cpu::concatenate"});

            DeviceTablePtr device_batch = DeviceTablePtr(std::move(rb));
            assert(device_batch && !device_batch.empty());
            // device_batch.convert_to<CudfTablePtr>(ctx, output_schema, PoolType::DEFAULT, cudf::get_default_stream());
            output_lock.lock();
            output_batches.push_back(std::move(device_batch));
            (*batches_received)++;
            output_lock.unlock();
            local_batches.clear();
        }
        // Here you can perform whatever (possibly async) cleanup is needed, e.g. closing
        // output file handles and flushing remaining work
        return arrow::Future<>::MakeFinished();
    }

    bool should_copy_to_gpu() {
        assert(next_engine_type != EngineType::UNDEFINED);
        return is_gpu_engine(next_engine_type);
    }

    uint32_t* batches_received;
    arrow::Future<> finish;
    std::vector<DeviceTablePtr>& output_batches;
    std::shared_ptr<Schema> output_schema;
    std::mutex output_lock;
    EngineType& next_engine_type;
    std::shared_ptr<MaximusContext>& ctx;

    std::mutex local_lock;
    std::vector<DeviceTablePtr> local_batches;
};

class ProxyOperator {
public:
    ProxyOperator(std::shared_ptr<MaximusContext>& ctx,
                  std::vector<std::shared_ptr<Schema>> input_schemas,
                  std::vector<std::shared_ptr<arrow::acero::ExecNodeOptions>> node_sequence_options,
                  std::vector<std::string> node_sequence_names,
                  int operator_id,
                  EngineType& next_engine_type,
                  PhysicalOperatorType& next_op_type,
                  const std::vector<int>& blocking_ports = {})
            : ctx_(ctx)
            , input_schemas(std::move(input_schemas))
            , node_sequence_options_(std::move(node_sequence_options))
            , node_sequence_names_(std::move(node_sequence_names))
            , id_(operator_id)
            , next_engine_type(next_engine_type)
            , next_op_type(next_op_type) {
        assert(ctx_);

        auto num_ports = this->input_schemas.size();
        port_types     = std::vector<PortType>(num_ports, PortType::STREAMING);

        for (const auto& port : blocking_ports) {
            set_blocking_port(port);
        }

        no_more_input_ = std::vector<bool>(num_ports, false);

        init();
    }

    ProxyOperator(std::shared_ptr<MaximusContext>& ctx,
                  std::vector<std::shared_ptr<Schema>> input_schemas,
                  std::shared_ptr<arrow::acero::ExecNodeOptions> node_options,
                  const std::string& node_name,
                  int operator_id,
                  EngineType& next_engine_type,
                  PhysicalOperatorType& next_op_type,
                  const std::vector<int>& blocking_ports = {})
            : ProxyOperator(
                  ctx,
                  input_schemas,
                  std::vector<std::shared_ptr<arrow::acero::ExecNodeOptions>>{node_options},
                  std::vector<std::string>{node_name},
                  operator_id,
                  next_engine_type,
                  next_op_type,
                  blocking_ports) {
        assert(node_name != "fetch" &&
               "The Acero's limit node (FetchNode) can only be used within a "
               "FusedOperator. When using it alone, use the native limit node.");
    }

    ~ProxyOperator() {
        // ensure the plan has finished
        assert(!exec_plan_ || !plan_started_ || finish_.is_finished());
        // ensure there are no pending batches that have not been exported
        assert(output_batches_ready_ == output_batches_exported_);
    }

    void set_blocking_port(int port) {
        assert(port < port_types.size());
        port_types[port] = PortType::BLOCKING;
    }

    void reset() {
        inputs_.clear();
        input_generators_.clear();
        input_producers_.clear();

        init();
    }

    void init() {
        assert(ctx_);
        assert(inputs_.empty());
        assert(input_generators_.empty());
        assert(input_producers_.empty());
        assert(!node_sequence_options_.empty() && node_sequence_options_[0]);
        // assert(!mini_plan_);

        std::vector<arrow::acero::Declaration::Input> inputs;
        inputs.reserve(input_schemas.size());

        for (auto& schema : input_schemas) {
            arrow::PushGenerator<std::optional<arrow::ExecBatch>> gen;
            auto producer = gen.producer();
            input_generators_.emplace_back(std::move(gen));
            input_producers_.emplace_back(std::move(producer));
            std::string label = "source" + std::to_string(inputs_.size());

            inputs_.emplace_back(
                "source", arrow::acero::SourceNodeOptions{schema->get_schema(), gen}, label);

            // Convert std::vector<Declaration> to std::vector<Input>
            inputs.emplace_back(inputs_.back());
        }

        node_sequence_.reserve(node_sequence_options_.size());

        for (int i = 0; i < node_sequence_options_.size(); i++) {
            auto& node_options = node_sequence_options_[i];
            assert(node_options);
            std::string node_name = node_sequence_names_[i];
            assert(!node_name.empty());

            if (i == 0) {
                node_sequence_.emplace_back(node_sequence_names_[i],
                                            inputs,
                                            node_sequence_options_[i],
                                            node_sequence_names_[i]);
            } else {
                node_sequence_.emplace_back(node_sequence_names_[i],
                                            std::vector<arrow::acero::Declaration::Input>{},
                                            node_sequence_options_[i],
                                            node_sequence_names_[i]);
            }
        }

        arrow::Future<> finish                           = arrow::Future<>::Make();
        std::shared_ptr<CustomSinkNodeConsumer> consumer = std::make_shared<CustomSinkNodeConsumer>(
            output_batches_, &output_batches_ready_, finish, next_engine_type, ctx_);

        ac::Declaration consuming_sink{
            "consuming_sink", {}, ac::ConsumingSinkNodeOptions(consumer)};

        node_sequence_.push_back(consuming_sink);

        auto full_declaration = arrow::acero::Declaration::Sequence(node_sequence_);

        exec_plan_ = ctx_->get_mini_exec_plan();
        assert(exec_plan_);
        std::ignore = full_declaration.AddToPlan(exec_plan_.get());

        auto maybe_output_schema = arrow::acero::DeclarationToSchema(full_declaration);
        if (!maybe_output_schema.ok()) {
            CHECK_STATUS(maybe_output_schema.status());
        }
        auto arrow_schema = maybe_output_schema.ValueOrDie();
        output_schema     = std::make_shared<Schema>(arrow_schema);

        assert(output_schema);

        CHECK_STATUS(exec_plan_->Validate());
    }

    std::vector<ArrowTableBatchPtr> to_batches(ArrowTablePtr table) {
        arrow::TableBatchReader reader(table);
        reader.set_chunksize(1 << 15);
        auto maybe_batches = reader.ToRecordBatches();
        if (!maybe_batches.ok()) {
            CHECK_STATUS(maybe_batches.status());
        }
        return maybe_batches.ValueOrDie();
    }

    std::vector<AceroTableBatchPtr> to_acero_batches(std::shared_ptr<MaximusContext>& ctx,
                                                     std::shared_ptr<Schema> schema,
                                                     std::vector<ArrowTableBatchPtr> batches) {
        std::vector<AceroTableBatchPtr> acero_batches;
        for (auto& batch : batches) {
            DeviceTablePtr device_batch(std::move(batch));
            device_batch.convert_to<AceroTableBatchPtr>(ctx, schema);
            assert(device_batch.is_acero_table_batch());
            acero_batches.push_back(std::move(device_batch.as_acero_table_batch()));
        }
        return acero_batches;
    }

    void on_add_input(DeviceTablePtr device_input, int port) {
        if (!plan_started_) {
            plan_started_ = true;
            exec_plan_->StartProducing();
            finish_ = exec_plan_->finished();
        }
        assert(port < input_generators_.size());

        auto& producer = input_producers_[port];

        // here we want the profiling regions within convert_to not to be nested within these regions
        assert(operator_name != "");

        std::vector<AceroTableBatchPtr> batches;
        profiler::close_regions({operator_name, "add_input"});
        if (!device_input.is_acero_table_batch() && filter_pipeline(port)) {
            device_input.convert_to<ArrowTablePtr>(ctx_, input_schemas[port]);
            auto arrow_batches = to_batches(device_input.as_arrow_table());
            batches = to_acero_batches(ctx_, input_schemas[port], std::move(arrow_batches));
        } else {
            device_input.convert_to<AceroTableBatchPtr>(ctx_, input_schemas[port]);
            batches.push_back(std::move(device_input.as_acero_table_batch()));
        }
        profiler::open_regions({operator_name, "add_input"});

        for (auto batch : batches) {
            // std::cout << "Pushing batch to input generator for port " << port << std::endl;
            producer.Push(std::move(*batch));
        }

        if (is_streaming_port(port)) {
            assert(received_all_blocking_inputs());
        }
    }

    void on_no_more_input(int port) {
        assert(port < input_generators_.size());
        auto& producer = input_producers_[port];
        producer.Close();

        assert(port < no_more_input_.size());
        assert(port >= 0);
        no_more_input_[port] = true;

        if (received_all_inputs()) {
            // std::cout << "All inputs received for " << node_name_ << ", waiting for the plan to finish...\n";
            if (plan_started_) {
                assert(finish_.is_valid());
                finish_.Wait();
                assert(finish_.is_finished());
            }
        }
        // std::cout << "on_no_more_input: " << node_name_ << " port: " << port << ", output_batches_ready = " << output_batches_ready_.load() << ", output_batches_exported = " << output_batches_exported_ << std::endl;
    }

    bool has_more_batches_impl(bool blocking) {
        if (blocking) {
            assert(!plan_started_ || finish_.is_finished());
            assert(received_all_inputs());
        }
        // assert(received_all_inputs());
        return output_batches_ready_ > output_batches_exported_;
    }

    DeviceTablePtr export_next_batch_impl() {
        assert(output_batches_ready_ > output_batches_exported_);
        assert(output_batches_exported_ < output_batches_.size());
        assert(output_schema);
        auto batch = std::move(output_batches_[output_batches_exported_]);
        ++output_batches_exported_;
        return std::move(batch);
    }

    bool is_streaming_port(int port) const {
        assert(port < port_types.size());
        assert(port_types[port] == PortType::STREAMING || port_types[port] == PortType::BLOCKING);
        return port_types[port] == PortType::STREAMING;
    }

    bool is_blocking_port(int port) const {
        assert(port < port_types.size());
        assert(port_types[port] == PortType::STREAMING || port_types[port] == PortType::BLOCKING);
        return port_types[port] == PortType::BLOCKING;
    }

    bool received_all_blocking_inputs() const {
        bool received = true;
        for (int i = 0; i < port_types.size(); i++) {
            if (is_blocking_port(i) && !no_more_input_[i]) {
                received = false;
                break;
            }
        }
        return received;
    }

    bool received_all_inputs() {
        bool received = true;
        for (int i = 0; i < port_types.size(); i++) {
            if (!no_more_input_[i]) {
                received = false;
                break;
            }
        }
        return received;
    }

public:
    std::vector<std::shared_ptr<Schema>> input_schemas;
    std::shared_ptr<Schema> output_schema;

    std::string operator_name = "";

    EngineType& next_engine_type;
    PhysicalOperatorType& next_op_type;

protected:
    std::shared_ptr<MaximusContext> ctx_;

    // generators for sources (inputs) and the sink (output)
    std::vector<arrow::PushGenerator<std::optional<cp::ExecBatch>>> input_generators_;
    std::vector<arrow::PushGenerator<std::optional<cp::ExecBatch>>::Producer> input_producers_;

    // the nodes representing the sources (inputs) and the sink (output)
    std::vector<arrow::acero::Declaration> inputs_;
    std::vector<arrow::acero::Declaration> node_sequence_;
    std::vector<std::shared_ptr<arrow::acero::ExecNodeOptions>> node_sequence_options_;
    std::vector<std::string> node_sequence_names_;

    std::vector<PortType> port_types;
    std::vector<bool> no_more_input_;

    std::vector<DeviceTablePtr> output_batches_;
    uint32_t output_batches_ready_{0};
    uint32_t output_batches_exported_ = 0;

    std::shared_ptr<ac::ExecPlan> exec_plan_;

    arrow::Future<> finish_;

    int id_ = -1;

    bool plan_started_ = false;

private:
    bool starts_with_filter() const {
        return node_sequence_names_.size() >= 1 && node_sequence_names_[0] == "filter";
    }

    bool starts_with_project() const {
        return node_sequence_names_.size() >= 1 && node_sequence_names_[0] == "project";
    }

    bool starts_with_hash_join() const {
        return node_sequence_names_.size() >= 1 && node_sequence_names_[0] == "hashjoin";
    }

    bool starts_with_group_by() const {
        return node_sequence_names_.size() >= 1 && node_sequence_names_[0] == "aggregate";
    }

    bool is_fusion() const { return node_sequence_names_.size() > 1; }

    bool contains_filter() const {
        for (auto& node : node_sequence_names_) {
            if (node == "filter") return true;
        }
        return false;
    }

    bool filter_pipeline(int port) const {
        if (is_fusion()) return contains_filter();
        assert(next_op_type != PhysicalOperatorType::UNDEFINED);
        assert(next_engine_type != EngineType::UNDEFINED);
        if (starts_with_filter()) {
            if (is_gpu_engine(next_engine_type)) return true;
            std::vector<PhysicalOperatorType> blocking = {PhysicalOperatorType::ORDER_BY,
                                                          PhysicalOperatorType::GROUP_BY,
                                                          PhysicalOperatorType::HASH_JOIN,
                                                          PhysicalOperatorType::DISTINCT};

            // if the next operator is not blocking, then batch the current input.
            // This would in turn also batch the output.
            // The blocking operators prefer large batches, which is why
            // we don't batch it further.
            auto it = std::find(blocking.begin(), blocking.end(), next_op_type);
            return it == blocking.end();
        }
        return false;
    }
};
}  // namespace maximus::acero
