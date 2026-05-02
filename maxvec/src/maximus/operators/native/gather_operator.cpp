#include <maximus/operators/native/gather_operator.hpp>
#include <arrow/table.h>

namespace maximus::native {

GatherOperator::GatherOperator(std::shared_ptr<MaximusContext>& ctx,
                               std::shared_ptr<Schema> input_schema,
                               std::shared_ptr<GatherProperties> properties)
        : AbstractGatherOperator(ctx, input_schema, std::move(properties)) {
    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::NATIVE);
    // Initialize port tracking
    port_completed_.resize(this->properties->num_inputs, false);
}

void GatherOperator::on_add_input(DeviceTablePtr device_input, int port) {
    assert(device_input);

    const auto& operator_name = name();
    profiler::close_regions({operator_name, "add_input"});
    device_input.convert_to<TableBatchPtr>(ctx_, input_schemas[0]);
    profiler::open_regions({operator_name, "add_input"});

    auto batch = device_input.as_table_batch();
    input_batches_.push_back(batch);
}

void GatherOperator::on_no_more_input(int port) {
    // Track which ports have completed
    if (!port_completed_[port]) {
        port_completed_[port] = true;
        ++completed_ports_;
    }

    // Only process when ALL input ports have completed
    if (completed_ports_ < properties->num_inputs) {
        return;
    }

    if (input_batches_.empty()) return;

    // Convert all RecordBatches to Tables and concatenate
    std::vector<std::shared_ptr<arrow::Table>> tables;
    for (const auto& batch : input_batches_) {
        auto record_batch = batch->get_table_batch();
        auto table = arrow::Table::FromRecordBatches({record_batch});
        if (!table.ok()) {
            throw std::runtime_error("Failed to convert RecordBatch to Table: " + table.status().ToString());
        }
        tables.push_back(*table);
    }

    auto concat_result = arrow::ConcatenateTables(tables);
    if (!concat_result.ok()) {
        throw std::runtime_error("Failed to concatenate tables: " + concat_result.status().ToString());
    }
    auto full_table = *concat_result;

    // Convert back to RecordBatch
    auto reader = arrow::TableBatchReader(*full_table);
    std::shared_ptr<arrow::RecordBatch> combined_batch;
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    while (true) {
        auto result = reader.Next();
        if (!result.ok()) {
            throw std::runtime_error("Failed to read batch: " + result.status().ToString());
        }
        if (*result == nullptr) break;
        batches.push_back(*result);
    }

    // Convert all batches to a single table and then to a single batch
    auto final_table_result = arrow::Table::FromRecordBatches(batches);
    if (!final_table_result.ok()) {
        throw std::runtime_error("Failed to create final table: " + final_table_result.status().ToString());
    }
    auto final_table = *final_table_result;
    auto combine_result = final_table->CombineChunks();
    if (!combine_result.ok()) {
        throw std::runtime_error("Failed to combine chunks: " + combine_result.status().ToString());
    }
    final_table = *combine_result;

    auto batch_reader = arrow::TableBatchReader(*final_table);
    auto batch_result = batch_reader.Next();
    if (!batch_result.ok() || *batch_result == nullptr) {
        throw std::runtime_error("Failed to get final batch");
    }

    auto output_batch = std::make_shared<TableBatch>(ctx_, *batch_result);
    outputs_.push_back(DeviceTablePtr(std::move(output_batch)));
}

}  // namespace maximus::native
