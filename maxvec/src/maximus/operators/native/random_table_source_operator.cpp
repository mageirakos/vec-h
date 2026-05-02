#include <iostream>
#include <maximus/operators/native/random_table_source_operator.hpp>
#include <maximus/utils/arrow_helpers.hpp>

namespace maximus {

RandomTableSourceOperator::RandomTableSourceOperator(
    std::shared_ptr<MaximusContext>& ctx,
    std::shared_ptr<Schema> schema,
    std::shared_ptr<RandomTableSourceProperties> properties)
        : AbstractOperator(PhysicalOperatorType::RANDOM_TABLE_SOURCE, ctx, schema, schema)
        , properties(std::move(properties)) {
    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::NATIVE);
}

bool RandomTableSourceOperator::has_more_batches_impl(bool blocking) {
    assert(num_ports() == 1);
    return !finished_;
}

DeviceTablePtr RandomTableSourceOperator::export_next_batch_impl() {
    TableBatchPtr output;
    CHECK_STATUS(generate_random_data(output));
    assert(output);

    // Update the generated_rows counter
    generated_rows_ += output->num_rows();

    // Check if we have generated enough rows
    if (generated_rows_ >= properties->total_num_rows) {
        finished_ = true;
    }

    return DeviceTablePtr(std::move(output));
}

Status RandomTableSourceOperator::generate_random_data(TableBatchPtr& table) {
    assert(properties->total_num_rows >= generated_rows_);
    // in case the output_batch_size doesn't divide the total_num_rows, the last batch is going to be smaller
    std::size_t num_rows_to_generate =
        std::min(properties->total_num_rows - generated_rows_, properties->output_batch_size);

    assert(output_schema);
    assert(output_schema->get_schema());

    // Generate a data chunk
    auto fields = output_schema->get_schema()->fields();
    std::shared_ptr<arrow::RecordBatch> batch =
        generate_batch(fields, num_rows_to_generate, properties->seed, ctx_->get_memory_pool());

    table = std::make_shared<TableBatch>(ctx_, batch);

    return Status::OK();
}

void RandomTableSourceOperator::on_add_input(maximus::DeviceTablePtr input, int port) {
    assert(false && "RandomTableSourceOperator doesn't support input.");
}

void RandomTableSourceOperator::on_no_more_input(int port) {
    assert(false && "RandomTableSourceOperator doesn't support input.");
}

}  // namespace maximus
