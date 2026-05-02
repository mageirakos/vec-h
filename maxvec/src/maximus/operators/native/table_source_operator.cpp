#include <iostream>
#include <maximus/io/parquet.hpp>
#include <maximus/operators/native/table_source_operator.hpp>
#include <maximus/utils/utils.hpp>

namespace maximus::native {

TableSourceOperator::TableSourceOperator(std::shared_ptr<MaximusContext>& ctx,
                                         std::shared_ptr<TableSourceProperties> properties)
        : AbstractTableSourceOperator(ctx, std::move(properties)) {
    set_device_type(DeviceType::CPU);
    set_engine_type(EngineType::NATIVE);

    assert(ctx_ && "TableSourceOperator's context must not be null");
    auto pool = ctx_->get_memory_pool();
    assert(pool && "TableSourceOperator's memory pool must not be null");

    // currently we read the whole table at once
    // make_sink();
    assert(this->properties);
    auto& include_columns = this->properties->include_columns;
    auto device_table     = this->properties->table;
    if (device_table) {
        assert(this->properties->path == "");
        assert(device_table.is_table());
        full_table_ = device_table.as_table();
        assert(full_table_);
        assert(full_table_->get_context() == ctx &&
               "TableSourceOperator's context must match the table's context");

        // filter the columns if needed
        if (!include_columns.empty()) {
            full_table_ = full_table_->select_columns(include_columns);
            assert(full_table_);
        }
    } else {
        assert(this->properties->path != "" &&
               "TableSourceOperator: either the path or the table must be set");

        // read the table from the path
        profiler::close_regions({"Executor::schedule"});
        profiler::open_regions({"Executor::execute",
                                "OPERATORS",
                                device_type_to_string(device_type),
                                name(),
                                "constructor"});
        auto device_table = read_table(ctx_,
                                       this->properties->path,
                                       this->properties->schema,
                                       include_columns,
                                       DeviceType::CPU);
        profiler::close_regions({"Executor::execute",
                                 "OPERATORS",
                                 device_type_to_string(device_type),
                                 name(),
                                 "constructor"});
        profiler::open_regions({"Executor::schedule"});
        assert(device_table.is_table());
        full_table_ = device_table.as_table();
        assert(full_table_);
    }

    assert((include_columns.empty() ||
            full_table_->get_table()->num_columns() == include_columns.size()) &&
           "TableSourceOperator: the number of columns in the table does not match the number of "
           "columns in the include_columns list");

    assert(full_table_);
    table_reader_ = std::make_shared<arrow::TableBatchReader>(full_table_->get_table());
    assert(table_reader_);

    auto schema = full_table_->get_schema();

    if (this->properties->output_batch_size > 0) {
        table_reader_->set_chunksize(this->properties->output_batch_size);
    }

    assign_input_schemas({schema});
    assign_output_schema(schema);

    assert(input_schemas.size() == 1 && input_schemas[0]);
    assert(output_schema);
}

void TableSourceOperator::read_next() {
    assert(!finished_ && "TableSourceOperator::read_next() called after the end of the file");
    assert(!output_ && "TableSourceOperator::read_next() called when the "
                       "output_ is not empty");

    std::shared_ptr<arrow::RecordBatch> next;
    if (table_reader_) {
        // read the next batch from the table_reader_
        CHECK_STATUS(table_reader_->ReadNext(&next));
    }

    if (next) {
        output_ = std::move(next);
        // output_ = std::make_shared<TableBatch>(ctx_, std::move(next));
        // std::cout << "output_batch = " << output_->slice(0, 10)->to_string() << "\n";
    }
}

bool TableSourceOperator::has_more_batches_impl(bool blocking) {
    if (output_) return true;

    if (finished_) return false;

    assert(input_schemas[0]);
    assert(output_schema);
    assert(*input_schemas[0] == *output_schema);

    read_next();

    if (!output_) {
        finished_ = true;
    }

    return !finished_;
}

DeviceTablePtr TableSourceOperator::export_next_batch_impl() {
    assert(output_);
    return DeviceTablePtr(std::move(output_));
}
}  // namespace maximus::native
