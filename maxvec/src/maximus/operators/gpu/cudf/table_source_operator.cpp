#include <maximus/gpu/cuda_api.hpp>
#include <maximus/gpu/cudf/csv.hpp>
#include <maximus/operators/gpu/cudf/table_source_operator.hpp>
#include <maximus/utils/utils.hpp>
#include <typeinfo>

namespace maximus::cudf {

TableSourceOperator::TableSourceOperator(std::shared_ptr<MaximusContext> &_ctx,
                                         std::shared_ptr<TableSourceProperties> _properties)
        : AbstractTableSourceOperator(_ctx, std::move(_properties)) {
    assert(ctx_);
    auto gctx = ctx_->get_gpu_context();

    assert(gctx != nullptr && "MaximusGContext must be initialized "
                              "before creating a GPU GroupByOperator");

    assert(typeid(*gctx) == typeid(maximus::gpu::MaximusCudaContext) &&
           "MaximusGContext must be a MaximusCudaContext");

    // assert(properties->schema && !properties->table && !properties->batch_reader &&
    //        !properties->path.empty() && "TableSourceProperties must have a schema and a path");

    assert(this->properties);
    auto device_table = this->properties->table;
    if (device_table) {
        assert(device_table.is_gtable());
        table = device_table.as_gtable();
        assert(table);

        if (!this->properties->include_columns.empty()) {
            table->select_columns(this->properties->include_columns);
        }
    } else {
        // read the table from the path
        device_table = read_table(ctx_,
                                  this->properties->path,
                                  this->properties->schema,
                                  this->properties->include_columns,
                                  DeviceType::GPU);
        assert(device_table.is_gtable());
        table = device_table.as_gtable();
        // note table->get_schema() and properties->schema here don't have to match!
        // if include_columns is specified, then properties->schema will contain the full schema
        // and the table->get_schema() will contain only the columns from the include_columns
    }
    assert(table);

    output_schema = table->get_schema();
    assert(output_schema && output_schema->size() > 0 && "Output schema not set in table source.");

    set_device_type(DeviceType::GPU);
    set_engine_type(EngineType::CUDF);
}

bool TableSourceOperator::has_more_batches_impl(bool blocking) {
    return table != nullptr;
}

DeviceTablePtr TableSourceOperator::export_next_batch_impl() {
    assert(table);
    return DeviceTablePtr(std::move(table));
}

}  // namespace maximus::cudf
