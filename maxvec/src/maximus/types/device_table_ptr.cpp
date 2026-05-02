#include <iostream>
#include <maximus/error_handling.hpp>
#include <maximus/types/device_table_ptr.hpp>

#ifdef MAXIMUS_WITH_CUDA
#include <cudf/concatenate.hpp>
#endif

namespace maximus {

std::string device_type_to_string(DeviceType type) {
    switch (type) {
        case DeviceType::CPU:
            return "CPU";
        case DeviceType::GPU:
            return "GPU";
        default:
            return "UNDEFINED";
    }
}

bool DeviceTablePtr::is_none() const {
    return std::holds_alternative<Empty>(value);
}

bool DeviceTablePtr::empty() const {
    // ivoke the bool() operator
    assert(static_cast<bool>(*this) && "DeviceTablePtr is None or nullptr!");

    if (is_table()) {
        assert(as_table());
        return as_table()->num_rows() == 0;
    }

    if (is_table_batch()) {
        assert(as_table_batch());
        return as_table_batch()->num_rows() == 0;
    }

    if (is_arrow_table()) {
        assert(as_arrow_table());
        return as_arrow_table()->num_rows() == 0;
    }

    if (is_arrow_table_batch()) {
        assert(as_arrow_table_batch());
        return as_arrow_table_batch()->num_rows() == 0;
    }

    if (is_acero_table_batch()) {
        assert(as_acero_table_batch());
        return as_acero_table_batch()->length == 0;
    }

#ifdef MAXIMUS_WITH_CUDA
    if (is_gtable()) {
        assert(as_gtable());
        return as_gtable()->get_num_rows() == 0;
    }

    if (is_cudf_table()) {
        assert(as_cudf_table());
        return as_cudf_table()->num_rows() == 0;
    }

#endif
    throw std::runtime_error("DeviceTablePtr::empty(): the value is incorrect.");
}

bool DeviceTablePtr::is_table_batch() const {
    return std::holds_alternative<TableBatchPtr>(value);
}

bool DeviceTablePtr::is_table() const {
    return std::holds_alternative<TablePtr>(value);
}

bool DeviceTablePtr::is_arrow_table_batch() const {
    return std::holds_alternative<ArrowTableBatchPtr>(value);
}

bool DeviceTablePtr::is_arrow_table() const {
    return std::holds_alternative<ArrowTablePtr>(value);
}

bool DeviceTablePtr::is_acero_table_batch() const {
    return std::holds_alternative<AceroTableBatchPtr>(value);
}

bool DeviceTablePtr::is_gtable() const {
    return std::holds_alternative<GTablePtr>(value);
}

bool DeviceTablePtr::is_cudf_table() const {
    return std::holds_alternative<CudfTablePtr>(value);
}

[[nodiscard]] bool DeviceTablePtr::is_batch_type() const {
    return is_table_batch() || is_arrow_table_batch() || is_acero_table_batch() || is_gtable() ||
           is_cudf_table();
}

[[nodiscard]] bool DeviceTablePtr::is_table_type() const {
    return is_table() || is_arrow_table();
}

bool DeviceTablePtr::on_cpu() const {
    return is_table_batch() || is_table() || is_arrow_table_batch() || is_arrow_table() ||
           is_acero_table_batch();
}

bool DeviceTablePtr::on_gpu() const {
    return is_gtable() || is_cudf_table();
}

TableBatchPtr DeviceTablePtr::as_table_batch() const {
    assert(!is_none());
    assert(on_cpu());
    assert(is_table_batch());

    return std::get<TableBatchPtr>(value);
}

TablePtr DeviceTablePtr::as_table() const {
    assert(!is_none());
    assert(on_cpu());
    assert(is_table());

    return std::get<TablePtr>(value);
}

ArrowTableBatchPtr DeviceTablePtr::as_arrow_table_batch() const {
    assert(!is_none());
    assert(on_cpu());
    assert(is_arrow_table_batch());

    return std::get<ArrowTableBatchPtr>(value);
}

ArrowTablePtr DeviceTablePtr::as_arrow_table() const {
    assert(!is_none());
    assert(on_cpu());
    assert(is_arrow_table());

    return std::get<ArrowTablePtr>(value);
}

AceroTableBatchPtr DeviceTablePtr::as_acero_table_batch() const {
    assert(!is_none());
    assert(on_cpu());
    assert(is_acero_table_batch());

    return std::get<AceroTableBatchPtr>(value);
}

GTablePtr DeviceTablePtr::as_gtable() const {
    assert(!is_none());
    assert(on_gpu());
    assert(is_gtable());
    return std::get<GTablePtr>(value);
}
CudfTablePtr DeviceTablePtr::as_cudf_table() const {
    assert(!is_none());
    assert(on_gpu());
    assert(is_cudf_table());
    return std::get<CudfTablePtr>(value);
}

bool DeviceTablePtr::on_device(const DeviceType &device_type) const {
    assert(device_type != DeviceType::UNDEFINED);
    if (device_type == DeviceType::CPU) {
        return on_cpu();
    } else {
        assert(device_type == DeviceType::GPU);
        return on_gpu();
    }
}

// Conversion operator to bool
DeviceTablePtr::operator bool() const {
    // return true if this is not None and the std::shared_ptr are not null
    if (is_none()) {
        return false;
    }
    if (is_table()) {
        return as_table() != nullptr;
    }
    if (is_table_batch()) {
        return as_table_batch() != nullptr;
    }
    if (is_arrow_table()) {
        return as_arrow_table() != nullptr;
    }
    if (is_arrow_table_batch()) {
        return as_arrow_table_batch() != nullptr;
    }
    if (is_acero_table_batch()) {
        return as_acero_table_batch() != nullptr;
    }
    if (is_gtable()) {
        return as_gtable() != nullptr;
    }
    if (is_cudf_table()) {
        return as_cudf_table() != nullptr;
    }
    return false;
}

std::string DeviceTablePtr::to_string() const {
    if (is_none()) {
        return "NONE";
    }
    if (is_table_batch()) {
        return "TableBatch";
    }
    if (is_table()) {
        return "Table";
    }
    if (is_arrow_table_batch()) {
        return "ArrowTableBatch";
    }
    if (is_arrow_table()) {
        return "ArrowTable";
    }
    if (is_acero_table_batch()) {
        return "AceroTableBatch";
    }
    if (is_gtable()) {
        return "GTable";
    }
    if (is_cudf_table()) {
        return "CudfTable";
    }
    throw std::runtime_error("Unsupported type");
}


DeviceTablePtr merge_batches(std::shared_ptr<MaximusContext> &ctx,
                             std::vector<DeviceTablePtr> &batches,
                             std::shared_ptr<Schema> schema) {
    if (batches.size() == 0) return DeviceTablePtr();
    std::vector<ArrowTableBatchPtr> arrow_table_batches;
    arrow_table_batches.reserve(batches.size());

#ifdef MAXIMUS_WITH_CUDA
    std::vector<::cudf::table_view> gtable_batches;
    gtable_batches.reserve(batches.size());
#endif

    for (auto &b : batches) {
        if (b.on_device(DeviceType::CPU)) {
            b.convert_to<ArrowTableBatchPtr>(ctx, schema);
            assert(b.is_arrow_table_batch());
            arrow_table_batches.push_back(b.as_arrow_table_batch());
        }
#ifdef MAXIMUS_WITH_CUDA
        if (b.on_device(DeviceType::GPU)) {
            assert(b.on_device(DeviceType::GPU) && "Unknown device type.");
            b.convert_to<CudfTablePtr>(ctx, schema);
            assert(b.is_cudf_table());
            gtable_batches.push_back(b.as_cudf_table()->view());
        }
#endif
    }

    // either all batches have to be cpu batches or all have to be gpu batches
    assert(arrow_table_batches.size() == 0 || arrow_table_batches.size() == batches.size());

    if (arrow_table_batches.size() > 0) {
        auto maybe_arrow_table =
            ::arrow::Table::FromRecordBatches(schema->get_schema(), arrow_table_batches);
        if (!maybe_arrow_table.ok()) {
            CHECK_STATUS(maybe_arrow_table.status());
        }
        auto arrow_table = std::move(maybe_arrow_table.ValueOrDie());
        assert(schema->get_schema()->Equals(*arrow_table->schema()));
        return DeviceTablePtr(std::move(arrow_table));
    }

#ifdef MAXIMUS_WITH_CUDA
    assert(gtable_batches.size() == batches.size());
    CudfTablePtr table =
        std::move(::cudf::concatenate(::cudf::host_span<::cudf::table_view const>(gtable_batches)));
    return DeviceTablePtr(std::move(table));
#endif

    throw std::runtime_error("Unsupported table types for merge_batches.");
}

// template-instantiate the convert_to function for all variant-types
template void DeviceTablePtr::convert_to<TableBatchPtr>(const std::shared_ptr<MaximusContext> &ctx,
                                                        const std::shared_ptr<Schema> &schema,
                                                        PoolType pool_type);
template void DeviceTablePtr::convert_to<TablePtr>(const std::shared_ptr<MaximusContext> &ctx,
                                                   const std::shared_ptr<Schema> &schema,
                                                   PoolType pool_type);
template void DeviceTablePtr::convert_to<ArrowTableBatchPtr>(
    const std::shared_ptr<MaximusContext> &ctx,
    const std::shared_ptr<Schema> &schema,
    PoolType pool_type);
template void DeviceTablePtr::convert_to<ArrowTablePtr>(const std::shared_ptr<MaximusContext> &ctx,
                                                        const std::shared_ptr<Schema> &schema,
                                                        PoolType pool_type);
template void DeviceTablePtr::convert_to<AceroTableBatchPtr>(
    const std::shared_ptr<MaximusContext> &ctx,
    const std::shared_ptr<Schema> &schema,
    PoolType pool_type);
template void DeviceTablePtr::convert_to<GTablePtr>(const std::shared_ptr<MaximusContext> &ctx,
                                                    const std::shared_ptr<Schema> &schema,
                                                    PoolType pool_type);
template void DeviceTablePtr::convert_to<CudfTablePtr>(const std::shared_ptr<MaximusContext> &ctx,
                                                       const std::shared_ptr<Schema> &schema,
                                                       PoolType pool_type);

[[nodiscard]] TableBatchPtr DeviceTablePtr::as_cpu() const {
    return as_table_batch();
}

[[nodiscard]] GTablePtr DeviceTablePtr::as_gpu() const {
    return as_gtable();
}

void DeviceTablePtr::to_cpu(std::shared_ptr<MaximusContext> &ctx, std::shared_ptr<Schema> &schema) {
    convert_to<TableBatchPtr>(ctx, schema);
}
void DeviceTablePtr::to_gpu(std::shared_ptr<MaximusContext> &ctx, std::shared_ptr<Schema> &schema) {
    convert_to<GTablePtr>(ctx, schema);
}
}  // namespace maximus
