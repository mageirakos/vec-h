#include <future>
#include <maximus/profiler/profiler.hpp>
#include <maximus/utils/utils.hpp>
#include <string>
#include <arrow/table.h>

#ifdef MAXIMUS_WITH_CUDA
#include <maximus/gpu/cudf/csv.hpp>
#include <maximus/gpu/cudf/parquet.hpp>
#endif

namespace maximus {

bool ends_with(const std::string& full_string, const std::string& ending) {
    if (ending.size() > full_string.size()) return false;

    return full_string.compare(full_string.size() - ending.size(), ending.size(), ending) == 0;
}

bool starts_with(const std::string& full_string, const std::string& start) {
    return full_string.size() >= start.size() && full_string.compare(0, start.size(), start) == 0;
}

bool contains(const std::string& full_string, const std::string& substring) {
    return full_string.find(substring) != std::string::npos;
}

DeviceTablePtr read_table(std::shared_ptr<MaximusContext>& ctx,
                          std::string path,
                          const std::shared_ptr<Schema>& schema,
                          const std::vector<std::string>& include_columns,
                          const DeviceType& storage_device) {
    profiler::open_regions({"read_table"});
    TablePtr cpu_table;
    GTablePtr gpu_table;
    if (ends_with(path, ".csv")) {
        if (storage_device == DeviceType::CPU) {
            auto status = Table::from_csv(ctx, path, schema, include_columns, cpu_table);
            CHECK_STATUS(status);
        }
        if (storage_device == DeviceType::GPU) {
#ifdef MAXIMUS_WITH_CUDA
            auto status = gpu::read_csv_cudf(
                ctx, path, schema, include_columns, gpu_table);
            CHECK_STATUS(status);
#else
            throw std::runtime_error("Maximus must be built with the GPU support.");
#endif
        }
    } else if (ends_with(path, ".parquet")) {
        if (storage_device == DeviceType::CPU) {
            auto status = Table::from_parquet(ctx, path, schema, include_columns, cpu_table);
            CHECK_STATUS(status);
        }
        if (storage_device == DeviceType::GPU) {
#ifdef MAXIMUS_WITH_CUDA
            auto status = gpu::read_parquet_cudf(
                ctx, path, schema, include_columns, gpu_table);
            CHECK_STATUS(status);
#else
            profiler::close_regions({"read_table"});
            throw std::runtime_error("Maximus must be built with the GPU support.");
#endif
        }
    } else {
        profiler::close_regions({"read_table"});
        throw std::runtime_error("Unsupported file format: " + path);
    }

    if (storage_device == DeviceType::GPU) {
        assert(gpu_table);
        profiler::close_regions({"read_table"});
        return DeviceTablePtr(std::move(gpu_table));
    }

    if (storage_device == DeviceType::CPU) {
        assert(cpu_table);
        profiler::close_regions({"read_table"});
        return DeviceTablePtr(std::move(cpu_table));
    }

    profiler::close_regions({"read_table"});
    throw std::runtime_error("Unsupported storage device: " +
                             device_type_to_string(storage_device));
}

DeviceTablePtr read_table_partitioned(std::shared_ptr<MaximusContext>& ctx,
                                      const std::vector<std::string>& paths,
                                      const std::shared_ptr<Schema>& schema,
                                      const std::vector<std::string>& include_columns,
                                      const DeviceType& storage_device) {
    if (paths.size() == 1) {
        return read_table(ctx, paths[0], schema, include_columns, storage_device);
    }
    // NOTE SOS: It is read CPU -> moveto -> GPU . Not direct DISK->GPU if partitioned.

    // Launch one async task per partition; always read on CPU first.
    std::vector<std::future<TablePtr>> futures;
    futures.reserve(paths.size());

    for (const auto& path : paths) {
        futures.emplace_back(std::async(std::launch::async,
            [ctx, path, schema, include_columns]() mutable {
                TablePtr frag;
                auto status = Table::from_parquet(ctx, path, schema, include_columns, frag);
                CHECK_STATUS(status);
                return frag;
            }));
    }

    // Collect results and extract underlying arrow::Table pointers.
    std::vector<std::shared_ptr<arrow::Table>> arrow_tables;
    arrow_tables.reserve(paths.size());

    for (auto& fut : futures) {
        auto frag = fut.get();
        arrow_tables.push_back(frag->get_table());
    }

    // Concatenate all fragments into a single arrow::Table.
    auto pool = ctx->get_memory_pool();
    auto concat_result =
        arrow::ConcatenateTables(arrow_tables, arrow::ConcatenateTablesOptions::Defaults(), pool);
    CHECK_STATUS(concat_result.status());

    auto cpu_table = std::make_shared<Table>(ctx, concat_result.ValueUnsafe());
    DeviceTablePtr dtp(cpu_table);

    if (storage_device == DeviceType::GPU) {
#ifdef MAXIMUS_WITH_CUDA
        dtp.convert_to<GTablePtr>(ctx, schema);
#else
        throw std::runtime_error("Maximus must be built with the GPU support.");
#endif
    }

    return dtp;
}

}  // namespace maximus
