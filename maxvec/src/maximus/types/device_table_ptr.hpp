#pragma once

#include <chrono>
#include <maximus/profiler/profiler.hpp>
#include <maximus/types/table.hpp>
#include <maximus/types/table_batch.hpp>
#include <maximus/utils/arrow_helpers.hpp>
#include <variant>

#ifdef MAXIMUS_WITH_CUDA
#include <cudf/interop.hpp>
#include <maximus/gpu/cudf/cudf_types.hpp>
#include <maximus/gpu/cudf/interop.hpp>
#include <maximus/gpu/cudf/table_reader.hpp>
#include <maximus/gpu/cudf/table_writer.hpp>
#include <maximus/gpu/gtable/gtable.hpp>
#else


#endif

namespace cudf {
class table_view;
class table;
}  // namespace cudf

namespace maximus {
// forward declarations
namespace gpu {
class GTable;
}  // namespace gpu

using ArrowTableBatchPtr = std::shared_ptr<arrow::RecordBatch>;
using ArrowTablePtr      = std::shared_ptr<arrow::Table>;
using AceroTableBatchPtr = std::shared_ptr<arrow::compute::ExecBatch>;
using GTablePtr          = std::shared_ptr<gpu::GTable>;
using CudfTablePtr       = std::shared_ptr<::cudf::table>;
using CudfTableView      = ::cudf::table_view;

enum class DeviceType : uint8_t { UNDEFINED, CPU, GPU };

std::string device_type_to_string(DeviceType type);

enum class PoolType { NON_PINNED, PINNED, DEFAULT };

class DeviceTablePtr {
    struct Empty {};

public:
    std::variant<Empty,
                 TableBatchPtr,
                 TablePtr,
                 ArrowTableBatchPtr,
                 ArrowTablePtr,
                 AceroTableBatchPtr,
                 GTablePtr,
                 CudfTablePtr>
        value = Empty();

    DeviceTablePtr() = default;

    DeviceTablePtr(const DeviceTablePtr &other)            = default;
    DeviceTablePtr &operator=(const DeviceTablePtr &other) = default;
    DeviceTablePtr(DeviceTablePtr &&other)                 = default;
    DeviceTablePtr &operator=(DeviceTablePtr &&other)      = default;

    explicit DeviceTablePtr(TableBatchPtr value): value(std::move(value)) {}

    explicit DeviceTablePtr(TablePtr value): value(std::move(value)) {}

    explicit DeviceTablePtr(ArrowTableBatchPtr value): value(std::move(value)) {}

    explicit DeviceTablePtr(ArrowTablePtr value): value(std::move(value)) {}

    explicit DeviceTablePtr(AceroTableBatchPtr value): value(std::move(value)) {}

    explicit DeviceTablePtr(GTablePtr value): value(std::move(value)) {}

    explicit DeviceTablePtr(CudfTablePtr value): value(std::move(value)) {}

    [[nodiscard]] bool is_table_batch() const;

    [[nodiscard]] bool is_table() const;

    [[nodiscard]] bool is_arrow_table_batch() const;

    [[nodiscard]] bool is_arrow_table() const;

    [[nodiscard]] bool is_acero_table_batch() const;

    [[nodiscard]] bool is_gtable() const;

    [[nodiscard]] bool is_ftable() const;

    [[nodiscard]] bool is_cudf_table() const;

    [[nodiscard]] bool is_none() const;

    [[nodiscard]] bool empty() const;

    [[nodiscard]] bool is_batch_type() const;

    [[nodiscard]] bool is_table_type() const;

    template<typename T>
    [[nodiscard]] bool is() const {
        // return whether the actual value inside std::variant is of type T
        return std::holds_alternative<T>(value);
    }

    // Conversion operator to bool
    explicit operator bool() const;

    [[nodiscard]] bool on_cpu() const;

    template<typename T>
    [[nodiscard]] bool type_on_cpu() const {
        // based on the template type T, check whether the actual value inside std::variant is on CPU
        return std::is_same_v<T, TableBatchPtr> || std::is_same_v<T, TablePtr> ||
               std::is_same_v<T, ArrowTableBatchPtr> || std::is_same_v<T, ArrowTablePtr> ||
               std::is_same_v<T, AceroTableBatchPtr>;
    }

    [[nodiscard]] bool on_gpu() const;

    template<typename T>
    [[nodiscard]] bool type_on_gpu() const {
        // based on the template type T, check whether the actual value inside std::variant is on GPU
        return std::is_same_v<T, GTablePtr> || std::is_same_v<T, CudfTablePtr>;
    }

    [[nodiscard]] bool on_device(const DeviceType &device_type) const;

    [[nodiscard]] TableBatchPtr as_table_batch() const;

    [[nodiscard]] TablePtr as_table() const;

    [[nodiscard]] ArrowTableBatchPtr as_arrow_table_batch() const;

    [[nodiscard]] ArrowTablePtr as_arrow_table() const;

    [[nodiscard]] AceroTableBatchPtr as_acero_table_batch() const;

    [[nodiscard]] GTablePtr as_gtable() const;

    [[nodiscard]] CudfTablePtr as_cudf_table() const;

    [[nodiscard]] TableBatchPtr as_cpu() const;

    [[nodiscard]] GTablePtr as_gpu() const;

    void to_cpu(std::shared_ptr<MaximusContext> &ctx, std::shared_ptr<Schema> &schema);
    void to_gpu(std::shared_ptr<MaximusContext> &ctx, std::shared_ptr<Schema> &schema);

    template<typename T>
    [[nodiscard]] T as() const {
        assert(!is_none());
        return std::get<T>(value);
    }

    std::string to_string() const;

    template<typename T>
    std::string type_to_string() const {
        if (std::is_same_v<T, TableBatchPtr>) {
            return "TableBatch";
        }
        if (std::is_same_v<T, TablePtr>) {
            return "Table";
        }
        if (std::is_same_v<T, ArrowTableBatchPtr>) {
            return "ArrowTableBatch";
        }
        if (std::is_same_v<T, ArrowTablePtr>) {
            return "ArrowTable";
        }
        if (std::is_same_v<T, AceroTableBatchPtr>) {
            return "AceroTableBatch";
        }
        if (std::is_same_v<T, GTablePtr>) {
            return "GTable";
        }
        if (std::is_same_v<T, CudfTablePtr>) {
            return "CudfTable";
        }
        throw std::runtime_error("Unsupported type");
    }

    template<typename T>
    void convert_to(const std::shared_ptr<MaximusContext> &ctx,
                    const std::shared_ptr<Schema> &schema,
                    PoolType pool_type = PoolType::DEFAULT) {
        std::string outer_region = "DataTransformation";
        std::string inner_region = "";
        std::string types_region = to_string() + "->" + type_to_string<T>();
        auto prefer_pinned       = false;
        if (on_cpu() && type_on_gpu<T>()) {
            inner_region  = "CPU->GPU";
            prefer_pinned = true;
        }
        if (on_gpu() && type_on_cpu<T>()) {
            inner_region  = "GPU->CPU";
            prefer_pinned = true;
        }
        if (on_cpu() && type_on_cpu<T>()) {
            inner_region = "CPU->CPU";
        }
        if (on_gpu() && type_on_gpu<T>()) {
            inner_region = "GPU->GPU";
        }
        assert(inner_region != "" && "inner_region must not be empty");

        arrow::MemoryPool *pool = ctx->get_memory_pool();

        switch (pool_type) {
            case PoolType::NON_PINNED:
                pool = ctx->get_memory_pool();
                break;
            case PoolType::PINNED:
                pool = ctx->get_pinned_memory_pool_if_available();
                break;
            case PoolType::DEFAULT:
                pool = prefer_pinned ? ctx->get_pinned_memory_pool_if_available()
                                     : ctx->get_memory_pool();
                break;
        }

        if (is_none()) {
            return;
        }

        // check whether the actual value inside std::variant is already of type T
        if (is<T>()) {
            return;
        }

        if (is_table()) {
            auto table = as_table();
            assert(table);

            if (std::is_same_v<T, ArrowTablePtr>) {
                profiler::open_regions({outer_region, inner_region, types_region});
                auto arrow_table = table->get_table();
                value            = std::move(arrow_table);
                profiler::close_regions({outer_region, inner_region, types_region});
                return;
            }

            // this might require full data-copy (if more than 1 batch)
            if (std::is_same_v<T, ArrowTableBatchPtr>) {
                profiler::open_regions({outer_region, inner_region, types_region});
                auto arrow_table = table->get_table();

                auto arrow_batch = to_record_batch(arrow_table, pool);

                // check  the schemas are equal
                // std::cout << "arrow batch schema = " << arrow_batch->schema()->ToString() << std::endl;
                // std::cout << "target schema = " << schema->get_schema()->ToString() << std::endl;
                assert(arrow_batch->schema()->Equals(*schema->get_schema()));

                value = std::move(arrow_batch);
                profiler::close_regions({outer_region, inner_region, types_region});
                return;
            }

            if (std::is_same_v<T, TableBatchPtr>) {
                // first convert it to arrow::RecordBatch and then to TableBatch
                convert_to<ArrowTableBatchPtr>(ctx, schema, pool_type);
                profiler::open_regions({outer_region, inner_region, types_region});
                auto arrow_batch = as_arrow_table_batch();

                TableBatchPtr table_batch;
                auto status =
                    TableBatch::from_record_batch(ctx, std::move(arrow_batch), table_batch);
                if (!status.ok()) {
                    CHECK_STATUS(status);
                }

                assert(*table_batch->get_schema() == *schema);

                value = std::move(table_batch);
                profiler::close_regions({outer_region, inner_region, types_region});
                return;
            }

            if (std::is_same_v<T, AceroTableBatchPtr>) {
                convert_to<TableBatchPtr>(ctx, schema, pool_type);
                convert_to<AceroTableBatchPtr>(ctx, schema, pool_type);
                return;
            }

            if (std::is_same_v<T, AceroTableBatchPtr>) {
                convert_to<TableBatchPtr>(ctx, schema, pool_type);
                convert_to<AceroTableBatchPtr>(ctx, schema, pool_type);
                return;
            }

#ifdef MAXIMUS_WITH_CUDA
            profiler::open_regions({outer_region, inner_region, types_region});
            assert(type_on_gpu<T>());

            auto gcontext = ctx->get_gpu_context();
            assert(gcontext);
            assert(as_table());
            auto arrow_table = as_table()->get_table();
            assert(arrow_table);

            // auto start = std::chrono::steady_clock::now();
            std::unique_ptr<::cudf::table> cudf_table = cudf_from_arrow_table(
                arrow_table, schema, ctx->get_h2d_stream(), ctx->mr, pool);
            assert(cudf_table);
            /*
            auto end = std::chrono::steady_clock::now();

            auto timing =
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            std::cout << "Time [ms] = " << timing << std::endl;
            */

            // ctx->h2d_stream.synchronize();
            ctx->tables_pending_copy.push_back(arrow_table);

            // Table -> CudfTable
            if (std::is_same_v<T, CudfTablePtr>) {
                value = std::move(cudf_table);
                profiler::close_regions({outer_region, inner_region, types_region});
                return;
            }

            // Table -> GTable
            if (std::is_same_v<T, GTablePtr>) {
                assert(*table->get_schema() == *schema);
                value = std::move(gpu::cudf_to_gtable(ctx, schema, std::move(cudf_table)));
                profiler::close_regions({outer_region, inner_region, types_region});
                return;
            }
#endif
            throw std::runtime_error(
                "Unsupported conversion from Table to an unsupported table type.");
        }

        if (is_table_batch()) {
            auto table_batch = as_table_batch();
            assert(table_batch);

            // TableBatch -> Table
            if (std::is_same_v<T, TablePtr>) {
                profiler::open_regions({outer_region, inner_region, types_region});
                TablePtr table;
                auto status = Table::from_table_batches(ctx, {table_batch}, table);

                CHECK_STATUS(status);
                value = std::move(table);
                profiler::close_regions({outer_region, inner_region, types_region});
                return;
            }

            if (std::is_same_v<T, ArrowTableBatchPtr>) {
                profiler::open_regions({outer_region, inner_region, types_region});
                ArrowTableBatchPtr arrow_table_batch;
                auto status = table_batch->to_record_batch(arrow_table_batch);
                if (!status.ok()) {
                    CHECK_STATUS(status);
                }
                assert(arrow_table_batch->schema()->Equals(*schema->get_schema()));
                value = std::move(arrow_table_batch);
                profiler::close_regions({outer_region, inner_region, types_region});
                return;
            }

            if (std::is_same_v<T, ArrowTablePtr>) {
                // first convert it to arrow::RecordBatch
                convert_to<ArrowTableBatchPtr>(ctx, schema, pool_type);
                profiler::open_regions({outer_region, inner_region, types_region});
                assert(is_arrow_table_batch());
                auto arrow_table_batch = as_arrow_table_batch();

                auto maybe_arrow_table =
                    arrow::Table::FromRecordBatches(schema->get_schema(), {arrow_table_batch});

                if (!maybe_arrow_table.ok()) {
                    CHECK_STATUS(maybe_arrow_table.status());
                }

                auto arrow_table = std::move(maybe_arrow_table.ValueOrDie());

                value = std::move(arrow_table);
                profiler::close_regions({outer_region, inner_region, types_region});
                return;
            }

            // TableBatch -> AceroTableBatch
            if (std::is_same_v<T, AceroTableBatchPtr>) {
                profiler::open_regions({outer_region, inner_region, types_region});
                AceroTableBatchPtr acero_table_batch;
                auto status = table_batch->to_exec_batch(acero_table_batch);
                CHECK_STATUS(status);
                value = std::move(acero_table_batch);
                profiler::close_regions({outer_region, inner_region, types_region});
                return;
            }

#ifdef MAXIMUS_WITH_CUDA
            if (type_on_gpu<T>()) {
                // first convert it to Table, then from Table -> T
                convert_to<TablePtr>(ctx, schema, PoolType::PINNED);
                convert_to<T>(ctx, schema, PoolType::DEFAULT);
                return;
            }
#endif

            throw std::runtime_error(
                "Unsupported conversion from TableBatch to an unsupported table type.");
        }

        if (is_arrow_table_batch()) {
            profiler::open_regions({outer_region, inner_region, types_region});
            // first convert it to TableBatch and then to the desired type
            auto arrow_table_batch = as_arrow_table_batch();
            assert(arrow_table_batch);

            TableBatchPtr table_batch;
            auto status =
                TableBatch::from_record_batch(ctx, std::move(arrow_table_batch), table_batch);

            if (!status.ok()) {
                CHECK_STATUS(status);
            }

            value = std::move(table_batch);
            profiler::close_regions({outer_region, inner_region, types_region});

            convert_to<T>(ctx, schema, pool_type);
            return;
        }

        if (is_arrow_table()) {
            profiler::open_regions({outer_region, inner_region, types_region});
            // first convert it to Table and then to the desired type
            auto arrow_table = as_arrow_table();
            assert(arrow_table);

            TablePtr table;
            auto status = Table::from_table(ctx, std::move(arrow_table), table);

            // std::cout << "Table after conversion: " << table->to_string() << std::endl;

            if (!status.ok()) {
                CHECK_STATUS(status);
            }
            // std::cout << "Table schema: " << table->get_schema()->to_string() << std::endl;
            // std::cout << "Schema: " << schema->to_string() << std::endl;
            // std::cout << "====================" << std::endl;
            assert(*table->get_schema() == *schema);
            value = std::move(table);
            profiler::close_regions({outer_region, inner_region, types_region});
            convert_to<T>(ctx, schema, pool_type);
            return;
        }

        if (is_acero_table_batch()) {
            profiler::open_regions({outer_region, inner_region, types_region});
            auto acero_table_batch = as_acero_table_batch();
            assert(acero_table_batch);

            // first step: convert acero table batch to table batch
            TableBatchPtr table_batch;

            if (type_on_gpu<T>()) pool = ctx->get_pinned_memory_pool_if_available();
            auto status =
                TableBatch::from_exec_batch(ctx, schema, acero_table_batch, pool, table_batch);

            value = std::move(table_batch);
            profiler::close_regions({outer_region, inner_region, types_region});

            // convert table batch to the desired type
            convert_to<T>(ctx, schema, pool_type);
            return;
        }

#ifdef MAXIMUS_WITH_CUDA
        if (is_gtable()) {
            auto gtable = as_gtable();
            assert(gtable);
            // TODO: there is a mismatch between cudf schema and arrow schema since nullability is ignored by cudf.
            // this means double not null -> double during arrow->cudf conversion
            // we have to compare the schema types, without the nullability comparison
            // assert(*gtable->get_schema() == *schema);

            profiler::open_regions({outer_region, inner_region, types_region});
            value = std::move(gpu::gtable_to_cudf(gtable));
            profiler::close_regions({outer_region, inner_region, types_region});
            convert_to<T>(ctx, schema, pool_type);
            return;
        }

        if (is_cudf_table()) {
            auto cudf_table = as_cudf_table();
            assert(cudf_table);

            if (std::is_same_v<T, GTablePtr>) {
                profiler::open_regions({outer_region, inner_region, types_region});
                value = std::move(
                    gpu::cudf_to_gtable(ctx, schema, std::move(cudf_table)));
                profiler::close_regions({outer_region, inner_region, types_region});
                return;
            }

            if (std::is_same_v<T, ArrowTableBatchPtr>) {
                profiler::open_regions({outer_region, inner_region, types_region});
                auto column_metadata = gpu::to_cudf_column_metadata(schema->get_schema());
                auto arrow_table     = cudf_to_arrow_table(cudf_table->view(),
                                                       schema,
                                                       ctx->get_d2h_stream(),
                                                       ctx->mr,
                                                       pool);
                auto arrow_batch     = to_record_batch(arrow_table, pool);
                value                = std::move(arrow_batch);
                profiler::close_regions({outer_region, inner_region, types_region});
                return;
            }

            if (std::is_same_v<T, ArrowTablePtr>) {
                profiler::open_regions({outer_region, inner_region, types_region});
                auto column_metadata = gpu::to_cudf_column_metadata(schema->get_schema());
                auto arrow_table     = cudf_to_arrow_table(cudf_table->view(),
                                                       schema,
                                                       ctx->get_d2h_stream(),
                                                       ctx->mr,
                                                       pool);
                value                = std::move(arrow_table);
                profiler::close_regions({outer_region, inner_region, types_region});
                return;
            }

            if (type_on_cpu<T>()) {
                pool_type = PoolType::PINNED;
                // for all other conversions, first convert to ArrowTableBatchPtr and then to the desired type
                convert_to<ArrowTablePtr>(ctx, schema, pool_type);
                convert_to<T>(ctx, schema, pool_type);
            }

            if (type_on_gpu<T>()) {
                // for all other conversions, first convert to GTable and then to the desired type
                convert_to<GTablePtr>(ctx, schema, pool_type);
                convert_to<T>(ctx, schema, pool_type);
            }
            return;
        }
#endif

        throw std::runtime_error(
            "Unsupported conversion from an unsupported table type to another table type.");
    }
};

DeviceTablePtr merge_batches(std::shared_ptr<MaximusContext> &ctx,
                             std::vector<DeviceTablePtr> &batches,
                             std::shared_ptr<Schema> schema);

}  // namespace maximus
