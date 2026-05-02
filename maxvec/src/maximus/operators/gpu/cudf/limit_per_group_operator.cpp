#include <cudf/copying.hpp>
#include <cudf/concatenate.hpp>
#include <maximus/gpu/cuda_api.hpp>
#include <maximus/operators/gpu/cudf/limit_per_group_operator.hpp>
#include <maximus/utils/cuda_helpers.hpp>
#include <typeinfo>

namespace maximus::cudf {

LimitPerGroupOperator::LimitPerGroupOperator(
    std::shared_ptr<MaximusContext>& _ctx,
    std::shared_ptr<Schema> _input_schema,
    std::shared_ptr<LimitPerGroupProperties> _properties)
        : AbstractLimitPerGroupOperator(_ctx, _input_schema, std::move(_properties))
        , GpuOperator(_ctx, {_input_schema}, get_id(), {0})  // Port 0 is BLOCKING
{
    assert(ctx_);
    auto gctx = ctx_->get_gpu_context();

    assert(gctx != nullptr && "MaximusGContext must be initialized "
                              "before creating a GPU LimitPerGroupOperator");

    assert(typeid(*gctx) == typeid(maximus::gpu::MaximusCudaContext) &&
           "MaximusGContext must be a MaximusCudaContext");

    output_schema = std::make_shared<Schema>(*input_schemas[0]);

    set_device_type(DeviceType::GPU);
    set_engine_type(EngineType::CUDF);

    operator_name = name();
}

void LimitPerGroupOperator::on_add_input(DeviceTablePtr device_input, int port) {
    proxy_add_input(device_input, port);
}

void LimitPerGroupOperator::run_kernel(std::shared_ptr<MaximusContext>& ctx,
                                       std::vector<CudfTablePtr>& input_tables,
                                       std::vector<CudfTablePtr>& output_tables) {
    assert(input_tables.size() == 1);
    if (!input_tables[0]) return;

    auto& input = input_tables[0];
    auto input_view = input->view();
    int64_t num_rows = input_view.num_rows();

    if (num_rows == 0) return;

    // Find key column index in schema
    int key_idx = -1;
    auto schema = input_schemas[0]->get_schema();
    for (int i = 0; i < schema->num_fields(); ++i) {
        if (schema->field(i)->name() == properties->group_key) {
            key_idx = i;
            break;
        }
    }
    if (key_idx < 0) {
        throw std::runtime_error(
            "LimitPerGroupOperator: group key column not found: " + properties->group_key);
    }

    auto key_col = input_view.column(key_idx);

    // Copy key column to host for group boundary detection
    profiler::close_regions({operator_name, "no_more_input"});
    profiler::open_regions({"DataTransformation", "GPU->CPU", "key_column_copy"});

    std::vector<int64_t> host_keys(key_col.size());
    maximus::copy_d2h_async(
        key_col.data<int64_t>(),
        host_keys.data(),
        key_col.size(),
        ctx->get_d2h_stream().value());
    ctx->wait_d2h_copy();

    profiler::close_regions({"DataTransformation", "GPU->CPU", "key_column_copy"});
    profiler::open_regions({operator_name, "no_more_input"});

    // Find contiguous group boundaries and compute per-group slice ranges
    int64_t limit_k = properties->limit_k;
    std::vector<std::pair<::cudf::size_type, ::cudf::size_type>> kept_ranges;

    int64_t group_start = 0;
    for (int64_t i = 0; i <= num_rows; ++i) {
        bool is_boundary = (i == num_rows) || (i > 0 && host_keys[i] != host_keys[i - 1]);
        if (is_boundary) {
            // Emit kept range for the group [group_start, i)
            int64_t group_len = i - group_start;
            int64_t keep = std::min(group_len, limit_k);
            if (keep > 0) {
                kept_ranges.emplace_back(
                    static_cast<::cudf::size_type>(group_start),
                    static_cast<::cudf::size_type>(group_start + keep));
            }
            group_start = i;
        }
    }

    if (kept_ranges.empty()) return;

    // Use cudf::slice to extract kept rows (zero-copy views)
    // Then create owned tables from the views and concatenate
    if (kept_ranges.size() == 1 &&
        kept_ranges[0].first == 0 &&
        kept_ranges[0].second == num_rows) {
        // All rows kept — pass through
        output_tables.emplace_back(std::move(input));
        return;
    }

    // Build slice indices for cudf::slice (pairs of [begin, end])
    std::vector<::cudf::size_type> slice_indices;
    slice_indices.reserve(kept_ranges.size() * 2);
    for (auto& [start, end] : kept_ranges) {
        slice_indices.push_back(start);
        slice_indices.push_back(end);
    }

    auto slices = ::cudf::slice(input_view, slice_indices);

    if (slices.size() == 1) {
        output_tables.emplace_back(std::make_shared<::cudf::table>(slices[0]));
    } else {
        // Create owned tables from slice views, then concatenate
        std::vector<::cudf::table_view> table_views;
        table_views.reserve(slices.size());
        for (auto& sv : slices) {
            if (sv.num_rows() > 0) {
                table_views.push_back(sv);
            }
        }
        if (!table_views.empty()) {
            output_tables.emplace_back(
                ::cudf::concatenate(::cudf::host_span<::cudf::table_view const>(table_views)));
        }
    }
}

void LimitPerGroupOperator::on_no_more_input(int port) {
    proxy_no_more_input(port);
}

bool LimitPerGroupOperator::has_more_batches_impl(bool blocking) {
    return proxy_has_more_batches(blocking);
}

DeviceTablePtr LimitPerGroupOperator::export_next_batch_impl() {
    return std::move(proxy_export_next_batch());
}

}  // namespace maximus::cudf
