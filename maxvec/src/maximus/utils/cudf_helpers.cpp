#include <maximus/utils/cudf_helpers.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/filling.hpp>

std::unique_ptr<cudf::column> maximus::make_sequence_column(
        const cudf::size_type size,
        rmm::mr::device_memory_resource* mr,
        cudaStream_t stream) {
    rmm::cuda_stream_view stream_view{stream};
    rmm::device_async_resource_ref mr_ref{mr};

    // Start value = 0
    cudf::numeric_scalar<int64_t> start(0, true, stream_view);

    // Use the overload that defaults step = 1
    return cudf::sequence(
        size,
        start,
        stream_view,
        mr_ref
    );
}

std::unique_ptr<cudf::column> maximus::make_sequence_column(
    const cudf::size_type size,
    const int64_t step,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
    rmm::cuda_stream_view stream_view{stream};
    rmm::device_async_resource_ref mr_ref{mr};

    // Start = 0
    cudf::numeric_scalar<int64_t> start_literal(0, true, stream_view);
    cudf::numeric_scalar<int64_t> step_literal(step, true, stream_view);


    // Use cuDF sequence overload with step
    return cudf::sequence(
        size,   // number of elements
        start_literal,
        step_literal,   // step parameter
        stream_view,
        mr_ref
    );
}

std::unique_ptr<cudf::column> maximus::make_sequence_column(
    const int64_t start,
    const int64_t step,
    const int64_t end,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
    rmm::cuda_stream_view stream_view{stream};
    rmm::device_async_resource_ref mr_ref{mr};

    // Compute size
    int64_t size = (end - start) / step + 1;
    if (size <= 0) {
        throw std::runtime_error("Invalid parameters for make_sequence_column: start/end/step combination produces non-positive size");
    }

    // cuDF numeric scalars for start and step
    cudf::numeric_scalar<int64_t> start_scalar(start, true, stream_view);
    cudf::numeric_scalar<int64_t> step_scalar(step, true, stream_view);

    // Generate sequence
    return cudf::sequence(
        static_cast<cudf::size_type>(size),
        start_scalar,
        step_scalar,
        stream_view,
        mr_ref
    );
}

std::unique_ptr<cudf::column> maximus::gather_column(
    cudf::table_view const& table,
    int column_index,
    cudf::column_view const& gather_map,
    cudf::out_of_bounds_policy bounds_policy,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
{
    cudf::table_view src_tv{
        std::vector<cudf::column_view>{ table.column(column_index) }
    };

    auto gathered = cudf::gather(
        src_tv, gather_map, bounds_policy, stream, mr);

    return std::move(gathered->release()[0]);
}