#pragma once

#include <stdint.h>
#include <memory>
#include <cudf/copying.hpp>

#include <maximus/utils/cuda_helpers.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace maximus {

std::unique_ptr<cudf::column> make_sequence_column(
        const cudf::size_type size,
        rmm::mr::device_memory_resource* mr,
        cudaStream_t stream);

std::unique_ptr<cudf::column> make_sequence_column(
    const cudf::size_type size,
    const int64_t step,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream);

std::unique_ptr<cudf::column> make_sequence_column(
    const int64_t start,
    const int64_t step,
    const int64_t end,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream);

// creates a cudf column from the given host std::vector
template <typename T>
std::unique_ptr<cudf::column> make_device_column(
    std::vector<T> const& host_data,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
        using namespace cudf;

        auto col = cudf::make_fixed_width_column(
            data_type{type_to_id<T>()},
            host_data.size(),
            mask_state::UNALLOCATED,
            stream,
            mr);

        assert(col->size() == static_cast<cudf::size_type>(host_data.size()) &&
                     "Column allocation size mismatch");

        // Copy host → device
        copy_h2d_async(host_data.data(),
                       col->mutable_view().template data<T>(),
                       host_data.size(),
                       stream);

        return col;
}

template <typename T>
T const* get_device_data_ptr(cudf::column_view const& col)
{
    return col.data<T>() + col.offset();
}

template <typename T>
T* get_device_data_ptr(cudf::column& col)
{
    auto view = col.mutable_view();
    return view.data<T>() + view.offset();
}

template <typename T>
std::vector<T> copy_device_column_to_host(
    cudf::column_view const& col,
    cudaStream_t stream)
{
    assert(col.type().id() == cudf::type_to_id<T>() &&
                 "Column type does not match template type T");

    std::vector<T> host(col.size());

    // Copy device → host
    copy_d2h_async(col.data<T>(),
                   host.data(),
                   col.size(),
                   stream);

    return host;
}

/**
 * Gather a single column from a table using a gather map.
 *
 * Equivalent to:
 *   gather(table[[column_index]], gather_map)
 *
 * @param table        Source table
 * @param column_index Index of column to gather
 * @param gather_map   Row indices to gather
 * @return             Gathered column (size = gather_map.size())
 */
std::unique_ptr<cudf::column> gather_column(
    cudf::table_view const& table,
    int column_index,
    cudf::column_view const& gather_map,
    cudf::out_of_bounds_policy bounds_policy,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

}