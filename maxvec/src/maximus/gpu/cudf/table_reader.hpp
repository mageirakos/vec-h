#pragma once

#include <cudf/table/table.hpp>
#include <maximus/gpu/cudf/cudf_types.hpp>
#include <maximus/gpu/gtable/cuda/cuda_buffer.hpp>
#include <maximus/gpu/gtable/gtable.hpp>

namespace maximus {

namespace gpu {

/**
 * To convert a GTable to a cudf table
 */
std::shared_ptr<cudf::table> gtable_to_cudf(std::shared_ptr<GTable> tab);

/**
 * To read a GTable as a cudf table view
 */
std::shared_ptr<cudf::table_view> gtable_to_cudf_view(std::shared_ptr<GTable> tab);

}  // namespace gpu
}  // namespace maximus