#pragma once

#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <maximus/types/types.hpp>
#include <memory>

namespace maximus {

class Schema;

namespace gpu {

/**
 * To convert a Maximus DataType to a cudf data type
 */
cudf::data_type to_cudf_type(const std::shared_ptr<DataType> &type);

/**
 * To convert a cudf data type to a Maximus DataType
 */
std::shared_ptr<DataType> to_maximus_type(const cudf::data_type &type);

/**
 * To convert a cudf data type to an Arrow DataType
 */
cudf::data_type to_cudf_type(const std::shared_ptr<arrow::DataType> &type);

/**
 * To convert an arrow data type to a cudf data type
*/
std::shared_ptr<arrow::DataType> to_arrow_type(const cudf::data_type &type);

/**
* Convert arrow::Schema to cudf::column_metadata
*/
std::vector<cudf::column_metadata> to_cudf_column_metadata(
    const std::shared_ptr<arrow::Schema> &schema);

/**
 * Create an empty cudf::table (0 rows) with column types matching the given schema.
 */
std::shared_ptr<cudf::table> make_empty_cudf_table(const std::shared_ptr<Schema> &schema);

}  // namespace gpu
}  // namespace maximus
