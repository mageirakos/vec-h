#pragma once

#include <cudf/io/parquet.hpp>
#include <maximus/gpu/cuda_api.hpp>
#include <maximus/types/device_table_ptr.hpp>

namespace maximus {

namespace gpu {

// if the schema is given, it will be used to set the column names and types
Status read_parquet_cudf(std::shared_ptr<MaximusContext>& ctx,
                         const std::string& path,
                         const std::shared_ptr<Schema>& schema,
                         const std::vector<std::string>& include_columns,
                         GTablePtr& table);

Status write_parquet_cudf(const std::string& path,
                          GTablePtr& table,
                          std::unique_ptr<std::vector<uint8_t>>& result);

}  // namespace gpu

}  // namespace maximus