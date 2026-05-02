#pragma once

#include <cudf/io/csv.hpp>
#include <maximus/gpu/cuda_api.hpp>
#include <maximus/types/device_table_ptr.hpp>

namespace maximus {

namespace gpu {

// if the schema is given, it will be used to set the column names and types
Status read_csv_cudf(std::shared_ptr<MaximusContext>& ctx,
                     const std::string& path,
                     const std::shared_ptr<Schema>& schema,
                     const std::vector<std::string>& include_columns,
                     GTablePtr& table);

Status write_csv_cudf(const std::string& path, GTablePtr& table);

}  // namespace gpu

}  // namespace maximus