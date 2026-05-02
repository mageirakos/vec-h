#pragma once

#include <maximus/context.hpp>
#include <maximus/types/table.hpp>
#include <maximus/utils/utils.hpp>

namespace maximus {

Status streaming_reader_parquet(const std::shared_ptr<MaximusContext>& ctx,
                                const std::string& path,
                                const std::shared_ptr<Schema>& schema,
                                std::vector<std::string> include_columns,
                                std::shared_ptr<arrow::RecordBatchReader>& reader);

Status read_parquet(const std::shared_ptr<MaximusContext>& ctx,
                    const std::string& path,
                    const std::shared_ptr<Schema>& schema,
                    std::vector<std::string> include_columns,
                    TablePtr& tableOut);

}  // namespace maximus