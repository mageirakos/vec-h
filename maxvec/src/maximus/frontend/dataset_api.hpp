#pragma once
#include <arrow/compute/api.h>  // Expressions, compute kernels
#include <arrow/dataset/api.h>  // Core Dataset API
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/file_base.h>     // FileSource, FileFormat base
#include <arrow/dataset/file_csv.h>      // CsvFileFormat (optional, if using CSV)
#include <arrow/dataset/file_parquet.h>  // ParquetFileFormat
#include <arrow/dataset/scanner.h>       // Scanner, ScanOptions
#include <arrow/filesystem/api.h>        // Filesystem abstraction (for paths)
#include <arrow/io/api.h>                // Input/output streams
#include <arrow/result.h>                // arrow::Result
#include <arrow/status.h>                // arrow::Status
#include <arrow/table.h>                 // arrow::Table

#include <maximus/types/expression.hpp>  // Expression wrapper
#include <maximus/utils/utils.hpp>

namespace maximus {

namespace ds = arrow::dataset;
namespace cp = arrow::compute;

// 1. Determine file format from path
arrow::Result<std::shared_ptr<ds::FileFormat>> get_file_format(const std::string& path);

// 2. Full pipeline: file → dataset → filtered/projected table
arrow::Result<std::shared_ptr<ds::Scanner>> build_scanner_from_files(
    const std::shared_ptr<MaximusContext>& ctx,
    const std::vector<std::string>& file_paths,
    const std::shared_ptr<arrow::Schema>& schema,
    const std::vector<std::string>& include_columns,
    const std::shared_ptr<Expression>& filter_expr,
    const std::vector<std::shared_ptr<Expression>>& exprs,
    const std::vector<std::string>& column_names);

}  // namespace maximus
