#include <arrow/util/thread_pool.h>

#include <maximus/frontend/dataset_api.hpp>

namespace maximus {

// 1. Determine file format from path
arrow::Result<std::shared_ptr<ds::FileFormat>> get_file_format(const std::string& path) {
    if (ends_with(path, ".parquet")) {
        return std::make_shared<ds::ParquetFileFormat>();
    } else if (ends_with(path, ".csv")) {
        auto csv_format = std::make_shared<ds::CsvFileFormat>();
        return csv_format;
    }
    return arrow::Status::Invalid("Unsupported file format: ", path);
}

// 2. Full pipeline: file → dataset → filtered/projected table
// arrow::Result<std::shared_ptr<arrow::Table>> table_source_filter_project(
arrow::Result<std::shared_ptr<ds::Scanner>> build_scanner_from_files(
    const std::shared_ptr<MaximusContext>& ctx,
    const std::vector<std::string>& file_paths,
    const std::shared_ptr<arrow::Schema>& schema,
    const std::vector<std::string>& include_columns,
    const std::shared_ptr<Expression>& filter_expr,
    const std::vector<std::shared_ptr<Expression>>& exprs,
    const std::vector<std::string>& column_names) {
    assert(schema);
    if (file_paths.empty()) {
        return arrow::Status::Invalid("No file paths provided.");
    }

    assert(ctx && "The context must not be null in build_scanner_from_files.");
    auto pool = ctx->get_memory_pool();
    assert(pool && "The memory pool must not be null in build_scanner_from_files.");
    auto io_context = ctx->get_io_context();
    assert(io_context && "The io context must not be null in build_scanner_from_files.");

    assert(exprs.size() == column_names.size());
    std::vector<std::string> included_columns = include_columns;
    if (included_columns.empty()) {
        for (const auto& field : schema->fields()) {
            included_columns.push_back(field->name());
        }
    }

    // Use the first file to derive the file system and format
    const std::string& representative_file = file_paths[0];
    ARROW_ASSIGN_OR_RAISE(auto fs, arrow::fs::FileSystemFromUriOrPath(representative_file));
    ARROW_ASSIGN_OR_RAISE(auto format, get_file_format(representative_file));

    // Normalize all paths
    std::vector<std::string> normalized_paths;
    for (const auto& path : file_paths) {
        ARROW_ASSIGN_OR_RAISE(auto info, fs->GetFileInfo(path));
        if (info.type() == arrow::fs::FileType::File) {
            normalized_paths.push_back(info.path());
        }
    }
    if (normalized_paths.empty()) {
        return arrow::Status::IOError("No valid input files found.");
    }

    ds::FileSystemFactoryOptions options;
    ARROW_ASSIGN_OR_RAISE(
        auto factory, ds::FileSystemDatasetFactory::Make(fs, normalized_paths, format, options));
    ARROW_ASSIGN_OR_RAISE(auto dataset, factory->Finish(schema));
    ARROW_ASSIGN_OR_RAISE(auto scanner_builder, dataset->NewScan());

    ARROW_RETURN_NOT_OK(scanner_builder->Pool(pool));

    ARROW_ASSIGN_OR_RAISE(auto thread_pool, arrow::internal::ThreadPool::Make(ctx->n_io_threads));
    ARROW_ASSIGN_OR_RAISE(auto scan_options, scanner_builder->GetScanOptions());
    scan_options->io_context = *io_context;

    // Apply filter if present
    if (filter_expr) {
        auto arrow_filter = filter_expr->get_expression();
        if (arrow_filter && !arrow_filter->Equals(cp::literal(true))) {
            ARROW_RETURN_NOT_OK(scanner_builder->Filter(*arrow_filter));
        }
    }

    // Apply projection
    if (!exprs.empty()) {
        std::vector<cp::Expression> projections;
        for (const auto& e : exprs) {
            projections.push_back(*e->get_expression());
        }
        ARROW_RETURN_NOT_OK(scanner_builder->Project(projections, column_names));
    } else {
        std::vector<cp::Expression> proj_exprs;
        proj_exprs.reserve(included_columns.size());
        for (const auto& col : included_columns) {
            auto expr = Expression::from_field_ref(col);
            proj_exprs.push_back(*expr->get_expression());
        }
        ARROW_RETURN_NOT_OK(scanner_builder->Project(proj_exprs, included_columns));
    }

    ARROW_RETURN_NOT_OK(scanner_builder->UseThreads(true));
    return scanner_builder->Finish();
}

}  // namespace maximus
