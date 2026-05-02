#include <iostream>
#include <maximus/io/csv.hpp>
#include <maximus/io/file.hpp>
#include <maximus/utils/utils.hpp>

namespace maximus {

Status streaming_reader_csv(const std::shared_ptr<MaximusContext> &ctx,
                            const std::string &path,
                            const std::shared_ptr<Schema> &schema,
                            const std::vector<std::string> &include_columns,
                            std::shared_ptr<arrow::RecordBatchReader> &reader) {
    assert(ctx);
    auto pool = ctx->get_memory_pool();
    assert(pool);
    auto io_context = ctx->get_io_context();
    assert(io_context);

    std::shared_ptr<arrow::io::RandomAccessFile> input;
    auto status = input_file(ctx, path, input);
    if (!status.ok()) {
        return status;
    }

    assert(ctx->csv_batch_size > 0);
    CSVOptions csv_options(schema, include_columns, ctx->csv_batch_size);

    auto maybe_reader = arrow::csv::StreamingReader::Make(*io_context,
                                                          input,
                                                          csv_options.read_options,
                                                          csv_options.parse_options,
                                                          csv_options.convert_options);

    if (!maybe_reader.ok()) {
        return Status(ErrorCode::ArrowError, maybe_reader.status().message());
    }

    reader = maybe_reader.ValueOrDie();
    return Status::OK();
}

// the schema can be nullptr here
Status read_csv(const std::shared_ptr<MaximusContext> &ctx,
                const std::string &path,
                const std::shared_ptr<Schema> &schema,
                const std::vector<std::string> &include_columns,
                TablePtr &table) {
    // std::cout << "read_csv(" << path << ")\n";
    PE("IO");
    PE("read_csv");
    // auto start_time = std::chrono::high_resolution_clock::now();
    assert(ends_with(path, ".csv") && "read_csv(...) only supports csv files");

    assert(ctx);
    auto pool = ctx->get_memory_pool();
    assert(pool);
    auto io_context = ctx->get_io_context();
    assert(io_context);

    std::shared_ptr<arrow::io::RandomAccessFile> input;
    auto status = input_file(ctx, path, input);
    if (!status.ok()) {
        return status;
    }

    assert(ctx->csv_batch_size > 0);
    CSVOptions csv_options = CSVOptions(schema, include_columns, ctx->csv_batch_size);

    const auto maybe_reader = arrow::csv::TableReader::Make(*io_context,
                                                            input,
                                                            csv_options.read_options,
                                                            csv_options.parse_options,
                                                            csv_options.convert_options);

    if (!maybe_reader.ok()) {
        return Status(ErrorCode::ArrowError, maybe_reader.status().message());
    }

    auto reader = maybe_reader.ValueOrDie();

    auto maybe_table = reader->Read();
    if (!maybe_table.ok()) {
        return Status(ErrorCode::ArrowError, maybe_table.status().message());
    }

    table = std::make_shared<Table>(ctx, maybe_table.ValueOrDie());

    assert(include_columns.empty() || table->num_columns() == include_columns.size());

    // auto end_time = std::chrono::high_resolution_clock::now();

    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // std::cout << "CPU: read_csv = " << duration << std::endl;

    PL("read_csv");
    PL("IO");
    return Status::OK();
}
}  // namespace maximus
