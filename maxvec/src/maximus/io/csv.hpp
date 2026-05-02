#pragma once
#include <arrow/csv/api.h>

#include <maximus/context.hpp>
#include <maximus/types/table.hpp>

namespace maximus {

struct CSVOptions {
    CSVOptions(const std::shared_ptr<Schema>& schema,
               const std::vector<std::string>& include_columns,
               const int32_t batch_size)
            : schema(schema), include_columns(include_columns) {
        read_options    = arrow::csv::ReadOptions::Defaults();
        parse_options   = arrow::csv::ParseOptions::Defaults();
        convert_options = arrow::csv::ConvertOptions::Defaults();

        read_options.use_threads = true;
        read_options.block_size  = batch_size;

        // if schema is not nullptr, we will use it to set the column names and types
        if (this->schema) {
            convert_options.column_types = this->schema->column_types();
        }

        if (!this->include_columns.empty()) {
            convert_options.include_columns = this->include_columns;
        }
    }

    std::shared_ptr<Schema> schema;
    std::vector<std::string> include_columns;
    arrow::csv::ReadOptions read_options;
    arrow::csv::ParseOptions parse_options;
    arrow::csv::ConvertOptions convert_options;
};

// if the schema is given, it will be used to set the column names and types
Status streaming_reader_csv(const std::shared_ptr<MaximusContext>& ctx,
                            const std::string& path,
                            const std::shared_ptr<Schema>& schema,
                            const std::vector<std::string>& include_columns,
                            std::shared_ptr<arrow::RecordBatchReader>& reader);

// if the schema is given, it will be used to set the column names and types
Status read_csv(const std::shared_ptr<MaximusContext>& ctx,
                const std::string& path,
                const std::shared_ptr<Schema>& schema,
                const std::vector<std::string>& include_columns,
                TablePtr& table);

}  // namespace maximus