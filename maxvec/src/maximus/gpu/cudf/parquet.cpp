#include <maximus/gpu/cudf/parquet.hpp>
#include <maximus/profiler/profiler.hpp>
#include <maximus/utils/utils.hpp>
// WIP 3
// #include <cudf/unary.hpp>
// #include <cudf/lists/lists_column_view.hpp>
#include <typeinfo>

namespace maximus::gpu {

Status read_parquet_cudf(std::shared_ptr<MaximusContext>& ctx,
                         const std::string& path,
                         const std::shared_ptr<Schema>& schema,
                         const std::vector<std::string>& include_columns,
                         GTablePtr& gtable) {
    PE("IO");
    PE("read_parquet_cudf");

    assert(ends_with(path, ".parquet") && "read_parquet(...) only supports parquet files");
    assert(ctx && "ctx cannot be nullptr");

    auto stream = ctx->get_h2d_stream();
    auto* mr = &ctx->pool_mr;

    cudf::io::source_info source(path);
    cudf::io::parquet_reader_options_builder options =
        cudf::io::parquet_reader_options::builder(source);

    if (!include_columns.empty()) {
        options.columns(include_columns);
    }

    if (schema) {
        std::vector<cudf::io::reader_column_schema> column_schemas;

        // Get the column types from the schema
        auto schema_types = schema->column_types();  // map<string, arrow::DataType>
        
        // CRITICAL: When include_columns is specified, we must build column_schemas
        // in the same order as include_columns (which is passed to cuDF's options.columns()).
        // Otherwise, cuDF will expect schema info for all columns but only read included ones.
        std::vector<std::string> parquet_column_order = 
            include_columns.empty() ? schema->column_names() : include_columns;

        // std::cout << "Building reader_column_schema for Parquet read:\n";

        for (const auto& col_name : parquet_column_order) {
            auto arrow_type = schema_types.at(col_name);
            cudf::io::reader_column_schema col_schema;

            if (arrow_type->id() == arrow::Type::LIST || arrow_type->id() == arrow::Type::LARGE_LIST) {
                auto value_type = (arrow_type->id() == arrow::Type::LIST)
                                    ? std::static_pointer_cast<arrow::ListType>(arrow_type)->value_type()
                                    : std::static_pointer_cast<arrow::LargeListType>(arrow_type)->value_type();

                // Child schema
                cudf::io::reader_column_schema child_schema;
                child_schema.set_type_length(value_type->byte_width());

                // Parent LIST schema
                cudf::io::reader_column_schema list_schema;
                list_schema.add_child(child_schema);

                column_schemas.push_back(std::move(list_schema));
                // std::cout << "LIST column: " << col_name
                //          << ", element type: " << value_type->ToString()
                //          << ", child type_length: " << child_schema.get_type_length() << "\n";

            } else if (arrow_type->id() == arrow::Type::STRING) {
                col_schema.set_type_length(4);
                // std::cout << "STRING column: " << col_name << ", type_length = 4\n";
                column_schemas.push_back(std::move(col_schema));

            } else {
                col_schema.set_type_length(arrow_type->byte_width());
                // std::cout << "PRIMITIVE column: " << col_name
                //           << ", type = " << arrow_type->ToString()
                //           << ", type_length = " << col_schema.get_type_length() << "\n";
                column_schemas.push_back(std::move(col_schema));
            }
        }

        if (!column_schemas.empty()) {
            options.set_column_schema(column_schemas);
        }
    }

    // std::cout << "Reading Parquet file: " << path << "\n";
    CudfTablePtr cudf_table;

    try {
        cudf::io::table_with_metadata result = cudf::io::read_parquet(options, stream, mr);
        cudf_table = std::move(result.tbl);
        // std::cout << "Parquet read successful!\n";
    } catch (const std::exception& e) {
        std::cerr << "Error reading Parquet file: " << e.what() << "\n";
        throw;
    }

    // Create the GTable with the correct schema
    // If include_columns was specified, use the subschema matching those columns
    std::shared_ptr<Schema> gtable_schema = schema;
    if (!include_columns.empty() && schema) {
        gtable_schema = schema->subschema(include_columns);
    }
    gtable = std::make_shared<GTable>(ctx, gtable_schema, std::move(cudf_table));

    PL("read_parquet_cudf");
    PL("IO");
    return Status::OK();
}

Status write_parquet_cudf(const std::string& path,
                          GTablePtr& table,
                          std::unique_ptr<std::vector<uint8_t>>& result) {
    PE("IO");
    PE("write_parquet_cudf");
    // auto start_time = std::chrono::high_resolution_clock::now();
    assert(ends_with(path, ".parquet") && "write_parquet(...) only supports parquet files");

    auto ctx = table->get_context();
    auto stream = ctx->get_d2h_stream();

    cudf::io::sink_info sink = cudf::io::sink_info(path);

    // std::shared_ptr<cudf::table_view> cudf_table = gtable_to_cudf_view(table);

    cudf::io::parquet_writer_options_builder options =
        cudf::io::parquet_writer_options::builder(sink, *table->get_table());

    result = cudf::io::write_parquet(options, stream);

    PL("write_parquet_cudf");
    PL("IO");
    return Status::OK();
}

}  // namespace maximus::gpu
