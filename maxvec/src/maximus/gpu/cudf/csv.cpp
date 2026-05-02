#include <maximus/gpu/cudf/csv.hpp>
#include <maximus/profiler/profiler.hpp>
#include <maximus/utils/utils.hpp>
#include <typeinfo>

namespace maximus::gpu {

Status read_csv_cudf(std::shared_ptr<MaximusContext>& ctx,
                     const std::string& path,
                     const std::shared_ptr<Schema>& schema,
                     const std::vector<std::string>& include_columns,
                     GTablePtr& table) {
    PE("IO");
    PE("read_csv_cudf");
    // auto start_time = std::chrono::high_resolution_clock::now();
    assert(ends_with(path, ".csv") && "read_csv(...) only supports csv files");
    assert(ctx != nullptr && "ctx cannot be nullptr");

    auto stream = ctx->get_h2d_stream();
    auto* mr = &ctx->pool_mr;

    cudf::io::source_info source = cudf::io::source_info(path);

    cudf::io::csv_reader_options_builder options = cudf::io::csv_reader_options::builder(source);

    if (!include_columns.empty()) {
        options.use_cols_names(include_columns);
    }

    if (schema) {
        std::map<std::string, cudf::data_type> column_types;
        std::unordered_map<std::string, std::shared_ptr<arrow::DataType>> schema_types =
            schema->column_types();
        std::transform(
            schema_types.begin(),
            schema_types.end(),
            std::inserter(column_types, column_types.end()),
            [](const std::pair<std::string, std::shared_ptr<arrow::DataType>>& schema_type) {
                return std::make_pair(schema_type.first, to_cudf_type(schema_type.second));
            });
        options.dtypes(column_types);
    }

    // std::cout << (schema == nullptr) << std::endl;
    cudf::io::table_with_metadata result = cudf::io::read_csv(options, stream, mr);

    arrow::FieldVector out_arrow_schema;
    cudf::table_view out_view = result.tbl->view();
    for (int i = 0; i < result.tbl->num_columns(); ++i) {
        out_arrow_schema.push_back(std::make_shared<arrow::Field>(
            result.metadata.schema_info[i].name, to_arrow_type(result.tbl->get_column(i).type())));
    }
    std::shared_ptr<Schema> out_schema =
        std::make_shared<Schema>(std::make_shared<arrow::Schema>(out_arrow_schema));

    table = cudf_to_gtable(ctx, out_schema, std::move(result.tbl));

    // auto end_time = std::chrono::high_resolution_clock::now();

    //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // std::cout << "GPU: read_csv = " << duration << std::endl;
    PL("read_csv_cudf");
    PL("IO");
    return Status::OK();
}

Status write_csv_cudf(const std::string& path, GTablePtr& table) {
    PE("IO");
    PE("write_csv_cudf");
    // auto start_time = std::chrono::high_resolution_clock::now();
    assert(ends_with(path, ".csv") && "write_csv(...) only supports csv files");

    auto ctx = table->get_context();
    auto stream = ctx->get_d2h_stream();

    cudf::io::sink_info sink = cudf::io::sink_info(path);

    auto cudf_table = gtable_to_cudf_view(table);

    cudf::io::csv_writer_options_builder options =
        cudf::io::csv_writer_options::builder(sink, *cudf_table);

    cudf::io::write_csv(options, stream);

    PL("write_csv_cudf");
    PL("IO");
    return Status::OK();
}

}  // namespace maximus::gpu
