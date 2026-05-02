#include <maximus/gpu/gtable/gtable.hpp>

namespace maximus::gpu {
GTable::GTable(const std::shared_ptr<MaximusContext>& ctx,
    std::shared_ptr<Schema> schema,
    std::unique_ptr<cudf::table>&& table): ctx_(ctx),
                                         schema_(std::move(schema)),
                                         cudf_table_(std::move(table)) {}

GTable::GTable(const std::shared_ptr<MaximusContext>& ctx,
    std::shared_ptr<Schema> schema,
    std::shared_ptr<cudf::table>&& table): ctx_(ctx),
                                         schema_(std::move(schema)),
                                         cudf_table_(std::move(table)) {}

const std::shared_ptr<MaximusContext>& GTable::get_context() const {
    return ctx_;
}

std::shared_ptr<Schema> GTable::get_schema() {
    return schema_;
}

std::shared_ptr<cudf::table> GTable::get_table() {
    return cudf_table_;
}

int GTable::get_num_columns() {
    return cudf_table_ ? cudf_table_->num_columns() : 0;
}

int64_t GTable::get_num_rows() {
    return cudf_table_ ? cudf_table_->num_rows() : 0;
}

std::shared_ptr<GTable> GTable::clone() {

    if (!cudf_table_) return nullptr;

    rmm::cuda_stream_view stream = ctx_->get_kernel_stream();
    auto mr                      = ctx_->mr;

    auto cloned_table = std::make_unique<cudf::table>(*cudf_table_, stream, mr);

    return std::make_shared<GTable>(ctx_, schema_, std::move(cloned_table));
}

void GTable::select_columns(const std::vector<std::string>& include_columns) {
    if (!cudf_table_ || !schema_) return;

    // Release ownership of all columns from the current table
    auto released_cols = cudf_table_->release(); // vector<unique_ptr<column>>

    // Determine which column indices to keep
    std::vector<int> column_indices;
    for (const auto &col_name : include_columns) {
        int idx = schema_->column_index(col_name);
        if (idx >= 0 && idx < released_cols.size())
            column_indices.push_back(idx);
    }

    // Move only the selected columns into the new table
    std::vector<std::unique_ptr<cudf::column>> selected_columns;
    selected_columns.reserve(column_indices.size());
    for (int idx : column_indices) {
        selected_columns.push_back(std::move(released_cols[idx]));
        // All other columns in released_cols will be destroyed when released_cols goes out of scope
    }

    // Replace the original table with the new table containing only the selected columns
    cudf_table_ = std::make_unique<cudf::table>(std::move(selected_columns));

    // Update the schema to match the new columns
    schema_ = schema_->subschema(include_columns);
}
}