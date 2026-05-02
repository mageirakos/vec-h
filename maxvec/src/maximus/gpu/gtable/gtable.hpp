#pragma once

#include <memory>
#include <vector>
#include <string>

#include <cudf/table/table.hpp>

#include <maximus/context.hpp>
#include <maximus/types/schema.hpp>

namespace maximus {
namespace gpu {

class GTable {
public:
    GTable() = default;

    GTable(const std::shared_ptr<MaximusContext> &ctx,
           std::shared_ptr<Schema> schema,
           std::unique_ptr<cudf::table>&& table);

    GTable(const std::shared_ptr<MaximusContext> &ctx,
       std::shared_ptr<Schema> schema,
       std::shared_ptr<cudf::table>&& table);

    // Get device context
    const std::shared_ptr<MaximusContext> &get_context() const;

    // Get schema
    std::shared_ptr<Schema> get_schema();

    // Get table
    std::shared_ptr<cudf::table> get_table();

    // Number of columns
    int get_num_columns();

    // Number of rows
    int64_t get_num_rows();

    // Clone table
    std::shared_ptr<GTable> clone();

    // Select columns by name in-place, moving them from the original table
    void select_columns(const std::vector<std::string> &include_columns);

private:
    std::shared_ptr<Schema> schema_;
    std::shared_ptr<cudf::table> cudf_table_;
    const std::shared_ptr<MaximusContext> ctx_;
};

}  // namespace gpu
}  // namespace maximus
