#pragma once

#include <maximus/context.hpp>
#include <maximus/error_handling.hpp>
#include <maximus/types/schema.hpp>
#include <maximus/types/table_batch.hpp>
#include <string>
#include <vector>

namespace maximus {

// forward declaration
class Table;
using TablePtr = std::shared_ptr<Table>;

// internally, we use arrow::Table
using InternalTable    = arrow::Table;
using InternalTablePtr = std::shared_ptr<InternalTable>;

// A wrapper around the arrow::Table class that provides additional functionality
class Table : public std::enable_shared_from_this<Table> {
public:
    // creates an empty table
    Table() = default;

    // creates a table by wrapping the given internal table (0-copy)
    Table(const std::shared_ptr<MaximusContext> &ctx, InternalTablePtr tab);

    // creates a table by wrapping the given arrow table (0-copy)
    static Status from_table(const std::shared_ptr<MaximusContext> &ctx,
                             std::shared_ptr<arrow::Table> table,
                             TablePtr &table_out);

    // creates a table by wrapping the given table batches (0-copy)
    static Status from_table_batches(const std::shared_ptr<MaximusContext> &ctx,
                                     std::vector<TableBatchPtr> table_batches,
                                     TablePtr &table_out);

    // creates a table from a json string (full copy)
    // An example of a json string is:
    /*
        R"([
            ["x", 1, 1],
            ["y", 0, 3],
            ["z", 1, 5]
          ])"
    */
    static Status from_json(const std::shared_ptr<MaximusContext> &ctx,
                            const std::shared_ptr<Schema> &schema,
                            const std::vector<std::string> &json,
                            TablePtr &table_out);

    // creates a table from a csv file (full copy)
    // The include_columns parameter is used to specify which columns to include in the table
    static Status from_csv(const std::shared_ptr<MaximusContext> &ctx,
                           const std::string &path,
                           const std::shared_ptr<Schema> &schema,
                           const std::vector<std::string> &include_columns,
                           TablePtr &table_out);

    // creates a table from a parquet file (full copy)
    // The include_columns parameter is used to specify which columns to include in the table
    static Status from_parquet(const std::shared_ptr<MaximusContext> &ctx,
                               const std::string &path,
                               const std::shared_ptr<Schema> &schema,
                               const std::vector<std::string> &include_columns,
                               TablePtr &table_out);

    // converts a subset of the table to a string, using the given delimiter
    std::string to_string(int64_t first_row,
                          int64_t end_row,
                          int first_col,
                          int end_col,
                          const std::string delim = ", ") const;

    // converts the whole table to a string, using the default column delimiter
    std::string to_string() const;

    // converts the whole table to a string by using the arrow's ToString method
    std::string to_arrow_string() const;

    // prints the whole table to std::cout using the to_string() method
    void print() const;

    // retrieves the number of rows in a table
    int64_t num_rows() const;

    // retrieves the number of columns in a table
    int32_t num_columns() const;

    // checks whether the table is empty, i.e. if nullptr or 0 rows
    bool empty() const;

    // retrieves the context used by the table
    const std::shared_ptr<maximus::MaximusContext> &get_context() const;

    // retrieves the column names of the table
    std::vector<std::string> column_names();

    // adds a column to the table at the given position and with a given name
    Status add_column(const int32_t position,
                      const std::string &column_name,
                      std::shared_ptr<arrow::Array> input_column);

    // adds a column with a constant value at given position and a given name
    Status add_const_column(const int32_t position,
                            const std::string &column_name,
                            const int32_t value);

    // removes the column at the given position
    Status remove_column(const int32_t position);

    // deallocates the table and releases the memory
    void clear();

    // retrieves the schema of the table
    std::shared_ptr<Schema> get_schema() const;

    // implements the equality operator
    bool operator==(const Table &other) const;

    // retrieves the internal table pointers
    InternalTablePtr get_table() const;
    InternalTablePtr get_table();

    // slices the rows [offset, offset + length) of the table (0-copy)
    TablePtr slice(int64_t offset, int64_t length) const;

    // clones the table fully copying the data.
    // If prefer_pinned_pool is true, the table will be stored in pinned memory
    // If to_single_chunk is true, the table will be stored as a single batch (chunk)
    TablePtr clone(bool prefer_pinned_pool, bool to_single_chunk) const;

    // select a subset of columns from a table and return a new table (0-copy)
    TablePtr select_columns(const std::vector<std::string> &column_names) const;

private:
    std::shared_ptr<maximus::MaximusContext> ctx_;
    InternalTablePtr table_ = nullptr;
};

}  // namespace maximus
