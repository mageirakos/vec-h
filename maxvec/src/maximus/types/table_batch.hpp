#pragma once

#include <maximus/context.hpp>
#include <maximus/error_handling.hpp>
#include <maximus/types/schema.hpp>
#include <string>
#include <vector>

namespace maximus {

// forward declaration
class TableBatch;
using TableBatchPtr = std::shared_ptr<TableBatch>;

// internally, we use arrow::RecordBatch
using InternalTableBatch    = arrow::RecordBatch;
using InternalTableBatchPtr = std::shared_ptr<InternalTableBatch>;

// A wrapper around the arrow::RecordBatch class that provides additional functionality
class TableBatch : public std::enable_shared_from_this<TableBatch> {
public:
    // creates an empty table batch
    TableBatch() = default;

    // creates a table batch by wrapping the given internal table batch (0-copy)
    TableBatch(const std::shared_ptr<MaximusContext> &ctx, InternalTableBatchPtr tab);

    // creates a table batch by wrapping the given arrow::RecordBatch (0-copy)
    static Status from_record_batch(const std::shared_ptr<MaximusContext> &ctx,
                                    std::shared_ptr<arrow::RecordBatch> table,
                                    TableBatchPtr &table_out);

    // creates a table batch by wrapping the given arrow::compute::ExecBatch
    // This is not always 0-copy, since the ExecBatch sometimes compresses the data using the run-length encoding
    // (e.g. if a column has a constant value)
    // Moreover, ExecBatch might have a boolean mask that filters out some rows.
    static Status from_exec_batch(const std::shared_ptr<MaximusContext> &ctx,
                                  std::shared_ptr<Schema> schema,
                                  std::shared_ptr<::arrow::compute::ExecBatch> batch,
                                  ::arrow::MemoryPool *pool,
                                  TableBatchPtr &table_out);

    // creates a table batch from a json string (full copy)
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
                            std::string_view json,
                            TableBatchPtr &table_out);

    // retrieves the internal arrow::RecordBatch (0-copy)
    Status to_record_batch(std::shared_ptr<arrow::RecordBatch> &output);

    // converts the internal arrow::RecordBatch to an arrow::compute::ExecBatch and returns it
    Status to_exec_batch(std::shared_ptr<arrow::compute::ExecBatch> &output);

    // converts a subset of the table batch to a string, using the given delimiter
    std::string to_string(int64_t first_row,
                          int64_t end_row,
                          int first_col,
                          int end_col,
                          const std::string delim = ", ") const;

    // converts the whole table batch to a string, using the default column delimiter
    std::string to_string() const;

    // converts the whole table batch to a string by using the arrow's ToString method
    std::string to_arrow_string() const;

    // prints the whole table batch to std::cout using the to_string() method
    void print() const;

    // retrieves the number of rows in a table batch
    int64_t num_rows() const;

    // retrieves the number of columns in a table batch
    int32_t num_columns() const;

    // checks whether the table batch is empty, i.e. if nullptr or 0 rows
    bool empty() const;

    // retrieves the context used by the table batch
    const std::shared_ptr<maximus::MaximusContext> &get_context() const;

    // retrieves the column names of the table batch
    std::vector<std::string> column_names();

    // adds a column to the table batch at the given position and with a given name
    Status add_column(int32_t position,
                      const std::string &column_name,
                      std::shared_ptr<arrow::Array> input_column);

    // deallocates the table batch and releases the memory
    void clear();

    // retrieves the schema of the table batch
    std::shared_ptr<Schema> get_schema() const;

    // implements the equals operator
    bool operator==(const TableBatch &other) const;

    // retrieves the internal table batch pointers
    InternalTableBatchPtr get_table_batch() const;
    InternalTableBatchPtr get_table_batch();

    // slices the rows [offset, offset + length) of the table batch (0-copy)
    TableBatchPtr slice(int64_t offset, int64_t length) const;

    // clones the table batch fully copying the data.
    // if prefer_pinned_pool is true, the table batch will be stored in pinned memory
    TableBatchPtr clone(bool prefer_pinned_pool = false) const;

private:
    std::shared_ptr<maximus::MaximusContext> ctx_;
    InternalTableBatchPtr table_ = nullptr;
};

}  // namespace maximus
