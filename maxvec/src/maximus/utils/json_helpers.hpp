#include <arrow/api.h>

namespace maximus {

std::shared_ptr<arrow::RecordBatch> rb_from_json(const std::shared_ptr<arrow::Schema> &,
                                                 std::string_view);

std::shared_ptr<arrow::Table> table_from_json(const std::shared_ptr<arrow::Schema> &,
                                              const std::vector<std::string> &json);
}  // namespace maximus