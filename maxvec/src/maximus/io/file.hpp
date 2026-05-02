#include <arrow/io/api.h>

#include <maximus/context.hpp>

namespace maximus {
Status input_file(const std::shared_ptr<MaximusContext> &ctx,
                  const std::string &path,
                  std::shared_ptr<arrow::io::RandomAccessFile> &input);
}