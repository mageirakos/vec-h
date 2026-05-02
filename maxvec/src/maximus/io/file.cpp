#include <arrow/io/api.h>

#include <maximus/io/csv.hpp>
#include <maximus/utils/utils.hpp>

namespace maximus {
Status input_file(const std::shared_ptr<MaximusContext> &ctx,
                  const std::string &path,
                  std::shared_ptr<arrow::io::RandomAccessFile> &input) {
    assert(ctx);
    auto pool = ctx->get_memory_pool();
    assert(pool);
    auto io_context = ctx->get_io_context();
    assert(io_context);

    auto maybe_input = arrow::io::ReadableFile::Open(path, pool);
    if (!maybe_input.ok()) {
        return Status(ErrorCode::ArrowError, maybe_input.status().message());
    }
    input = maybe_input.ValueOrDie();
    return Status::OK();
}
}  // namespace maximus
