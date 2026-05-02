#include <maximus/gpu/cudf/table_writer.hpp>

namespace maximus {
namespace gpu {

std::shared_ptr<GTable> cudf_to_gtable(const std::shared_ptr<MaximusContext> &ctx,
                                       std::shared_ptr<Schema> schema,
                                       std::shared_ptr<cudf::table> tab) {
    return std::make_shared<GTable>(ctx, schema, std::move(tab));
}

}  // namespace gpu
}  // namespace maximus
