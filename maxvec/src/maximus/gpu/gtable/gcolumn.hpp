#pragma once

#include <maximus/gpu/gtable/gcontext.hpp>
#include <maximus/types/types.hpp>

namespace maximus {

namespace gpu {

class GColumn {
public:
    GColumn() = default;

    GColumn(int length,
            int64_t null_count,
            std::shared_ptr<DataType> type,
            std::shared_ptr<GBuffer> null,
            std::shared_ptr<GBuffer> buf,
            std::shared_ptr<MaximusGContext> &ctx,
            std::vector<std::shared_ptr<GColumn>> children = {});

    /**
     * To get the device context
     */
    const std::shared_ptr<MaximusGContext> &get_context() const;

    /**
     * To get the data buffer
     */
    std::shared_ptr<GBuffer> &get_data_buffer();

    /**
     * To get the nullbitmap buffer
     */
    std::shared_ptr<GBuffer> &get_null_buffer();

    /**
     * To get the datatype
     */
    std::shared_ptr<DataType> &get_data_type();

    /**
     * To get the null count
     */
    int get_null_count();

    /**
     * To get the length of the column
     */
    int get_length();

    /**
     * To get a pointer to the data buffer
     */
    template<typename T>
    inline T *data() {
        return buf_->data<T>();
    }

    /**
     * To get a pointer to the null bitmap buffer
     */
    template<typename T>
    inline T *null() {
        return null_->data<T>();
    }

    /**
     * To get children of the column
     */
    std::vector<std::shared_ptr<GColumn>> &get_children();

    /**
     * To release access of children of the column
     */
    std::vector<std::shared_ptr<GColumn>> release_children();

    /**
     * To clone a GColumn
     */
    std::shared_ptr<GColumn> clone();

    /**
     * To transfer an array from the CPU to a GColumn on the GPU
     */
    static arrow::Status Make(std::shared_ptr<arrow::ArrayData> host_array,
                              std::shared_ptr<GColumn> &device_column,
                              std::shared_ptr<MaximusGContext> &device_ctx);

    /**
     * To transfer a GColumn from the GPU to an array on the CPU
     */
    static arrow::Status Compose(std::shared_ptr<GColumn> &device_column,
                                 int num_rows,
                                 std::shared_ptr<arrow::ArrayData> &host_array,
                                 arrow::MemoryPool *pool = arrow::default_memory_pool());

    /**
     * To transfer an array from the CPU to a GColumn on the GPU
     */
    static arrow::Status Make(std::shared_ptr<arrow::Array> host_array,
                              std::shared_ptr<GColumn> &device_column,
                              std::shared_ptr<MaximusGContext> &device_ctx);

    /**
     * To transfer a GColumn from the GPU to an array on the CPU
     */
    static arrow::Status Compose(std::shared_ptr<GColumn> &device_column,
                                 int num_rows,
                                 std::shared_ptr<arrow::Array> &host_array,
                                 arrow::MemoryPool *pool = arrow::default_memory_pool());

private:
    int length;
    int64_t null_count{0};
    std::shared_ptr<DataType> type_;
    std::shared_ptr<GBuffer> null_;
    std::shared_ptr<GBuffer> buf_;
    std::shared_ptr<MaximusGContext> ctx_;
    std::vector<std::shared_ptr<GColumn>> children_{};  // for nested types
};

}  // namespace gpu
}  // namespace maximus
