#include <iostream>
#include <maximus/gpu/cudf/cudf_types.hpp>
#include <maximus/gpu/gtable/gcolumn.hpp>
#include <maximus/types/types.hpp>

namespace maximus {

namespace gpu {

GColumn::GColumn(int length,
                 int64_t null_count,
                 std::shared_ptr<DataType> type,
                 std::shared_ptr<GBuffer> null,
                 std::shared_ptr<GBuffer> buf,
                 std::shared_ptr<MaximusGContext> &ctx,
                 std::vector<std::shared_ptr<GColumn>> children)
        : length(length)
        , null_count(null_count)
        , type_(std::move(type))
        , null_(std::move(null))
        , buf_(std::move(buf))
        , ctx_(ctx)
        , children_(std::move(children)) {
}

const std::shared_ptr<MaximusGContext> &GColumn::get_context() const {
    return ctx_;
}

std::shared_ptr<GBuffer> &GColumn::get_data_buffer() {
    return buf_;
}

std::shared_ptr<GBuffer> &GColumn::get_null_buffer() {
    return null_;
}

std::shared_ptr<DataType> &GColumn::get_data_type() {
    return type_;
}

int GColumn::get_null_count() {
    return null_count;
}

int GColumn::get_length() {
    return length;
}

std::vector<std::shared_ptr<GColumn>> &GColumn::get_children() {
    return children_;
}

std::vector<std::shared_ptr<GColumn>> GColumn::release_children() {
    return std::move(children_);
}

std::shared_ptr<GColumn> GColumn::clone() {
    std::shared_ptr<GBuffer> cloned_buf_ = nullptr, cloned_null_ = nullptr;
    std::shared_ptr<DataType> cloned_type_ = type_;  // std::make_shared<DataType>(*type_);
    if (buf_ != nullptr) {
        cloned_buf_ = buf_->clone();
    }
    if (null_ != nullptr) {
        cloned_null_ = null_->clone();
    }
    std::vector<std::shared_ptr<GColumn>> cloned_children;
    for (auto &child : children_) {
        cloned_children.push_back(child->clone());
    }
    return std::make_shared<GColumn>(
        length, null_count, cloned_type_, cloned_null_, cloned_buf_, ctx_, cloned_children);
}

arrow::Status GColumn::Make(std::shared_ptr<arrow::Array> host_array,
                            std::shared_ptr<GColumn> &device_column,
                            std::shared_ptr<MaximusGContext> &device_ctx) {
    std::shared_ptr<GBuffer> buf_, null_ = nullptr;
    std::vector<std::shared_ptr<GColumn>> children = {};

    if (host_array->type()->id() == arrow::Type::STRING) {
        std::shared_ptr<arrow::StringArray> string_array =
            std::static_pointer_cast<arrow::StringArray>(host_array);
        assert(string_array != nullptr);
        int64_t total_offset = string_array->offset(), length = string_array->length();

        uint8_t *ptr;
        // move offsets buffer to GPU
        std::shared_ptr<GBuffer> offsets;
        arrow::Status status = device_ctx->memcpy_host_to_device(string_array->value_offsets(),
                                                                 (length + 1) * sizeof(int32_t),
                                                                 total_offset * sizeof(int32_t),
                                                                 offsets);
        if (!status.ok()) return status;

        // Rescale offsets buffer if needed
        int64_t base_offset = string_array->value_offset(0);
        if (base_offset > 0) offsets->offset_buffer_by_value(base_offset);

        children.push_back(std::make_unique<GColumn>(
            // length + 1, 0, std::make_shared<DataType>(Type::INT32), nullptr, offsets, device_ctx));
            length + 1,
            0,
            arrow::int32(),
            nullptr,
            offsets,
            device_ctx));

        // Allocate memory and transfer data buffer to the GPU
        status = device_ctx->memcpy_host_to_device(string_array->value_data(),
                                                   string_array->value_offset(length),
                                                   string_array->value_offset(0),
                                                   buf_);
        if (!status.ok()) return status;

        // Allocate memory and transfer null-bitmap buffer to the GPU
        if (host_array->null_count() > 0) {
            status = device_ctx->memcpy_masks_host_to_device(
                host_array->null_bitmap(), host_array->length(), host_array->offset(), null_);
            if (!status.ok()) return status;
        }
    } else if (host_array->type()->id() == arrow::Type::BOOL) {
        std::shared_ptr<arrow::BooleanArray> bool_array =
            std::static_pointer_cast<arrow::BooleanArray>(host_array);
        // Allocate memory and transfer data buffer to the GPU
        arrow::Status status = device_ctx->memcpy_masks_host_to_device(
            host_array->data()->buffers[1], host_array->length(), host_array->offset(), buf_);
        if (!status.ok()) return status;
        status = device_ctx->transform_mask_to_bools(buf_, host_array->length(), buf_);
        if (!status.ok()) return status;

        // Allocate memory and transfer null-bitmap buffer to the GPU
        if (host_array->null_count() > 0) {
            status = device_ctx->memcpy_masks_host_to_device(
                host_array->null_bitmap(), host_array->length(), host_array->offset(), null_);
            if (!status.ok()) return status;
        }

    } else {
        std::shared_ptr<arrow::DataType> data_type = host_array->type();
        assert(data_type->byte_width() > 0);

        // Allocate memory and transfer data buffer to the GPU
        arrow::Status status =
            device_ctx->memcpy_host_to_device(host_array->data()->buffers[1],
                                              host_array->length() * data_type->byte_width(),
                                              host_array->offset() * data_type->byte_width(),
                                              buf_);

        // Allocate memory and transfer null-bitmap buffer to the GPU
        if (host_array->null_count() > 0) {
            status = device_ctx->memcpy_masks_host_to_device(
                host_array->null_bitmap(), host_array->length(), host_array->offset(), null_);
            if (!status.ok()) return status;
        }
    }

    // Construct GColumn on device
    device_column = std::make_shared<GColumn>(host_array->length(),
                                              host_array->null_count(),
                                              maximus::to_maximus_type(host_array->type()),
                                              // host_array->type()->id(),
                                              null_,
                                              buf_,
                                              device_ctx,
                                              children);
    return arrow::Status::OK();
}

arrow::Status GColumn::Compose(std::shared_ptr<GColumn> &device_column,
                               int num_rows,
                               std::shared_ptr<arrow::Array> &host_array,
                               arrow::MemoryPool *pool) {
    std::vector<std::shared_ptr<arrow::Buffer>> buf(2);
    if (device_column->type_->id() == Type::BOOL) {
        std::shared_ptr<GBuffer> bool_buf;
        arrow::Status status =
            device_column->ctx_->transform_bools_to_mask(device_column->buf_, num_rows, bool_buf);
        if (!status.ok()) return status;

        // Allocate memory on CPU and transfer data buffer
        status = device_column->ctx_->memcpy_device_to_host(bool_buf, (num_rows + 7) / 8, buf[1]);
        if (!status.ok()) return status;
    } else if (device_column->type_->id() == Type::STRING) {
        buf.resize(3);

        // Allocate memory on CPU and transfer data buffer
        arrow::Status status = device_column->ctx_->memcpy_device_to_host(
            device_column->buf_, device_column->buf_->get_size(), buf[2]);
        if (!status.ok()) return status;

        // Allocate memory on CPU and transfer offsets buffer
        std::shared_ptr<GColumn> offsets = std::move(device_column->release_children()[0]);
        status                           = device_column->ctx_->memcpy_device_to_host(
            offsets->buf_, offsets->buf_->get_size(), buf[1]);
        if (!status.ok()) return status;
    } else {
        // Allocate memory on CPU and transfer data buffer
        arrow::Status status = device_column->ctx_->memcpy_device_to_host(
            device_column->buf_, device_column->buf_->get_size(), buf[1]);
        if (!status.ok()) return status;
    }
    if ((int) device_column->null_count > 0) {
        // Allocate memory on CPU and transfer null-bitmap buffer
        arrow::Status status = device_column->ctx_->memcpy_device_to_host(
            device_column->null_, device_column->null_->get_size(), buf[0]);
        if (!status.ok()) return status;
    }

    // Construct array on CPU
    std::shared_ptr<arrow::ArrayData> host_array_data =
        std::make_shared<arrow::ArrayData>(std::move(maximus::to_arrow_type(device_column->type_)),
                                           num_rows,
                                           std::move(buf),
                                           device_column->null_count);
    host_array = arrow::MakeArray(host_array_data);
    return arrow::Status::OK();
}

}  // namespace gpu
}  // namespace maximus
