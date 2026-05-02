#pragma once

#include <cstdint>
#include <memory>

namespace maximus {

namespace gpu {

// Abstract class for buffer
class GBuffer {
public:
    GBuffer() = default;

    /**
     * To get pointer to data
     */
    template<typename T>
    T *data() {
        return static_cast<T *>(get_untyped());
    };

    /**
     * To get the size of the buffer
     */
    virtual uint64_t get_size() = 0;

    /**
     * To offset each element of the buffer
     */
    virtual void offset_buffer_by_value(int64_t offset, int offset_length = 4) = 0;

    /**
     * To clone the buffer
     */
    virtual std::shared_ptr<GBuffer> clone() = 0;

protected:
    int64_t sz_;

    virtual void *get_untyped() = 0;
};

/**
 * Class for AMD: Yet to be implemented
 */
// class RocmBuffer : public GBuffer {
//   public:
//   private:
// };

}  // namespace gpu
}  // namespace maximus