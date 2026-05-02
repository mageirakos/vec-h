#include <cstddef>
#include <cstdint>
#include <iostream>

// FNV-1a 64-bit hash implementation
uint64_t hash_memory_region(const void* ptr, size_t size) {
    const uint8_t* data = static_cast<const uint8_t*>(ptr);
    uint64_t hash       = 14695981039346656037ull;  // FNV offset basis

    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ull;  // FNV prime
    }

    return hash;
}

std::string float_array_to_string(const float* array, size_t size) {
    std::string result;
    for (size_t i = 0; i < size; ++i) {
        result += std::to_string(array[i]);
        if (i < size - 1) {
            result += ", ";
        }
    }
    return result;
}