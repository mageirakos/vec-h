#pragma once

#include <cuda_runtime_api.h>

namespace maximus {
// simple CUDA error check
void cuda_check(cudaError_t e, const char* file, int line);

#define CUDA_CHECK(x) cuda_check((x), __FILE__, __LINE__)

// a wrapper around device to device cuda memcpy async
template <typename T>
void copy_d2d_async(const T* from, T* to, std::size_t n, cudaStream_t stream=nullptr) {
    auto status = cudaMemcpyAsync(to, from, n * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    CUDA_CHECK(status);
}

// a wrapper around host to device cuda memcpy async
template <typename T>
void copy_h2d_async(const T* from, T* to, std::size_t n, cudaStream_t stream=nullptr) {
    auto status = cudaMemcpyAsync(to, from, n * sizeof(T), cudaMemcpyHostToDevice, stream);
    CUDA_CHECK(status);
}

template <typename T>
void copy_d2h_async(const T* from, T* to, std::size_t n, cudaStream_t stream=nullptr) {
    auto status = cudaMemcpyAsync(to, from, n * sizeof(T), cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK(status);
}

void sync_stream(cudaStream_t stream=nullptr);

int get_gpu_device();
}
