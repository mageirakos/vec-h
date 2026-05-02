#include <stdio.h>
#include <iostream>

#include <maximus/utils/cuda_helpers.hpp>

void maximus::cuda_check(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
        std::abort();
    }
}

void maximus::sync_stream(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

int maximus::get_gpu_device() {
    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}