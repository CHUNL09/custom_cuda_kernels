#include <cuda_runtime.h>
#include <iostream>

void cudaCheck(cudaError_t err, const char* file, int line){
    if(err != cudaSuccess){
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
        exit(1);
    }
}

__global__ void elementwise_add_native(const float* a, const float* b, float* c, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}


void launch_elementwise_add_native(const float* a, const float* b, float* c, int N){
    int block_size = 1024;
    int grid_size = (N + block_size - 1) / block_size;
    elementwise_add_native<<<grid_size, block_size>>>(a, b, c, N);
}
