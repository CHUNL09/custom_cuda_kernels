#include <cuda_runtime.h>
#include <iostream>

void cudaCheck(cudaError_t err){
    if(err != cudaSuccess){
        std::cerr << "cuda error: "<< cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

__global__ void sum_native_kernel(const float* input, float* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        atomicAdd(output, input[idx]);
    }
}

int main(){
    const size_t N = 1000000;
    float* h_nums = (float*)malloc(N * sizeof(float));
    float* sum = (float*)malloc(sizeof(float));
    float* h_sum = (float*)malloc(sizeof(float));
    *sum = 0.0f;
    for(size_t i=0; i< N; i++){
        h_nums[i] = (float)i;
        *sum += h_nums[i];
    }
    std::cout << "sum: " << *sum << std::endl;

    float* d_sum = nullptr;
    float* d_nums = nullptr;
    float zero = 0.0f;
    cudaCheck(cudaMalloc((void**)&d_sum, sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_nums, N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_nums, h_nums, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_sum, &zero, sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 32 * 32;
    int gridSize = (N + blockSize - 1)/blockSize;
    sum_native_kernel<<<gridSize, blockSize>>>(d_nums, d_sum, N);
    cudaCheck(cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Cuda sum: " << *h_sum << std::endl;

    cudaCheck(cudaFree(d_sum));
    cudaCheck(cudaFree(d_nums));
    free(h_nums);
    free(sum);
    free(h_sum);
    return 0;
}