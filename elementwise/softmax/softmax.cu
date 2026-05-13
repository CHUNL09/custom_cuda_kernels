#include <iostream>
#include <cuda_runtime.h>


__global__ void softmax_max_kernel(const float* input, float* max_val, size_t N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;

    __shared__ float smem[blockDim.x];
    smem[tid] = (idx < N)? input[idx]: -INFINITY;
    __syncthreads();

    for(size_t offset = blockDim.x >> 1; offset > 0; offset >>=1){
        if(tid < offset){
            if(smem[tid + offset] > smem[tid]){
                smem[tid] = smem[tid + offset];
                __syncthreads();
            }
        }     
    }
    if(tid == 0){
        atomicMax(max_val, smem[0]);
    }
}

__global__ void softmax_exp_sum_kernel(const float* input, float max_val, float* exp_vals, float* sum, size_t N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;

    __shared__ float smem[blockDim.x];
    
    float exp_val = (idx < N)? expf(input[idx] - max_val): 0.0f;
    exp_vals[idx] = (idx < N)? exp_val: 0.0f;
    
    smem[tid] = exp_val;
    __syncthreads();

    for(size_t offset = blockDim.x >> 1; offset > 0; offset >>=1){
        if(tid <  offset){
            smem[tid] += smem[tid + offset];
            __syncthreads();
        }
    }
    if(tid == 0){
        atomicAdd(sum, smem[0]);
    }
}


__global__ void softmax_normalize_kernel(const float* exp_vals, float* output, float sum, size_t N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        output[idx] = exp_vals[idx] / sum;
    }
}


void softmax_gpu(const float* input, float* output, size_t N){
    int blockSize = 512;
    int gridSize = (N + blockSize - 1)/blockSize;

    float* max_val = nullptr;
    float* sum = nullptr;
    float* exp_vals = nullptr;

    cudaMalloc((void**)&max_val, sizeof(float));
    cudaMalloc((void**)&sum, sizeof(float));
    cudaMalloc((void**)&exp_vals, N * sizeof(float));
    
    float h_max_val = -INFINITY;
    float h_sum = 0.0f;
    cudaMemcpy(max_val, &h_max_val, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice)

    softmax_max_kernel<<<gridSize, blockSize>>>(input, max_val, N);
    cudaDeviceSynchronize();

    softmax_exp_sum_kernel<<<gridSize, blockSize>>>(input, *max_val, exp_vals, sum, N);
    cudaDeviceSynchronize();
    
    softmax_normalize_kernel<<<gridSize, blockSize>>>(exp_vals, output, *sum, N);
    cudaDeviceSynchronize();
    cudaFree(exp_vals);
    cudaFree(max_val);
    cudaFree(sum);
}