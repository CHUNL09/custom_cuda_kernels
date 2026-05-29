#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

#define WARP_SIZE 32

void softmax_matrix_cpu(const float* input, float* output, size_t N, size_t M){
    // matrix shape M x N
    for(int i=0; i<M; i++){
        float max_val = *(std::max_element(input + i * N, input + (i+1)* N));
        float row_sum = 0.0f;
        for(int j=0; j<N; j++){
            output[i*N + j] = expf(input[i*N + j] - max_val);
            row_sum += output[i*N + j];
        }
        for(int j=0; j<N; j++){
            output[i*N + j] /= row_sum;
        }
    }
}

void softmax_matrix_col_cpu(const float* input, float* output, size_t N, size_t M){
    // matrix shape M x N
    for(int i=0; i<N; i++){
        float max_val = -INFINITY;
        for(int j=0; j< M; j++){
            max_val = fmaxf(max_val, input[j*N + i]);
        }
        float col_sum = 0.0f;
        for(int j=0; j< M; j++){
            output[j*N + i] = expf(input[j*N + i] - max_val);
            col_sum += output[j*N + i];
        }
        for(int j=0; j< M; j++){
            output[j*N + i] /= col_sum;
        }
    }
}


__global__ void softmax_max_kernel(const float* input, float* max_val, size_t N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    __shared__ float smem[32];
    float max_local = (idx < N)? input[idx]:-INFINITY;
    for(int offset=WARP_SIZE>>1; offset >0; offset>>=1){
        max_local = fmaxf(max_local, __shfl_down_sync(0xFFFFFFFF, max_local, offset));
    }
    if(laneId == 0){
        smem[warpId] = max_local;
    }
    __syncthreads();
    if(warpId == 0){
        int warpNum = blockDim.x /WARP_SIZE;
        max_local = (laneId < warpNum)? smem[laneId]: -INFINITY;
        for(int offset=WARP_SIZE>>1; offset >0; offset>>=1){
            max_local = fmaxf(max_local, __shfl_down_sync(0xFFFFFFFF, max_local, offset));
        }
        if(laneId ==  0) atomicMaxFloat(max_val, max_local);
    }
}


__global__ void softmax_exp_sum_kernel(const float* input, float max_val, float* exp_vals, float* sum, size_t N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    __shared__ float smem[32];
    float sum_local = (idx < N)? expf(input[idx] - max_val): 0.0f;
    if(idx < N){
        exp_vals[idx] = sum_local;
    }

    for(int offset=WARP_SIZE>>1; offset >0; offset>>=1){
        sum_local += __shfl_down_sync(0xFFFFFFFF, sum_local, offset);
    }
    if(laneId == 0) smem[warpId] = sum_local;
    __syncthreads();

    if(warpId == 0){
        int warpNum = blockDim.x /WARP_SIZE;
        sum_local = (laneId < warpNum)? smem[laneId]: 0.0f;
        for(int offset=WARP_SIZE>>1; offset >0; offset>>=1){
            sum_local += __shfl_down_sync(0xFFFFFFFF, sum_local, offset);
        }
        if(laneId == 0) atomicAdd(sum, sum_local);
    }

}

__global__ void softmax_div_kernel(float* sum, float* exp_vals, size_t N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_vec = 4 * idx;
    float inv_sum = 1.0f / *sum;
    if(idx_vec + 3 < N){
        float4 tmp = reinterpret_cast<float4*>(&exp_vals[idx_vec])[0];
        exp_vals[idx_vec] = tmp.x * inv_sum;
        exp_vals[idx_vec + 1] = tmp.y * inv_sum;
        exp_vals[idx_vec + 2] = tmp.z * inv_sum;
        exp_vals[idx_vec + 3] = tmp.w * inv_sum;
    }else{
        for(int i = idx_vec; i< N; i++){
            exp_vals[i] = exp_vals[i] * inv_sum;
        }
    }
}


void softmax_gpu(const float* input, float* output, float* sum_device, float* max_device, size_t N, size_t M){
    // matrix shape M x N
    blockSize = 512;
    gridSize = (M*N + blockSize - 1)/blockSize;

    softmax_max_kernel<<<gridSize, blockSize>>>(input, max_device, M*N);
    cudaDeviceSynchronize();

    softmax_exp_sum_kernel<<<gridSize, blockSize>>>(input, *max_device, output, sum_device, M*N);
    cudaDeviceSynchronize();

    softmax_div_kernel<<<gridSize, blockSize>>>(sum_device, output, M*N);
    cudaDeviceSynchronize();

}
