#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

#define WARP_SIZE 32



__global__ void softmax_row_kernel(float* input, float* output, int M, int N){
    /*
    一个block负责一行，并且一个block内使用一个warp来计算
    */

    __shared__ float s_max_val;
    __shared__ float s_sum;
    int tid = threadIdx.x;
    int laneId = tid % WARP_SIZE;
    int row = blockIdx.x;
    if(row >= M){
        return;
    }
    int iteration = (N + WARP_SIZE - 1) / WARP_SIZE;

    float max_val = -INFINITY;
    for(int i=0; i<iteration; i++){
        int col = i * WARP_SIZE + laneId;
        max_val = (col < N)? fmaxf(max_val, input[row*N + col]) : max_val;
    }
    for(int offset=WARP_SIZE>>1; offset >0; offset>>=1){
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    if(laneId == 0) s_max_val = max_val;

    float sum = 0.0f;
    for(int i=0; i<iteration; i++){
        int col = i * WARP_SIZE + laneId;
        sum += expf(input[row*N + col] - max_val);
    }
    if(laneId == 0) s_sum = sum;

    float inv_sum = 1.0f / s_sum;
    for(int i=0; i<iteration; i++){
        int col = i * WARP_SIZE + laneId;
        output[row*N + col] = expf(input[row*N + col] - max_val) * inv_sum;
    }

}



__global__ void softmax_row_kernel(float* input, float* output, int M, int N){
    /*
    如果行数大于block size, 可以每个block负责多行
    并且一个block内使用一个warp来计算
    */

    __shared__ float s_max_val;
    __shared__ float s_sum;
    int tid = threadIdx.x;
    int laneId = tid % WARP_SIZE;

    int iteration = (N + WARP_SIZE - 1) / WARP_SIZE;

    for(int row=blockIdx.x; row < M; row += gridDim.x){

        float max_val = -INFINITY;
        for(int i=0; i<iteration; i++){
            int col = i * WARP_SIZE + laneId;
            max_val = (col < N)? fmaxf(max_val, input[row*N + col]) : max_val;
        }
        for(int offset=WARP_SIZE>>1; offset >0; offset>>=1){
            max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
        }
        if(laneId == 0) s_max_val = max_val;

        float sum = 0.0f;
        for(int i=0; i<iteration; i++){
            int col = i * WARP_SIZE + laneId;
            sum += expf(input[row*N + col] - max_val);
        }
        if(laneId == 0) s_sum = sum;

        float inv_sum = 1.0f / s_sum;
        for(int i=0; i<iteration; i++){
            int col = i * WARP_SIZE + laneId;
            output[row*N + col] = expf(input[row*N + col] - max_val) * inv_sum;
        }
    }
}