#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

#define WARP_SIZE 32

__global__ void sgemv(float* A, float* x, float* y, int M, int K){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int laneId = tx % WARP_SIZE;
    int row = blockIdx.x;
    if (row >= M){
        return;
    }

    float sum = 0.0f;
    for(int k=0; k<K; k += WARP_SIZE){
        int col = k + laneId;
        sum += (col < K) ?A[row * K + col] * x[col] : 0.0f;
    }
    for(int offset=WARP_SIZE>>1; offset>0; offset>>=1){
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if(laneId == 0){
        y[row] = sum;
    } 
}