#include <cuda_runtime.h>
#include <iostream>

#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define CEIL(M, N) (((M) + (N - 1)) / (N))


__global__ void sgemm_native(const float* A, const float* B, float* C, int M, int N, int K){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    if (idy < M && idx < N){
        float sum = 0.0f;
        for(int k=0; k < K; k++){
            // A[M][K] * B[K][N]
            sum += A[OFFSET(idy, k, K)] * B[OFFSET(k, idx, N)];
        }
        C[OFFSET(idy, idx, N)] = sum;
    }
}

template<const int BM, const int BN, const int BK>
__global__ void sgemm_semm(const float* A, const float* B, float* C, int M, int N, int K){
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int by = blockIdx.y;
    int bx = blockIdx.x;


    float sum = 0.0f;
    for(int i=0; i< CEIL(K, BK); i++){
        int A_row = by * BM + ty;
        int A_col = i * BK + tx;
        if(A_row < M && A_col < K){
            As[ty][tx] = A[OFFSET(A_row, A_col, K)];
        }
        else{
            As[ty][tx] = 0.0f;
        }
        
        // B[K][N]
        int B_row = i * BK + ty;
        int B_col = bx * BN + tx;
        if(B_row < K && B_col < N){
            Bs[ty][tx] = B[OFFSET(B_row, B_col, N)];
        }else{
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();

        for(int j=0; j< BK; j++){
            sum += As[ty][j] * Bs[j][tx];
        }
    }
    int C_row = by * BM + ty;
    int C_col = bx * BN + tx;
    C[OFFSET(C_row, C_col, N)] = sum;
}