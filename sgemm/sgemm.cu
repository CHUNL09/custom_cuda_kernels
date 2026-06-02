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


template<const int BM, const int BN, const int BK>
__global__ void sgemm_semm_v2(const float* A, const float* B, float* C, int M, int N, int K){
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    A = &A[(by * BM) * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];
    float sum = 0.0f;
    for(int k=0; k< K; k += BK){

        As[ty][tx] = A[ty * K + tx];

        Bs[ty][tx] = B[ty * N + tx];

        __syncthreads();
        A += BK;
        B += BK * N;

        for(int j=0; j< BK; j++){
            sum += As[ty][j] * Bs[j][tx];
        }
    }
    C[ty * N + tx] = sum;
}


template<const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm_thread_tile(const float* A, const float* B, float* C, int M, int N, int K){
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    int thread_num = block_row_thread * block_col_thread;

    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;
    
    //A[BM][BK]
    A = &A[by * BM * K];
    //B[BK][BN]
    B = &B[bx * BN];
    //C[BM][BN]
    C = &C[by * BM * N + bx * BN];

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = threadIdx.x / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = threadIdx.x / BN;

    float accum[TM][TN] = {0.0f};
    for(int k=0; k< K; k+= BK){
        for(int i=0; i< BM; i += a_tile_stride){
            As[a_tile_row + i][a_tile_col] = A[(a_tile_row + i)*K + a_tile_col];
        }
        for(int i=0; i< BK; i += b_tile_stride){
            Bs[b_tile_row + i][b_tile_col] = B[(b_tile_row + i)*N + b_tile_col];
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        for(int row=0; row < TM; row++){
            for(int col=0; col < TN; col++){
                for(int i=0; i< BK; i++){
                    accum[row][col] += As[ty+row][i] * Bs[i][tx+ col];
                }
            }
        }
        __syncthreads();
    }
    for(int row=0; row < TM; row++){
        for(int col=0; col < TN; col++){
            C[(ty + row) * N + (tx + col)] = accum[row][col];
        }
    }   
}


