#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>


void transpose_matrix_cpu(float* input, float* output, size_t N, size_t M){
    // matrix shape M x N
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            output[j*M + i] = input[i*N + j];
        }
    }

}


__global__ void transpose_native_kernel(const float* input, float* output, size_t N, size_t M){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < M && col < N){
        output[col*M + row] = input[row*N + col];
    }
}


__global__ void transpose_native_coalesced_write_kernel(const float* input, float* output, size_t N, size_t M){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < N && col < M){
        // output[col*M + row] = input[row*N + col];
        output[row*M + col] = input[row*N + col];
    }
}


template <const int TILE_DIM>
__global__ void transpose_coalesced_read_write_kernel(const float* input, float* output, size_t N, size_t M){
    __shared__ float smem[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;
    int in_col = bx + threadIdx.x;
    int in_row = by + threadIdx.y;
    // 合并读取
    if(in_col < N && in_row < M){
        // smem[x1][y1] = input[y1][x1]
        // (tx, ty) 写入 smem[ty][tx]
        smem[threadIdx.y][threadIdx.x] = input[in_row * N + in_col];
    }
    __syncthreads();

    int out_row = bx + threadIdx.y;
    int out_col = by + threadIdx.x;
    if(out_row < N && out_col < M){
        output[out_row * M + out_col] = smem[threadIdx.x][threadIdx.y];
    }
}


template <const int TILE_DIM>
__global__ void transpose_coalesced_read_write_xor_kernel(const float* input, float* output, size_t N, size_t M){
    __shared__ float smem[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x * TILE_DIM; 
    int by = blockIdx.y * TILE_DIM; 
    int in_row = by + threadIdx.y;
    int in_col = bx + threadIdx.x;
    if(in_row < M && in_col < N){
        smem[threadIdx.y][threadIdx.x ^ threadIdx.y] = input[in_row*N + in_col];
    }
    __syncthreads();

    int out_row = bx + threadIdx.y;
    int out_col = by + threadIdx.x;
    if(out_row < N && out_col < M){
        output[out_row * M + out_col] = smem[threadIdx.x][threadIdx.y ^ threadIdx.x];
    }

}