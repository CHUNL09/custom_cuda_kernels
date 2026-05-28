#include <cuda_runtime.h>
#include <iostream>

#define WARP_SIZE 32

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


__global__ void reduce_sum_smem_tree(const float* input, float* output, int N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    extern __shared__ float smem[];
    if (idx < N){
        smem[tid] = input[idx];
    }else{
        smem[tid] = 0.0f;
    }
    __syncthreads();
    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1){
        if(tid < offset){
            smem[tid] += smem[tid + offset];
        }
        __syncthreads();
    }
    if(tid == 0){
        atomicAdd(output, smem[0]);
    }
}


__global__ void reduce_sum_smem_tree_v2(const float* input, float* output, int N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    extern __shared__ float smem[];

    float sum = 0.0f;
    if(idx * 4 + 3 < N){
        const float4 tmp = reinterpret_cast<const float4*>(&input[idx * 4])[0];
        sum = tmp.x + tmp.y + tmp.z + tmp.w;
    }else{
        for(int i = idx * 4; i< N; i++){
            sum += input[i];
        }
    }

    smem[tid] = sum;
    __syncthreads();
    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1){
        if(tid < offset){
            smem[tid] += smem[tid + offset];
        }
        __syncthreads();
    }
    if(tid == 0){
        atomicAdd(output, smem[0]);
    }
}


__global__ void sum_warp_shf_kernel(const float* input, float* output, int N){
    __shared__ float s_y[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    float val = (idx < N) ? input[idx]: 0.0f;
    for(int offset=WARP_SIZE >> 1; offset > 0; offset >>=1){
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if(laneId == 0) s_y[warpId] = val;
    __syncthreads();
    if(warpId == 0){
        int warpNum = blockDim.x / WARP_SIZE;
        val = (laneId < warpNum) ? s_y[laneId]: 0.0f;
        for(int offset=WARP_SIZE >> 1; offset > 0; offset >>=1){
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if(laneId == 0) atomicAdd(output, val);
    }

}


__global__ void sum_warp_shf_float4_kernel(const float* input, float* output, int N){
    __shared__ float s_y[32];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    int idx_vec = idx * 4;
    float val = 0.0f;
    if(idx_vec + 3 < N){
        const float4 tmp = reinterpret_cast<const flaot4*>(&input[idx_vec])[0];
        val = tmp.x + tmp.y + tmp.z + tmp.w;
    }else{
        for(int i = idx_vec; i< N; i++){
            val += input[i];
        }
    }
    for(int offset=WARP_SIZE >> 1; offset > 0; offset >>=1){
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    if(laneId == 0) s_y[warpId] = val;
    __syncthreads();
    if(warpId == 0){
        int warpNum = blockDim.x / WARP_SIZE;
        val = (laneId < warpNum) ? s_y[laneId]: 0.0f;
        for(int offset=WARP_SIZE >> 1; offset > 0; offset >>=1){
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if(laneId == 0) atomicAdd(output, val);
    }
}


int main(){
    const size_t N = 1000000;
    float* h_nums = (float*)malloc(N * sizeof(float));
    double sum = 0.0;
    float* h_sum = (float*)malloc(sizeof(float));
    for(size_t i=0; i< N; i++){
        h_nums[i] = (float)i;
        sum += h_nums[i];
    }
    std::cout << "sum: " << sum << std::endl;

    float* d_sum = nullptr;
    float* d_nums = nullptr;
    float zero = 0.0f;
    cudaCheck(cudaMalloc((void**)&d_sum, sizeof(float)));
    cudaCheck(cudaMalloc((void**)&d_nums, N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_nums, h_nums, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_sum, &zero, sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 32 * 32;
    int gridSize = (N + blockSize - 1)/blockSize;
    int smem_size = blockSize * sizeof(float);
    reduce_sum_smem_tree<<<gridSize, blockSize, smem_size>>>(d_nums, d_sum, N);
    cudaCheck(cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    /*
    int elements_per_block = blockSize * elements_per_thread;  // 4096 元素/block
    int gridSize = (N + elements_per_block - 1) / elements_per_block;
    reduce_sum_smem_tree_v2<<<gridSize, blockSize, smem_size>>>(d_nums, d_sum, N);
    cudaCheck(cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    */
    std::cout << "Cuda sum: " << *h_sum << std::endl;

    cudaCheck(cudaFree(d_sum));
    cudaCheck(cudaFree(d_nums));
    free(h_nums);
    free(h_sum);
    return 0;
}