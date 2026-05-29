#include <cuda_runtime.h>
#include <iostream>

#define WARP_SIZE 32



void max_cpu(const float* input, float* max_val, size_t N){
    for(int i=0; i<N; i++){
        if(input[i] > *max_val){
            *max_val = input[i];
        }
    }
}


__device__ float atomicMaxFloat(float* addr, float value){
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int;
    int assumed;

    do{
        assumed = old;
        float max_val = fmaxf(value, __int_as_float(assumed));
        old = atomicCAS(addr_as_int, assumed, __float_as_int(max_val));
    }while(old != assumed);
    return __int_as_float(old);
}


__global__ void max_native_kernel(const float* input, float* max_val, size_t N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N){
        return;
    }
    atomicMaxFloat(max_val, input[idx]);
}


__global__ void max_reduce_kernel(const float* input, float* max_val, size_t N){
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    // based on block size
    extern __shared__ float smem[];
    smem[tid] = (idx < N)? input[idx]:-INFINITY;
    __syncthreads();
    for(int offset=blockDim.x >> 1; offset>0; offset >>=1){
        if(tid < offset){
            smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
        }
        __syncthreads();
    }
    if(tid==0){
        atomicMaxFloat(max_val, smem[0]);
    }
}


__global__ void max_reduce_warp_shf_kernel(const float* input, float* max_val, size_t N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    __shared__ float s_y[WARP_SIZE];
    float val = (idx < N)? input[idx]:-INFINITY;
    for(int offset=WARP_SIZE>>1; offset>0; offset>>=1){
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    if(laneId == 0) s_y[warpId] = val;
    __syncthreads();

    if(warpId == 0){
        int warpNum = blockDim.x / WARP_SIZE;
        val = (laneId < warpNum)? s_y[laneId]: -INFINITY;
        for(int offset=WARP_SIZE>>1; offset>0; offset>>=1){
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        if(laneId == 0) atomicMaxFloat(max_val, val);
    }
}


__global__ void max_reduce_warp_shf_float4_kernel(const float* input, float* max_val, size_t N){
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    int idx_vec = idx * 4;

    __shared__ float s_y[WARP_SIZE];
    float val = -INFINITY;
    if(idx_vec + 3 < N){
        const float4 tmp = reinterpret_cast<const float4*>(&input[idx_vec])[0];
        val = fmaxf(tmp.x, fmaxf(tmp.y, fmaxf(tmp.z, tmp.w)));
           }else{
        for(int i=idx_vec; i<N; i++){
            val = fmaxf(val, input[i]);
        }
    }
    for(int offset=WARP_SIZE>>1; offset>0; offset>>=1){
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    if(laneId == 0) s_y[warpId] = val;
    __syncthreads();
    if(warpId == 0){
        int warpNum = blockDim.x / WARP_SIZE;
        val = (laneId < warpNum)? s_y[laneId]: -INFINITY;
        for(int offset=WARP_SIZE>>1; offset>0; offset>>=1){
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        if(laneId == 0) atomicMaxFloat(max_val, val);
    }
}