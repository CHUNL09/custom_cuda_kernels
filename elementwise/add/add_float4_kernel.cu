#include <cuda_runtime.h>
#include <iostream>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&value)[0])

__global__ void elementwise_add_float4(const float* a, const float* b, float* c, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_vector = idx * 4; 
    if(idx_vector + 3 < N){
        float4 tmp_a = FLOAT4(a[idx_vector]);
        float4 tmp_b = FLOAT4(b[idx_vector]);
        float4 tmp_c;

        tmp_c.x = tmp_a.x + tmp_b.x;
        tmp_c.y = tmp_a.y + tmp_b.y;
        tmp_c.z = tmp_a.z + tmp_b.z;
        tmp_c.w = tmp_a.w + tmp_b.w;

        FLOAT4(c[idx_vector]) = tmp_c;
    }else{
        for (int i = idx, i < N; i++){
            c[i] = a[i] + b[i];
        }
    }
}


void launch_elementwise_add_float4(const float* a, const float* b, float* c, int N){
    int block_size = 1024;
    int grid_size = (N + block_size - 1) / block_size;
    elementwise_add_float4<<<grid_size, block_size>>>(a, b, c, N);
}