#include <cuda_runtime.h>
#include <iostream>


__global__ void sigmoid_native_kernel(const float* input, float* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        output[idx] = 1.0f/(1.0f + expf(-input[idx]));
    }
}


__global__ void sigmoid_native_float4_kernel(const float* input, float* output, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_vec = idx * 4;
    if(idx_vec + 3 < N){
        const float4 tmp = reinterpret_cast<const float4*>(&input[idx_vec])[0];
        output[idx_vec] = 1.0f/(1.0f + expf(-tmp.x));
        output[idx_vec + 1] = 1.0f/(1.0f + expf(-tmp.y));
        output[idx_vec + 2] = 1.0f/(1.0f + expf(-tmp.z));
        output[idx_vec + 3] = 1.0f/(1.0f + expf(-tmp.w));
    }else{
        for(int i = idx_vec; i < N; i++){
            output[i] = 1.0f/(1.0f + expf(-input[i]));
        }
    }

}


int main(){
    int N = 1024;
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(N * sizeof(float));

    cudaCheck(cudaMalloc(&d_input, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * sizeof(float)));

    cudaCheck(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    sigmoid_native_kernel<<<1, N>>>(d_input, d_output, N);

    cudaCheck(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return 0;
}