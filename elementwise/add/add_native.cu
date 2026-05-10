#include <cuda_runtime.h>
#include <iostream>

void cudaCheck(cudaError_t err, const char* file, int line){
    if(err != cudaSuccess){
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
        exit(1);
    }
}

__global__ void elementwise_add_native(const float* a, const float* b, float* c, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}


int main() {
    constexpr int N = 1024;
    float* a_h = (float*)malloc(sizeof(float) * N);
    float* b_h = (float*)malloc(sizeof(float) * N);
    float* c_h = (float*)malloc(sizeof(float) * N);
    float* c_h_verify = (float*)malloc(sizeof(float) * N);

    for(int i=0; i < N; i++){
        a_h[i] = i;
        b_h[i] = N - i -1;
        c_h_verify[i] = a_h[i] + b_h[i];
    }

    float* a_d = nullptr;
    float* b_d = nullptr;
    float* c_d = nullptr;

    cudaCheck(cudaMalloc((void**)&a_d, sizeof(float) * N), __FILE__, __LINE__);
    cudaCheck(cudaMalloc((void**)&b_d, sizeof(float) * N), __FILE__, __LINE__);
    cudaCheck(cudaMalloc((void**)&c_d, sizeof(float) * N), __FILE__, __LINE__);

    cudaCheck(cudaMemcpy(a_d, a_h, sizeof(float) * N, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    cudaCheck(cudaMemcpy(b_d, b_h, sizeof(float) * N, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    int block_size = 1024;
    int grid_size = (N + block_size - 1) / block_size;

    elementwise_add_native<<<grid_size, block_size>>>(a_d, b_d, c_d, N);

    cudaCheck(cudaMemcpy(c_h, c_d, sizeof(float) * N, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    
    std::cout<<"a_h: "<<std::endl;
    for(int i=0; i < N; i++){
        std::cout << a_h[i] << " " ;
    }
    std::cout<<std::endl;

    std::cout<<"b_h: "<<std::endl;
    for(int i=0; i < N; i++){
        std::cout << b_h[i] << " " ;
    }
    std::cout<<std::endl;

    std::cout<<"c_h: "<<std::endl;
    for(int i=0; i < N; i++){
        std::cout << c_h[i] << " " ;
    }
    std::cout<<std::endl;

    bool verify = true;
    for(int i=0; i < N; i++){
        if(c_h[i] != c_h_verify[i]){
            verify = false;
            break;
        }
    }
    std::cout<<"verify result: "<<(verify? "pass": "fail")<<std::endl;

    free(a_h);
    free(b_h);
    free(c_h);
    free(c_h_verify);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return 0;
}
