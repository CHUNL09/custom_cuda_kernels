#include <cuda_runtime.h>
#include <iostream>

#define WARP_SIZE 32


void CUDA_CHECK(cudaError_t err){
    if(err != cudaSuccess){
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

void layernorm_cpu(float* out, const float* inp,
                    const float* gamma, const float* beta,
                    int batch, int features, float eps) {
    for(int b=0; b < batch; b++){
        const float* x   = inp  + b * features;
        float*       y   = out  + b * features;

        float mean = 0.0f;
        for (int i = 0; i < features; i++)
            mean += x[i];
        mean /= features;

        float var = 0.0f;
        for (int i = 0; i < features; i++) {
            float d = x[i] - mean;
            var += d * d;
        }
        var /= features;

        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < features; i++) {
            y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

__global__ void layernorm_baseline(float* output, float* input, float* gamma, float* beta, int batch, int features, float eps){
    // one block handle one batch
    int batch_id = blockIdx.x;
    if(batch_id >= batch){
        return;
    }

    float inv_features = 1.0f / features;
    // calculate mean
    float mean = 0.0f;
    for(int i=0; i<features; i++){
        mean += input[batch_id * features + i];
    }
    mean = mean * inv_features;

    // calculate var
    float var = 0.0f;
    for(int i=0; i< features; i++){
        float diff = input[batch_id * features + i] - mean;
        var += diff * diff;
    }
    var = var * inv_features;

    // normalize
    float inv_sqrt_var = 1.0f / sqrtf(var + eps);
    for(int i=0; i< features; i++){
        output[batch_id * features + i] = (input[batch_id * features + i] - mean) * inv_sqrt_var;
    }
}


int main(){
    int batch = 64; 
    int features = 128;
    float eps = 1e-5f;
    size_t N = batch * features;

    // host alloc
    float* h_input = (float*)malloc(N * sizeof(float));
    float* h_gamma = (float*)malloc(features * sizeof(float));
    float* h_beta = (float*)malloc(features * sizeof(float));
    float* h_output_cpu = (float*)malloc(N * sizeof(float));
    float* h_output_gpu = (float*)malloc(N * sizeof(float));

    srand(42);
    for(int i=0; i<N; i++){
        h_input[i] = (float)(rand() % 100)/100.0f - 0.5f;
    }
    for(int i=0; i<features; i++){
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }

    // device alloc
    float* d_input = nullptr;
    float* d_gamma = nullptr;
    float* d_beta = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, features * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta, features * sizeof(float), cudaMemcpyHostToDevice));


    // CPU timing
    auto t0 = std::chrono::high_resolution_clock::now();
    layernorm_cpu(h_output_cpu, h_input, h_gamma, h_beta, batch, features, eps);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // GPU timing
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    dim3 grid(batch);
    dim3 block(1);

    cudaEventRecord(start);
    layernorm_baseline<<<grid, block>>>(d_output, d_input, d_gamma, d_beta, batch, features, eps);
    cudaEventRecord(end);
    CUDA_CHECK(cudaEventSynchronize(end));
    
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // check output
    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for(int i=0; i<N; i++){
        float e = fabsf(h_output_cpu[i] - h_output_gpu[i]);
        if(e > max_err){
            max_err = e;
        }
    }
    // print
    printf("==== LayerNorm Test ====\n");
    printf("batch=%d  features=%d\n", batch, features);
    printf("CPU time : %.3f ms\n", cpu_ms);
    printf("GPU time : %.3f ms\n", gpu_ms);
    printf("Max Abs Error : %e\n", max_err);
    printf("Result : %s\n", max_err < 1e-5f ? "PASS ✅" : "FAIL ❌");

    // free host
    free(h_input);
    free(h_gamma);
    free(h_beta);
    free(h_output_cpu);
    free(h_output_gpu);
    // free device
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}