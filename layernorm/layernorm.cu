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
        float norm = (input[batch_id * features + i] - mean) * inv_sqrt_var;
        output[batch_id * features + i] = norm * gamma[i] + beta[i];
    }
}


__global__ void layernorm_v2(float* output, float* input, float* gamma, float* beta, int batch, int features, float eps){
    // Using shared memory and let each thread to handle more features
    int tid = threadIdx.x; 
    int batch_id = blockIdx.x;
    if(batch_id >= batch) return;
    extern __shared__ float smem[];
    float sum = 0.0f;
    for(int i=tid; i<features; i += blockDim.x){
        sum += input[batch_id * features + i];
    }
    smem[tid] = sum;
    __syncthreads();
    for(int offset=blockDim.x >> 1; offset>0; offset>>=1){
        if(tid < offset){
            smem[tid] += smem[tid + offset];
            __syncthreads();
        }
    }
    float inv_features = 1.0f / features;
    float mean = smem[0] * inv_features;

    float var = 0.0f;
    for(int i=tid; i<features; i+=blockDim.x){
        float diff = input[batch_id * features + i] - mean;
        var += diff * diff;
    }
    smem[tid] = var;
    __syncthreads();
    for(int offset=blockDim.x >> 1; offset>0; offset>>=1){
        if(tid < offset){
            smem[tid] += smem[tid + offset];
            __syncthreads();
        }
    }
    var = smem[0] * inv_features;
    float inv_sqrt_var = 1.0f/ sqrtf(var + eps);
    for(int i=tid; i<features; i+=blockDim.x){
        float norm = inv_sqrt_var * (input[batch_id * features + i] - mean);
        output[batch_id * features + i] = norm * gamma[i] + beta[i];
    }
}


__global__ void layernorm_v3(float* output, float* input, float* gamma, float* beta, int batch, int features, float eps){
    // Using shared memory and let each thread to handle more features
    // also using float4 
    int tid = threadIdx.x; 
    int batch_id = blockIdx.x;
    if(batch_id >= batch) return;
    float* x = input + batch_id * features;
    float* out = output + batch_id * features;
    int vec_nums = features / 4;
    extern __shared__ float smem[];
    
    int tail_offset = vec_nums * 4;
    int tail = features - tail_offset;

    float sum = 0.0f;

    for(int i=tid; i<vec_nums; i += blockDim.x){
        float4 vec = reinterpret_cast<float4*>(x)[i];
        sum += vec.x + vec.y + vec.z + vec.w;
    }
    for(int i=tid; i<tail; i+=blockDim.x){
        sum += x[tail_offset + i];
    }
    smem[tid] = sum;
    __syncthreads();
    for(int offset=blockDim.x >> 1; offset>0; offset>>=1){
        if(tid < offset){
            smem[tid] += smem[tid + offset];
            __syncthreads();
        }
    }
    float inv_features = 1.0f / features;
    float mean = smem[0] * inv_features;

    float var = 0.0f;
    for(int i=tid; i<vec_nums; i+=blockDim.x){
        float4 vec = reinterpret_cast<float4*>(x)[i];
        float diff_x = vec.x - mean;
        float diff_y = vec.y - mean;
        float diff_z = vec.z - mean;
        float diff_w = vec.w - mean;
        var += diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + diff_w * diff_w;
    }

    for(int i=tid; i<tail; i+=blockDim.x){
        float diff = x[tail_offset + i] - mean;
        var += diff * diff;
    }
    smem[tid] = var;
    __syncthreads();
    for(int offset=blockDim.x >> 1; offset>0; offset>>=1){
        if(tid < offset){
            smem[tid] += smem[tid + offset];
            __syncthreads();
        }
    }
    var = smem[0] * inv_features;
    float inv_sqrt_var = 1.0f/ sqrtf(var + eps);
    for(int i=tid; i<vec_nums; i+=blockDim.x){
        float4 vec = reinterpret_cast<float4*>(x)[i];
        float4 norm;
        norm.x = inv_sqrt_var * (vec.x - mean);
        norm.y = inv_sqrt_var * (vec.y - mean);
        norm.z = inv_sqrt_var * (vec.z - mean);
        norm.w = inv_sqrt_var * (vec.w - mean);
        float4 gamma_vec = reinterpret_cast<float4*>(gamma)[i];
        float4 beta_vec = reinterpret_cast<float4*>(beta)[i];

        float4 output_vec;
        output_vec.x = norm.x * gamma_vec.x + beta_vec.x;
        output_vec.y = norm.y * gamma_vec.y + beta_vec.y;
        output_vec.z = norm.z * gamma_vec.z + beta_vec.z;
        output_vec.w = norm.w * gamma_vec.w + beta_vec.w;
        reinterpret_cast<float4*>(out)[i] = output_vec;
    }
    
    for(int i=tid; i<tail; i+=blockDim.x){
        int idx = tail_offset + i;
        float val = (x[idx] - mean) * inv_sqrt_var;
        out[idx] = val * gamma[idx] + beta[idx];
    }
}


__global__ void layernorm_v5(float* output, float* input, float* gamma, float* beta, int batch, int features, float eps){
    // Using warp shuffle to avoid shared memory
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int batch_id = blockIdx.x * (blockDim.x / WARP_SIZE) + warpId;

    if(batch_id >= batch) return;

    float* x = input + batch_id * features;
    float* out = output + batch_id * features;

    float sum = 0.0f;
    for(int i=laneId; i<features; i+=WARP_SIZE){
        sum += x[i];
    }
    for(int offset=WARP_SIZE >> 1; offset>0; offset>>=1){
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }
    float inv_features = 1.0f / features;
    float mean = sum * inv_features;
    float var = 0.0f;
    for(int i=laneId; i<features; i+=WARP_SIZE){
        float diff = x[i] - mean;
        var += diff * diff;
    }
    for(int offset=WARP_SIZE>>1; offset>0; offset >>=1){
        var += __shfl_xor_sync(0xFFFFFFFF, var, offset);
    }
    var = inv_features * var;
    float inv_sqrt_var = 1.0f/ sqrtf(var + eps);
    for(int i=laneId; i<features; i+=WARP_SIZE){
        out[i] = (x[i] - mean) * inv_sqrt_var * gamma[i] + beta[i];
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
    float* h_output_gpu_v1 = (float*)malloc(N * sizeof(float));
    float* h_output_gpu_v2 = (float*)malloc(N * sizeof(float));

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
    
    // GPU v1 native timing
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    dim3 grid(batch);
    dim3 block(1);

    cudaEventRecord(start);
    layernorm_baseline<<<grid, block>>>(d_output, d_input, d_gamma, d_beta, batch, features, eps);
    cudaEventRecord(end);
    CUDA_CHECK(cudaEventSynchronize(end));
    
    float gpu_ms_v1 = 0.0f;
    cudaEventElapsedTime(&gpu_ms_v1, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // check output
    CUDA_CHECK(cudaMemcpy(h_output_gpu_v1, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // GPU v2 shared mem + block reduced
    dim3 block(128);
    dim3 grid(batch);
    size_t smem_size = block.x * sizeof(float);
    cudaEventRecord(start);
    layernorm_v2<<<grid, block, smem_size>>>(
        d_output, d_input, d_gamma, d_beta,
        batch, features, eps
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_v2_ms = 0;
    cudaEventElapsedTime(&gpu_v2_ms, start, stop);

    cudaMemcpy(h_output_gpu_v2, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // accurancy check
    float max_err_v1 = 0.0f;
    float max_err_v2 = 0.0f;
    for(int i=0; i<N; i++){
        float e1 = fabsf(h_output_cpu[i] - h_output_gpu_v1[i]);
        float e2 = fabsf(h_output_cpu[i] - h_output_gpu_v2[i]);
        if(e1 > max_err_v1){
            max_err_v1 = e1;
        }
        if(e2 > max_err_v2){
            max_err_v2 = e2;
        }
    }
    // print
    printf("==== LayerNorm Test ====\n");
    printf("batch=%d  features=%d\n", batch, features);
    printf("CPU time : %.3f ms\n", cpu_ms);

    printf("GPU v1 (naive): %.3f ms   max err = %e\n", gpu_v1_ms, max_err_v1);
    printf("GPU v2 (shared mem + block reduced): %.3f ms   max err = %e\n", gpu_v2_ms, max_err_v2);

    bool pass = (max_err_v1 < 1e-5f && max_err_v2 < 1e-5f);
    printf("\nResult: %s\n", pass ? "PASS ✅" : "FAIL ❌");

    // free host
    free(h_input);
    free(h_gamma);
    free(h_beta);
    free(h_output_cpu);
    free(h_output_gpu_v1);
    free(h_output_gpu_v2);
    // free device
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}