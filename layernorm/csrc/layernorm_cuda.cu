#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32


template<typename scalar_t>
__global__ void layernorm_v5(scalar_t* output, scalar_t* input, scalar_t* gamma, scalar_t* beta, int batch, int features, float eps){
    // Using warp shuffle to avoid shared memory
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int batch_id = blockIdx.x * (blockDim.x / WARP_SIZE) + warpId;

    if(batch_id >= batch) return;

    scalar_t* x = input + batch_id * features;
    scalar_t* out = output + batch_id * features;

    float sum = 0.0f;
    for(int i=laneId; i<features; i+=WARP_SIZE){
        sum += static_cast<float>(x[i]);
    }
    for(int offset=WARP_SIZE >> 1; offset>0; offset>>=1){
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }
    float inv_features = 1.0f / features;
    float mean = sum * inv_features;
    float var = 0.0f;
    for(int i=laneId; i<features; i+=WARP_SIZE){
        float diff = static_cast<float>(x[i]) - mean;
        var += diff * diff;
    }
    for(int offset=WARP_SIZE>>1; offset>0; offset >>=1){
        var += __shfl_xor_sync(0xFFFFFFFF, var, offset);
    }
    var = inv_features * var;
    float inv_sqrt_var = 1.0f/ sqrtf(var + eps);
    for(int i=laneId; i<features; i+=WARP_SIZE){
        out[i] = static_cast<scalar_t>((static_cast<float>(x[i]) - mean) * inv_sqrt_var * static_cast<float>(gamma[i]) + static_cast<float>(beta[i])));
    }
}


torch::Tensor layer_norm_cuda(
    torch::Tensor &input,
    torch::Tensor &gamma,
    torch::Tensor &beta,
    double eps
){
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be on CUDA");
    TORCH_CHECK(beta.is_cuda(), "beta must be on CUDA");
    TORCH_CHECK(eps > 0, "eps must be positive");

    auto output = torch::empty_like(input);

    int batch = input.size(0);
    int features = input.size(1);

    int block_size = 256;
    int warps_per_block = block_size / WARP_SIZE;
    int grid_size = (batch + warps_per_block - 1)/warps_per_block;
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(),
        "layer_norm_cuda",
        ([&]{
            layernorm_v5<scalar_t><<<grid_size, block_size>>>(
                output.data_ptr<scalar_t>(), 
                input.data_ptr<scalar_t>(), 
                gamma.data_ptr<scalar_t>(), 
                beta.data_ptr<scalar_t>(), 
                batch, 
                features, 
                static_cast<float>(eps));
        })
    );
    return output;
}