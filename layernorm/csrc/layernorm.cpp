#include "torch/extension.h"
#include "layernorm.h"


torch::Tensor  layer_norm_cpu(
    torch::Tensor &input,
    torch::Tensor &gamma,
    torch::Tensor &beta,
    double eps
){
    TORCH_CHECK(input.device().is_cpu(), "Input must be CPU tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor [batch, features]");
    int batch = input.size(0);
    int features = input.size(1);

    auto output = torch::empty_like(input);

    auto input_a = input.accessor<float, 2>(0);
    auto gamma_a = gamma.accessor<float, 1>();
    auto beta_a = beta.accessor<float, 1>();
    auto output_a = output.accessor<float, 2>();

    for (int b = 0; b < batch; ++b) {
        float sum = 0.0f, sum_sq = 0.0f;
        
        // 计算均值和方差
        for (int f = 0; f < features; ++f) {
            float val = input_a[b][f];
            sum += val;
            sum_sq += val * val;
        }
        
        float mean = sum / features;
        float var = sum_sq / features - mean * mean;
        float inv_std = 1.0f / std::sqrt(var + static_cast<float>(eps));
        
        // 归一化输出
        for (int f = 0; f < features; ++f) {
            float normalized = (input_a[b][f] - mean) * inv_std;
            output_a[b][f] = normalized * gamma_a[f] + beta_a[f];
        }
    }
    
    return output;
}


TORCH_LIBRARY(TORCH_EXTENSION_NAME, m) {
    m.def("forward(Tensor input, Tensor gamma, Tensor beta, float eps=1e-5) -> Tensor");
}

TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, CPU, m) {
    m.def("forward", layer_norm_cpu);
}

TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, CUDA, m) {
    m.def("forward", layer_norm_cuda);
}
