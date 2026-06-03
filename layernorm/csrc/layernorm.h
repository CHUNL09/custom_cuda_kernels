#pragma once


torch::Tensor layer_norm_cuda(
    torch::Tensor &input,
    torch::Tensor &gamma,
    torch::Tensor &beta,
    double eps
);