#include <torch/extension.h>

void launch_elementwise_add_native(const float* a, const float* b, float* c, int N);

torch::Tensor add_native(torch::Tensor a, torch::Tensor b){
    TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda(), "a and b must be on CUDA device");
    TORCH_CHECK(a.numel() == b.numel(), "a and b must have the same size");
    torch::Tensor result = torch::zeros_like(a);
    launch_elementwise_add_native(a.data_ptr<float>(), b.data_ptr<float>(), result.data_ptr<float>(), a.numel());
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_native", &add_native, "Add two tensors elementwise on CUDA");
}