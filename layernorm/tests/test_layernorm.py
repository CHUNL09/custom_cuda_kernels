import pytest
import torch
import torch.nn as nn
import time


def custom_layer_norm(input_tensor, gamma, beta, eps):
    return torch.ops.layernorm_ops.forward(input_tensor, gamma, beta, eps)


class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, tuple) else (normalized_shape,)
        self.eps = eps
        self.gamma = torch.nn.Parameter(torch.ones(normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return custom_layer_norm(x, self.gamma, self.beta, self.eps)


@pytest.fixture
def device():
    """测试设备fixture"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(autouse=True)
def set_seed():
    """固定随机种子，保证测试可复现"""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


class TestLayerNormFunctional:

    @pytest.mark.parametrize("batch,features,eps", [
        (32, 128, 1e-5),
        (1, 4096, 1e-5),
        (64, 8, 1e-4),
        (16, 1024, 1e-6),
        (128, 256, 1e-5),
        (2, 1, 1e-5),          # 单特征
        (0, 128, 1e-5),        # 空批次（如果支持）
    ], ids=[
        "standard", "single_sample", "small_features", 
        "medium", "large_batch", "single_feature", "empty_batch"
    ])
    def test_forward_accuracy(self, device, batch, features, eps):
        if batch == 0:
            pytest.skip("Empty batch not supported by all operators")
        
        input_tensor = torch.randn(batch, features, device=device)
        gamma = torch.randn(features, device=device)
        beta = torch.randn(features, device=device)

        custom_out = custom_layer_norm(input_tensor, gamma, beta, eps)
        native_out = torch.nn.functional.layer_norm(
            input_tensor, 
            (features,), gamma, beta, eps)
        
        torch.testing.assert_close(custom_out, native_out, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("batch,features,eps", [
        (32, 128, 1e-5),
        (16, 512, 1e-5),
        (8, 256, 1e-4),
    ])
    def test_backward_accuracy(self, device, batch, features, eps):
        def custom_func(x, g, b):
            return custom_layer_norm(x, g, b, eps)
        input_tensor = torch.randn(batch, features, device=device, dtype=torch.float64, requires_grad=True)
        gamma = torch.randn(features, device=device, dtype=torch.float64, requires_grad=True)
        beta = torch.randn(features, device=device, dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(
            custom_func,
            (input_tensor, gamma, beta),
            eps=1e-5,
            atol=1e-4,
        )
    
    @pytest.mark.parametrize("dtype", [
        torch.float32,
        torch.float64,
        pytest.param(torch.float16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 requires CUDA")),
        pytest.param(torch.bfloat16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 requires CUDA")),
    ])
    def test_dtype_support(self, device, dtype):
        """测试不同数据类型支持"""
        batch, features = 32, 128
        input_tensor = torch.randn(batch, features, device=device, dtype=dtype)
        gamma = torch.ones(features, device=device, dtype=dtype)
        beta = torch.zeros(features, device=device, dtype=dtype)
        
        custom_out = custom_layer_norm(input_tensor, gamma, beta, 1e-5)
        native_out = torch.nn.functional.layer_norm(input_tensor, (features,), gamma, beta, 1e-5)
        
        rtol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-4
        torch.testing.assert_close(custom_out, native_out, rtol=rtol, atol=1e-3)
    

    def test_module_integration(self, device):
        batch, features = 32, 128
        
        custom_norm = LayerNorm(features).to(device)
        native_norm = nn.LayerNorm(features).to(device)
        
        # 参数对齐
        custom_norm.gamma.data = native_norm.weight.data.clone()
        custom_norm.beta.data = native_norm.bias.data.clone()
        
        input_tensor = torch.randn(batch, features, device=device)
        
        custom_out = custom_norm(input_tensor)
        native_out = native_norm(input_tensor)

        torch.testing.assert_close(custom_out, native_out, rtol=1e-3, atol=1e-4)


class TestLayerNormErrorHandling:
    """验证算子的错误处理能力"""
    
    @pytest.fixture
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_dimension_mismatch_gamma(self, device):
        """gamma维度不匹配"""
        batch, features = 32, 128
        input_tensor = torch.randn(batch, features, device=device)
        wrong_gamma = torch.randn(features + 1, device=device)
        beta = torch.randn(features, device=device)
        
        with pytest.raises((ValueError, RuntimeError)):
            custom_layer_norm(input_tensor, wrong_gamma, beta, 1e-5)

    def test_dimension_mismatch_beta(self, device):
        """beta维度不匹配"""
        batch, features = 32, 128
        input_tensor = torch.randn(batch, features, device=device)
        gamma = torch.randn(features, device=device)
        wrong_beta = torch.randn(features + 1, device=device)
        
        with pytest.raises((ValueError, RuntimeError)):
            custom_layer_norm(input_tensor, gamma, wrong_beta, 1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_mismatch(self):
        """设备和参数不匹配"""
        batch, features = 32, 128
        cpu_input = torch.randn(batch, features)
        cuda_gamma = torch.randn(features, device='cuda')
        cuda_beta = torch.randn(features, device='cuda')
        
        with pytest.raises(RuntimeError):
            custom_layer_norm(cpu_input, cuda_gamma, cuda_beta, 1e-5)

    def test_invalid_eps(self, device):
        """无效的eps值"""    
        batch, features = 32, 128
        input_tensor = torch.randn(batch, features, device=device)
        gamma = torch.randn(features, device=device)
        beta = torch.randn(features, device=device)
        
        # eps为负数
        with pytest.raises((ValueError, RuntimeError)):
            custom_layer_norm(input_tensor, gamma, beta, -1e-5)
    

class TestLayerNormPerformance:
    """性能基准测试"""
    
    @pytest.fixture
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @pytest.mark.parametrize("batch,features", [
        (1024, 4096),   # 大张量
        (512, 8192),    # 超大特征
        (2048, 2048),   # 大批次
    ])
    def test_forward_performance(self, device, batch, features):
        """前向传播性能基准"""
        input_tensor = torch.randn(batch, features, device=device)
        gamma = torch.randn(features, device=device)
        beta = torch.randn(features, device=device)
        
        # 预热
        for _ in range(10):
            _ = custom_layer_norm(input_tensor, gamma, beta, 1e-5)
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # 计时
        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            _ = custom_layer_norm(input_tensor, gamma, beta, 1e-5)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed_ms = (time.perf_counter() - start) / iterations * 1000
        # 性能断言（根据实际情况调整阈值）
        threshold_ms = 10.0
        assert elapsed_ms < threshold_ms, \
            f"Forward pass too slow: {elapsed_ms:.3f}ms > {threshold_ms}ms"
        
        print(f"Performance ({batch}x{features} on {device}): {elapsed_ms:.3f} ms/iter")

    def test_throughput(self, device):
        """吞吐量测试（不同批次大小）"""
        features = 2048
        batch_sizes = [1, 8, 32, 128, 512]
        
        results = {}
        for batch in batch_sizes:
            input_tensor = torch.randn(batch, features, device=device)
            gamma = torch.randn(features, device=device)
            beta = torch.randn(features, device=device)
            
            # 预热
            for _ in range(5):
                _ = custom_layer_norm(input_tensor, gamma, beta, 1e-5)

            if device == 'cuda':
                torch.cuda.synchronize()
            
            # 计时
            start = time.perf_counter()
            iterations = 50
            for _ in range(iterations):
                _ = custom_layer_norm(input_tensor, gamma, beta, 1e-5)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            elapsed_ms = (time.perf_counter() - start) / iterations * 1000
            results[batch] = elapsed_ms
            
            print(f"Batch {batch}: {elapsed_ms:.3f} ms/iter")
        # 验证吞吐量随批次大小增长合理（不应超线性增长太多）
    