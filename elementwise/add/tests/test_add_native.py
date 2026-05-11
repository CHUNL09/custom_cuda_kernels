import pytest
import torch
import add_native_ops

# 在每个测试文件中都需要的 fixture
@pytest.fixture(scope="module")
def device():
    """自动检测设备"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return 'cuda'

# 每个测试后清理 GPU 缓存
@pytest.fixture(autouse=True)
def cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ========== 基础功能测试 ==========

def test_basic_add(device):
    """基础功能测试：两个向量相加"""
    a = torch.randn(1000, device=device)
    b = torch.randn(1000, device=device)
    
    c_custom = add_native_ops.add_native(a, b)
    c_torch = a + b
    
    torch.testing.assert_close(c_custom, c_torch, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("shape", [
    (100,),
    (256, 256),
    (32, 32, 32),
    (1024, 1),
    (512, 512, 3),
    (1, 1000),
    (1000, 1000)
])
def test_different_shapes(device, shape):
    """测试不同形状"""
    a = torch.randn(*shape, device=device)
    b = torch.randn(*shape, device=device)
    
    c_custom = add_native_ops.add_native(a, b)
    c_torch = a + b
    
    torch.testing.assert_close(c_custom, c_torch, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("size", [0, 1, 10, 100, 1000, 10000, 100000, 1000000])
def test_various_sizes(device, size):
    """测试不同大小的输入"""
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    c_custom = add_native_ops.add_native(a, b)
    c_torch = a + b
    
    torch.testing.assert_close(c_custom, c_torch, rtol=1e-5, atol=1e-5)

# ========== 数据类型测试 ==========

@pytest.mark.parametrize("dtype", [
    torch.float16,
    torch.float32,
    torch.float64
])
def test_dtype_support(device, dtype):
    """测试不同数据类型"""
    if dtype == torch.float16 and device == 'cpu':
        pytest.skip("float16 requires CUDA")
    
    # float16 范围小一些避免溢出
    if dtype == torch.float16:
        a = torch.randn(100, device=device, dtype=dtype) * 0.1
        b = torch.randn(100, device=device, dtype=dtype) * 0.1
    else:
        a = torch.randn(100, device=device, dtype=dtype)
        b = torch.randn(100, device=device, dtype=dtype)
    
    c_custom = add_native_ops.add_native(a, b)
    c_torch = a + b
    
    atol = 1e-3 if dtype == torch.float16 else 1e-6
    torch.testing.assert_close(c_custom, c_torch, rtol=1e-3, atol=atol)

# ========== 边界情况测试 ==========

def test_edge_cases(device):
    """测试边界情况"""
    # 零值
    a = torch.zeros(100, device=device)
    b = torch.zeros(100, device=device)
    c = add_native_ops.add_native(a, b)
    assert torch.all(c == 0)
    
    # 负值
    a = torch.full((100,), -5.0, device=device)
    b = torch.full((100,), 3.0, device=device)
    c = add_native_ops.add_native(a, b)
    assert torch.all(c == -2.0)
    
    # 大数值
    a = torch.full((100,), 1e10, device=device)
    b = torch.full((100,), 1e10, device=device)
    c = add_native_ops.add_native(a, b)
    torch.testing.assert_close(c, torch.full((100,), 2e10, device=device))
    
    # 单元素
    a = torch.tensor([5.0], device=device)
    b = torch.tensor([3.0], device=device)
    c = add_native_ops.add_native(a, b)
    assert torch.allclose(c, torch.tensor([8.0], device=device))

def test_non_contiguous(device):
    """测试非连续内存"""
    # 创建非连续的张量（转置后）
    a = torch.randn(100, 100, device=device).t()
    b = torch.randn(100, 100, device=device).t()
    
    # 如果不支持非连续，应该报清晰的错误
    # with pytest.raises(RuntimeError, match="contiguous"):
    #     add_native_ops.add_native(a, b)

    assert torch.allclose(add_native_ops.add_native(a, b), a + b)

def test_device_mismatch():
    """测试设备不匹配"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    a = torch.randn(100, device='cuda')
    b = torch.randn(100, device='cpu')
    
    with pytest.raises(RuntimeError):
        add_native_ops.add_native(a, b)

# ========== 梯度测试 ==========

@pytest.mark.skip(reason="Gradient not implemented")
def test_gradient(device):
    """测试梯度传播"""
    a = torch.randn(100, device=device, requires_grad=True)
    b = torch.randn(100, device=device, requires_grad=True)
    
    c = add_native_ops.add_native(a, b)
    loss = c.sum()
    loss.backward()
    
    assert a.grad is not None
    assert b.grad is not None
    torch.testing.assert_close(a.grad, torch.ones_like(a))
    torch.testing.assert_close(b.grad, torch.ones_like(b))

@pytest.mark.skip(reason="Gradient not implemented")
def test_gradient_consistency(device):
    """测试梯度一致性（与 PyTorch 原生对比）"""
    for _ in range(10):
        a = torch.randn(50, device=device, requires_grad=True)
        b = torch.randn(50, device=device, requires_grad=True)
        
        # 自定义算子的梯度
        c_custom = add_native_ops.add_native(a, b)
        loss_custom = c_custom.sum()
        loss_custom.backward()
        grad_a_custom = a.grad.clone()
        grad_b_custom = b.grad.clone()
        
        # 清除梯度
        a.grad = None
        b.grad = None
        
        # PyTorch 原生的梯度
        c_torch = a + b
        loss_torch = c_torch.sum()
        loss_torch.backward()
        
        torch.testing.assert_close(grad_a_custom, a.grad, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(grad_b_custom, b.grad, rtol=1e-5, atol=1e-5)

# ========== 模型集成测试 ==========

@pytest.mark.skip(reason="Model integration not implemented")
def test_model_integration(device):
    """测试在实际模型中使用"""
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(128, 128)
        
        def forward(self, x):
            x = self.fc(x)
            # 添加偏置项
            x = add_native_ops.add_native(x, torch.ones_like(x))
            return x
    
    model = SimpleModel().to(device)
    x = torch.randn(32, 128, device=device)
    
    # 前向传播
    y = model(x)
    assert y.shape == (32, 128)
    
    # 反向传播
    loss = y.sum()
    loss.backward()
    
    # 检查梯度存在
    assert model.fc.weight.grad is not None
    assert model.fc.bias.grad is not None

# ========== 慢速测试（可选） ==========

@pytest.mark.slow
def test_large_tensor(device):
    """测试大张量"""
    size = 10_000_000  # 1000 万个元素
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    c = add_native_ops.add_native(a, b)
    assert c.numel() == size