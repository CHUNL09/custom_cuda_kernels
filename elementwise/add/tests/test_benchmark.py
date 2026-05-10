import pytest
import torch
import add_native_ops
import time

# 在这也定义 fixture
@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return 'cuda'

@pytest.fixture(autouse=True)
def cleanup():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ========== 性能基准测试 ==========

class TestPerformance:
    
    def test_custom_vs_pytorch(self, device):
        """比较性能和 PyTorch 原生操作"""
        sizes = [1000, 10000, 100000, 1000000]
        results = []
        
        for size in sizes:
            a = torch.randn(size, device=device)
            b = torch.randn(size, device=device)
            
            # 预热
            for _ in range(10):
                _ = add_native_ops.add_native(a, b)
                _ = a + b
            torch.cuda.synchronize()
            
            # 自定义算子性能
            start = time.perf_counter()
            for _ in range(100):
                _ = add_native_ops.add_native(a, b)
            torch.cuda.synchronize()
            custom_time = (time.perf_counter() - start) / 100 * 1000
            
            # PyTorch 原生性能
            start = time.perf_counter()
            for _ in range(100):
                _ = a + b
            torch.cuda.synchronize()
            torch_time = (time.perf_counter() - start) / 100 * 1000
            
            ratio = torch_time / custom_time
            results.append((size, custom_time, torch_time, ratio))
            
            print(f"\nSize {size:10d}: Custom={custom_time:.3f}ms, Torch={torch_time:.3f}ms, Ratio={ratio:.2f}x")
            
            # 性能不应该比 PyTorch 慢太多（2倍以内）
            assert custom_time < torch_time * 2, f"Custom kernel too slow at size {size}"
        
        print("\n✅ Performance test passed")
        
        # 可以在这里做更多的性能分析
        # 检查是否大规模数据下有合理的性能缩放
        for i in range(1, len(results)):
            size_ratio = results[i][0] / results[i-1][0]
            time_ratio = results[i][1] / results[i-1][1]
            assert time_ratio < size_ratio * 2, "Performance scaling is not linear"
    
    def test_memory_bandwidth(self, device):
        """估算内存带宽利用率"""
        size = 100_000_000  # 1亿个元素
        if torch.cuda.get_device_properties(0).total_memory < size * 4 * 3:
            pytest.skip("Not enough GPU memory")
        
        a = torch.randn(size, device=device)
        b = torch.randn(size, device=device)
        
        # 理论内存访问量: 读 a, 读 b, 写 c = 3 * size * 4 bytes
        bytes_transferred = 3 * size * 4 / 1e9  # GB
        
        # 测量时间
        torch.cuda.synchronize()
        start = time.perf_counter()
        c = add_native_ops.add_native(a, b)
        torch.cuda.synchronize()
        elapsed_time = time.perf_counter() - start
        
        bandwidth = bytes_transferred / elapsed_time  # GB/s
        
        print(f"\nData transferred: {bytes_transferred:.2f} GB")
        print(f"Elapsed time: {elapsed_time*1000:.2f} ms")
        print(f"Achieved bandwidth: {bandwidth:.2f} GB/s")
        
        # 获取 GPU 理论峰值带宽（这里是一个大致估计）
        theoretical_bandwidth = 900  # V100 约 900 GB/s，根据实际 GPU 调整
        
        utilization = (bandwidth / theoretical_bandwidth) * 100
        print(f"Bandwidth utilization: {utilization:.1f}%")
        
        # 这个只是参考，不作为断言
        assert bandwidth > 0
    
    @pytest.mark.skip(reason="Block size test not implemented")
    @pytest.mark.parametrize("block_size", [64, 128, 256, 512, 1024])
    def test_block_size_impact(self, device, block_size):
        """测试不同 block size 对性能的影响"""
        # 注意：这个测试需要你的 kernel 支持可配置的 block size
        size = 10_000_000
        a = torch.randn(size, device=device)
        b = torch.randn(size, device=device)
        
        # 这里假设你的算子可以接收 block_size 参数
        # 如果不行，这个测试可以跳过
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            # 如果你的算子不支持，注释掉这个测试
            # c = add_native_ops.add_native_with_block_size(a, b, block_size)
            pass
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        print(f"\nBlock size {block_size:4d}: {elapsed*100:.2f}ms per 10 runs")
    
    def test_with_ncu_analysis(self, device):
        """提示用户使用 NCU 进行深度分析"""
        print("\n" + "="*60)
        print("For deeper kernel analysis, run NCU command:")
        print("ncu --set full -k elementwise_add_native -o kernel_profile ./test_script.py")
        print("="*60)

# ========== 专门的压力测试 ==========

class TestStress:
    
    def test_many_small_ops(self, device):
        """大量小操作的性能测试"""
        n_ops = 1000
        size = 100
        
        # 预分配内存避免内存分配开销
        tensors_a = [torch.randn(size, device=device) for _ in range(n_ops)]
        tensors_b = [torch.randn(size, device=device) for _ in range(n_ops)]
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for a, b in zip(tensors_a, tensors_b):
            c = add_native_ops.add_native(a, b)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        print(f"\n{n_ops} small ops (size={size}): {elapsed*1000:.2f}ms")
        print(f"Average per op: {elapsed/n_ops*1000:.3f}ms")
    
    def test_memory_fragmentation(self, device):
        """测试长时间运行后的内存碎片问题"""
        sizes = [1000, 10000, 100000, 1000000]
        
        for i in range(10):
            for size in sizes:
                a = torch.randn(size, device=device)
                b = torch.randn(size, device=device)
                c = add_native_ops.add_native(a, b)
                # 让 c 被释放
                del c
                del a
                del b
        
        # 检查内存是否正常
        torch.cuda.empty_cache()
        memory_allocated = torch.cuda.memory_allocated()
        print(f"\nMemory allocated after stress test: {memory_allocated / 1024**2:.2f} MB")
        assert memory_allocated < 100 * 1024**2  # < 100 MB