#!/usr/bin/env python
"""
详细的 NCU 分析脚本，用于测量不同配置下的性能

ncu --set full -k elementwise_add_native -o kernel_profile python ncu_profile.py
"""

import torch
import add_native_ops
import argparse

def profile_kernel(size=1024*1024, iterations=10, device='cuda'):
    """运行 kernel 供 NCU 分析"""
    
    print(f"Profiling configuration:")
    print(f"  Size: {size} elements ({size*4/1024/1024:.2f} MB)")
    print(f"  Iterations: {iterations}")
    print(f"  Device: {device}")
    
    # 创建数据
    a = torch.randn(size, device=device)
    b = torch.randn(size, device=device)
    
    # 同步并清空缓存
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    # 标记 NCU 分析的起点（可选）
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push("Kernel_Execution")
    
    # 🔴 修复：实际需要分析的 kernel 调用
    for i in range(iterations):
        if i == iterations // 2:
            # 在中间那次调用打标记，方便 NCU 定位
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push(f"Iteration_{i}")
        
        # ⭐⭐⭐ 这才是关键：调用你的自定义算子 ⭐⭐⭐
        c = add_native_ops.add_native(a, b)
        
        if i == iterations // 2:
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()
    
    # 同步确保所有 kernel 执行完成
    torch.cuda.synchronize()
    
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()
    
    print(f"✅ Completed {iterations} kernel executions")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024*1024, 
                        help="Number of elements (default: 1M)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of kernel iterations (default: 10)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    
    args = parser.parse_args()
    
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    # 运行 profile
    profile_kernel(
        size=args.size,
        iterations=args.iterations,
        device=args.device
    )

if __name__ == "__main__":
    main()