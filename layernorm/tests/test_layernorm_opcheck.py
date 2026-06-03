import pytest
import torch


class TestLayerNormOp:
    """LayerNorm算子的opcheck验证"""
    
    @pytest.fixture(autouse=True)
    def check_operator_registered(self):
        """前置条件：算子必须已注册"""
        if not hasattr(torch.ops, 'layernorm_ops') or \
           not hasattr(torch.ops.layernorm_ops, 'forward'):
            pytest.skip("torch.ops.layernorm_ops.forward not registered")
    
    def test_opcheck_standard_case(self):
        """标准用例验证算子注册正确性"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch, features = 32, 128
        
        args = (
            torch.randn(batch, features, device=device),
            torch.ones(features, device=device),
            torch.zeros(features, device=device),
            1e-5
        )
        
        torch.library.opcheck(
            torch.ops.layernorm_ops.forward,
            args,
            test_utils=["test_schema", "test_faketensor", "test_aot_dispatch_dynamic"]
        )
    
    def test_opcheck_with_gradients(self):
        """带梯度的用例(检查autograd支持)"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch, features = 32, 128
        
        args = (
            torch.randn(batch, features, device=device, requires_grad=True),
            torch.ones(features, device=device, requires_grad=True),
            torch.zeros(features, device=device, requires_grad=True),
            1e-5
        )
        
        torch.library.opcheck(
            torch.ops.layernorm_ops.forward,
            args,
            test_utils=["test_schema", "test_faketensor", "test_aot_dispatch_dynamic"]
        )
    
    def test_opcheck_different_device(self):
        """CPU和CUDA设备(如果可用)"""
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        
        for device in devices:
            batch, features = 32, 128
            args = (
                torch.randn(batch, features, device=device),
                torch.ones(features, device=device),
                torch.zeros(features, device=device),
                1e-5
            )
            
            torch.library.opcheck(
                torch.ops.layernorm_ops.forward,
                args,
                test_utils=["test_schema", "test_faketensor", "test_aot_dispatch_dynamic"]
            )