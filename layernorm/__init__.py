import torch
from . import _C

def layer_norm(input, gamma, beta, eps=1e-5):
    return torch.ops.layernorm_ops._C.forward(input, gamma, beta, eps)
