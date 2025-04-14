# utils/torch_helper.py
import torch

def eye_like(ref: torch.Tensor, n: int = None):
    """生成与 ref 同 dtype/device 的单位阵 (n×n)。"""
    n = n or ref.size(-1)
    return torch.eye(n, dtype=ref.dtype, device=ref.device)

def zeros_like_shape(shape, ref: torch.Tensor):
    return torch.zeros(*shape, dtype=ref.dtype, device=ref.device)
