import torch
from collections.abc import Iterable

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    
    if not grads:
        return
    
    total_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
    
    if total_norm > max_l2_norm:
        for grad in grads:
            grad.mul_(max_l2_norm / (total_norm + eps))