import torch
import torch.nn as nn
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        
        std: float = (2 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, 0.0, std, a = -3*std, b = 3*std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")