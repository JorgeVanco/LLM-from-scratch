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
    
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        
        torch.nn.init.trunc_normal_(self.weight, 0.0, 1.0, a = -3, b = 3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        RMS = (self.eps + x.square().sum(dim=-1, keepdim=True) / self.d_model).sqrt()
        result = x / RMS * self.weight
        
        result = result.to(in_dtype)
        
        return result
    

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        # d_ff should be 8/3 d_model, while ensuring that
        # the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your
        # hardware
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_x = silu(self.w1(x))
        element_product = silu_x * self.w3(x)
        return self.w2(element_product)
    

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        pass
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        return
    
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_x = x.amax(dim=dim, keepdim=True)
    exponentiated_x = (x - max_x).exp()
    
    softmax = exponentiated_x / exponentiated_x.sum(dim=dim, keepdim=True)
    
    return softmax