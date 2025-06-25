import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import reduce

def cross_entropy(logits: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]) -> Tensor:
    
    batch_size = logits.size(0)
    
    max_logit = logits.amax(dim=-1, keepdim=True)
    
    logits_subtracted = logits - max_logit
    return (- logits_subtracted[torch.arange(batch_size), targets] + logits_subtracted.exp().sum(-1).log()).mean()
    return reduce(- logits_subtracted[torch.arange(batch_size), targets] + logits_subtracted.exp().sum(-1).log(), "... batch_size -> ", "mean")