from collections.abc import Callable, Iterable
from typing import Optional, Any
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None) -> None | Any:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data-= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None) -> None | Any:
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                grad = p.grad.data
                
                if "t" not in state:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m = state["m"]
                v = state["v"]
                t = state["t"]

                m.mul_(betas[0]).add_((1 - betas[0]) * grad)
                v.mul_(betas[1]).addcmul_((1 - betas[1]), grad, grad)
                # m = betas[0] * m + (1 - betas[0]) * grad
                # v = betas[1] * v + (1 - betas[1]) * grad.square()
                
                lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)

                p.data -= lr_t * m / (v.sqrt() + eps)   # update parameters
                p.data -= lr * weight_decay * p.data    # apply weight decay

                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
                 
                                
        return loss