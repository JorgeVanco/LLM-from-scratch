import torch
import torch.distributed as dist

class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module
        self.device = next(module.parameters()).device
        
        for param in module.parameters():
            dist.broadcast(param.data, src=0)
            
    def finish_gradient_synchronization(self) -> None:
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad.data /= dist.get_world_size()
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)


    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.module(*args, **kwargs)