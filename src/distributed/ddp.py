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
        params = [param.grad.data / dist.get_world_size() for param in self.module.parameters() if param.requires_grad]
        params_flatten = torch._utils._flatten_dense_tensors(params)
        dist.all_reduce(params_flatten, op=dist.ReduceOp.SUM)
        params_unflatten = torch._utils._unflatten_dense_tensors(params_flatten, params)
        
        i = 0
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad.data = params_unflatten[i]
                i += 1

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.module(*args, **kwargs)