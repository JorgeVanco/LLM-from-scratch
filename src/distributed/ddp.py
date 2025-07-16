import torch
import torch.distributed as dist

class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module
        self.device = next(module.parameters()).device
        self.handles = []
        world_size = dist.get_world_size()
        
        def hook(param: torch.Tensor) -> None:
            param.grad.data /= world_size
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op = True)
            self.handles.append(handle)
        
        for param in module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook)
            
    def finish_gradient_synchronization(self) -> None:
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.module(*args, **kwargs)