import torch
import torch.distributed as dist
import numpy as np
from collections import defaultdict
from typing import Type, Any, Iterable, Optional, Callable

class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module
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


class DDPBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float) -> None:
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.handles = []
        self.world_size = dist.get_world_size()
        
        def hook(bucket: DDPBucket) -> None:
            handle, flat_gradients = bucket.add_gradient_accumulation_count()
            if handle is not None:
                self.handles.append((handle, flat_gradients, bucket))

        elements_per_bucket = (bucket_size_mb * 1024 * 1024) / next(module.parameters()).element_size()
        
        bucket = DDPBucket(bucket_size_mb)
        for param in reversed(list(module.parameters())):
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                if bucket.current_num_elements + param.numel() > elements_per_bucket:
                    bucket = DDPBucket(self.bucket_size_mb)
                bucket.add(param)
                param.register_post_accumulate_grad_hook(lambda param, bucket=bucket: hook(bucket))

    def finish_gradient_synchronization(self) -> None:
        for handle, flat_gradients, bucket in self.handles:
            handle.wait()
            unflat_gradients = torch._utils._unflatten_dense_tensors(flat_gradients, bucket.params)
            for param, grad in zip(bucket.params, unflat_gradients):
                param.grad.data = grad
        self.handles.clear()
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.module(*args, **kwargs)


class DDPBucket:
    def __init__(self, bucket_size_mb: float) -> None:
        self.bucket_size_mb = bucket_size_mb
        self.current_num_elements = 0
        self.params = []
        self.gradient_accumulation_count = 0
        
    def add(self, param: torch.nn.Parameter) -> None:
        self.params.append(param)
        self.current_num_elements += param.numel()
    
    def add_gradient_accumulation_count(self):
        self.gradient_accumulation_count += 1
        if self.gradient_accumulation_count == len(self.params):
            self.gradient_accumulation_count = 0
            self.current_num_elements = 0
            flat_gradients = torch._utils._flatten_dense_tensors([p.grad for p in self.params])
            flat_gradients = flat_gradients / dist.get_world_size()
            handle = dist.all_reduce(flat_gradients, op=dist.ReduceOp.SUM, async_op=True)
            return handle, flat_gradients
        return None, None
    

class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.Parameter | dict], optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any) -> None:
        params = list(params)
        self.optimizer = None
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.rank: int = dist.get_rank()
        self.world_size: int = dist.get_world_size()
        self._rank_params: defaultdict = defaultdict(list)
        self._size_per_rank: list = [0] * self.world_size
        super().__init__(params, {})
    
    def step(self, closure: Optional[Callable] = None, **kwargs) -> None:
        self.optimizer.step(closure=closure, **kwargs)
        self._sync()
        
    def _sync(self) -> None:
        handles = []
        for rank in self._rank_params:
            for p in self._rank_params[rank]:
                handle = dist.broadcast(p.data, src=rank, async_op = True)
                handles.append(handle)
                
        _ = [h.wait() for h in handles]
        
    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)

        param_group_to_add = defaultdict(list)
        for group_name, params in param_group.items():
            for p in params:
                min_rank = np.argmin(self._size_per_rank)
                self._rank_params[min_rank].append(p)
                self._size_per_rank[min_rank] += p.data.nbytes
                
                if self.rank == min_rank:
                    param_group_to_add[group_name].append(p)

        if self.optimizer is None:
            self.optimizer = self.optimizer_cls([param_group_to_add], **self.kwargs)
            del self.kwargs
        else:
            self.optimizer.add_param_group(param_group_to_add)