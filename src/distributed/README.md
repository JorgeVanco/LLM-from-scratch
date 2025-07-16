The results for the `all reduce` benchmark.

|    | backend   |   data_size (MB) |   time_taken |    std_time |
|---:|:----------|-----------------:|-------------:|------------:|
|  0 | gloo      |                1 |  0.00260709  | 0.000921305 |
|  1 | gloo      |               10 |  0.00858294  | 0.000962009 |
|  2 | gloo      |              100 |  0.0737034   | 0.00614307  |
|  3 | gloo      |             1024 |  0.912003    | 0.139844    |
|  4 | nccl      |                1 |  4.75724e-05 | 3.10624e-05 |
|  5 | nccl      |               10 |  9.55087e-05 | 5.64021e-06 |
|  6 | nccl      |              100 |  0.000521944 | 6.47293e-05 |
|  7 | nccl      |             1024 |  0.00456288  | 4.72274e-05 |

The results for the simple `Individual DDP` benchmark.

|    | model_size   | backend   |   time_per_step |   std_time_step |   time_per_sync |   std_time_sync |   fraction_sync |
|---:|:-------------|:----------|----------------:|----------------:|----------------:|----------------:|----------------:|
|  0 | xl           | nccl      |        0.408564 |     0.000637803 |      0.0102613  |     0.000398993 |      0.0251155  |
|  1 | xl           | nccl      |        0.408698 |     0.00116798  |      0.0094661  |     7.51691e-05 |      0.0231616  |
|  2 | xl           | nccl      |        0.405134 |     0.000794918 |      0.00131627 |     7.47826e-05 |      0.00324898 |

Run 0 had the following implementation of the gradient synchronization:
```python
def finish_gradient_synchronization(self) -> None:
    for param in self.module.parameters():
        if param.requires_grad:
            param.grad.data /= dist.get_world_size()
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
```

Run 1 had the following implementation of the gradient synchronization:
```python
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
```

Run 2 overlaps computation and communication by calling `all_reduce` as soon as the gradient of the parameter is accumulated.
```python
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
```