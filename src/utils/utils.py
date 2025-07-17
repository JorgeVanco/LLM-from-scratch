import torch

# Note: For some reason casting the input to float32 does not work unless you put a print statement anywhere in the function
torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:

    max_x = x.amax(dim=dim, keepdim=True)
    exponentiated_x = (x - max_x).exp()

    sftmax = exponentiated_x / exponentiated_x.sum(dim=dim, keepdim=True)

    return sftmax
