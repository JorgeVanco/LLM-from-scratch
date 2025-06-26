import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_x = x.amax(dim=dim, keepdim=True)
    exponentiated_x = (x - max_x).exp()

    sftmax = exponentiated_x / exponentiated_x.sum(dim=dim, keepdim=True)

    return sftmax
