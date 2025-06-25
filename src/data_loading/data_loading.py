import torch
import numpy.typing as npt
import random

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:

    inputs = []
    targets = []
    max_index = len(dataset) - context_length - 1
    while len(inputs) < batch_size:
        i = random.randint(0, max_index)

        inputs.append(torch.LongTensor(dataset[i: i + context_length], device=device))
        targets.append(torch.LongTensor(dataset[i + 1: i + context_length + 1], device=device))
        
    return torch.stack(inputs), torch.stack(targets)