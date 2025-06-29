import torch
from torch.utils.data import Dataset, DataLoader
import numpy.typing as npt
import numpy as np
import random

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:

    inputs = []
    targets = []
    max_index = len(dataset) - context_length - 1
    while len(inputs) < batch_size:
        i = random.randint(0, max_index)

        inputs.append(torch.tensor(dataset[i: i + context_length], dtype=torch.long, device=device))
        targets.append(torch.tensor(dataset[i + 1: i + context_length + 1], dtype=torch.long, device=device))
        
    return torch.stack(inputs), torch.stack(targets)


class TextDataset(Dataset):
    def __init__(self, dataset: npt.NDArray, context_length: int = 1024) -> None:
        self.dataset = dataset
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.dataset) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")
        
        input_seq = torch.tensor(self.dataset[idx: idx + self.context_length], dtype=torch.long)
        target_seq = torch.tensor(self.dataset[idx + 1: idx + self.context_length + 1], dtype=torch.long)
        
        return input_seq, target_seq
    

def get_dataloader(
    dataset: npt.NDArray,
    batch_size: int = 32,
    context_length: int = 1024,
    shuffle: bool = True,
    num_workers: int = 0,
    generator: torch.Generator | None = None,
    pin_memory: bool = False,
    pin_memory_device: str = "",
    drop_last: bool = False
) -> DataLoader:
    
    dataset = TextDataset(dataset, context_length)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
        drop_last=drop_last
    )
    

def load_dataset(
    dataset_path: str,
    batch_size: int = 32,
    context_length: int = 1024,
    shuffle: bool = True,
    num_workers: int = 0,
    generator: torch.Generator | None = None,
    pin_memory: bool = False,
    pin_memory_device: str = "",
    drop_last: bool = False) -> DataLoader:
    
    dataset = np.memmap(dataset_path, dtype=np.uint16)
    dataloader = get_dataloader(dataset,
                                batch_size=batch_size,
                                context_length=context_length,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                generator=generator,
                                pin_memory=pin_memory,
                                pin_memory_device=pin_memory_device,
                                drop_last=drop_last
                            )
    return dataloader