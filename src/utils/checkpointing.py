import os
import torch
from typing import BinaryIO, IO

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration
        },
        out
    )


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
    obj = torch.load(src)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in obj["model"].items()})
    if optimizer is not None:
        optimizer.load_state_dict(obj["optimizer"])

    return obj["iteration"]