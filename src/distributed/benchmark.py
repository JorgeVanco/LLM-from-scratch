import os
from pathlib import Path
import timeit
from typing import Literal
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd

def setup(rank, world_size, backend="gloo") -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def all_reduce_benchmark_process(rank, world_size, backend, output_path: str) -> None:
    setup(rank, world_size, backend)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    
    
    for i in [1024**2 // 4, 1024**2 * 10 // 4, 1024**2 * 100 // 4, 1024**3 // 4]:
        data = torch.randn(i, dtype=torch.float32, device='cuda' if backend == "nccl" else 'cpu')
        if backend == "nccl":
            torch.cuda.synchronize()
        t0 = timeit.default_timer()
        dist.all_reduce(data, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()
        t1 = timeit.default_timer()
        elapsed = torch.tensor(t1 - t0)
        dist.all_reduce(elapsed, op=dist.ReduceOp.SUM ,async_op=False)
        elapsed /= world_size
        if rank == 0:
            data_size = data.element_size() * data.nelement() / 1024**2
            print(f"rank {rank} data size: {data_size}, time taken for all-reduce: {elapsed:.6f} seconds")
            if type(output_path) is str:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({
                "backend": [backend],
                "data_size (MB)": [data_size],
                "time_taken": [elapsed.item()]
            })
            df.to_csv(output_path, mode='a', header=not output_path.exists(), index=False)

def run_all_reduce_benchmark() -> None:
    """
    Run the all-reduce benchmark across multiple processes.
    This function is intended to be run in a distributed setting.
    """
    world_size = 4  # Number of processes
    backend: Literal["gloo", "nccl"] = "gloo"  # Change to "nccl" for GPU-based all-reduce
    output_path = "profiler_output/all_reduce_benchmark.csv"
    for backend in ["gloo", "nccl"]:
        print(f"Running all-reduce benchmark with backend: {backend}")
        mp.spawn(all_reduce_benchmark_process, args=(world_size, backend, output_path), nprocs=world_size, join=True)
    print(pd.read_csv(output_path).to_markdown())

if __name__ == "__main__":
    run_all_reduce_benchmark()
    print("All-reduce benchmark completed.")