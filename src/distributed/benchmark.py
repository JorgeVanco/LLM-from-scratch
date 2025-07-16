import os
from pathlib import Path
import timeit
from typing import Literal
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
from src.distributed import DDPIndividualParameters, DDPBucketed
from src.model import TransformerLM
from src.profiling import get_random_batch
from src.utils import cross_entropy

configs = {
    'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
    'medium': {'d_model': 1024, 'num_layers': 24, 'num_heads': 16},
    'large': {'d_model': 1280, 'num_layers': 36, 'num_heads': 20},
    'xl': {'d_model': 1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
    '2.7B': {'d_model': 2560, 'num_layers': 32, 'num_heads': 32},
}

def setup(rank, world_size, backend="gloo") -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def all_reduce_benchmark_process(rank, world_size, backend, output_path: str, warmup_steps: int = 5, benchmark_steps: int = 20) -> None:
    setup(rank, world_size, backend)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    
    device = torch.device('cuda' if backend == "nccl" else 'cpu')
    for i in [1024**2 // 4, 1024**2 * 10 // 4, 1024**2 * 100 // 4, 1024**3 // 4]:
        times = []
        for j in range(warmup_steps + benchmark_steps):
            data = torch.randn(i, dtype=torch.float32, device=device)
            if backend == "nccl":
                torch.cuda.synchronize()
            t0 = timeit.default_timer()
            dist.all_reduce(data, async_op=False)
            if backend == "nccl":
                torch.cuda.synchronize()
            t1 = timeit.default_timer()
            if j >= warmup_steps:
                elapsed = torch.tensor(t1 - t0).to(device)
                dist.all_reduce(elapsed, op=dist.ReduceOp.SUM ,async_op=False)
                elapsed /= world_size
                times.append(elapsed.item())

        if rank == 0:
            data_size = data.element_size() * data.nelement() / 1024**2
            print(f"rank {rank} data size: {data_size}, time taken for all-reduce: {elapsed:.6f} seconds")
            if type(output_path) is str:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({
                "backend": [backend],
                "data_size (MB)": [data_size],
                "time_taken": [np.mean(times)],
                "std_time": [np.std(times)],
            })
            df.to_csv(output_path, mode='a', header=not output_path.exists(), index=False)
   
            
def ddp_individual_parameters_benchmark(rank: int, world_size: int, backend: str, output_path: str, warmup_steps: int = 5, benchmark_steps: int = 20) -> None:
    setup(rank, world_size, backend)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    device = torch.device('cuda' if backend == "nccl" else 'cpu')
    model = TransformerLM(**configs['xl'], vocab_size=10000, context_length=256, rope_theta=10000).to(device)
    ddp_model = DDPIndividualParameters(model)

    if backend == "nccl":
        torch.cuda.synchronize()

    train_step_times = []
    sync_times = []
    for i in range(warmup_steps + benchmark_steps):
        batch = get_random_batch(10000, 4, 256, device=device)
        t0 = timeit.default_timer()
        logits = ddp_model(batch[0])
        loss = cross_entropy(logits.view(-1, logits.size(-1)), batch[1].view(-1))
        loss.backward()
        if backend == "nccl":
            torch.cuda.synchronize()
        t_sync = timeit.default_timer()
        ddp_model.finish_gradient_synchronization()
        if backend == "nccl":
            torch.cuda.synchronize()
        t1 = timeit.default_timer()
        elapsed = t1 - t0
        elapsed_sync = t1 - t_sync
        if i >= warmup_steps:
            train_step_times.append(elapsed)
            sync_times.append(elapsed_sync)
    if rank == 0:
        if type(output_path) is str:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "model_size": ["xl"],
            "backend": [backend],
            "time_per_step": [np.mean(train_step_times)],
            "std_time_step": [np.std(train_step_times)],
            "time_per_sync": [np.mean(sync_times)],
            "std_time_sync": [np.std(sync_times)],
            "fraction_sync": [np.mean(sync_times) / np.mean(train_step_times)],
        })
        df.to_csv(output_path, mode='a', header=not output_path.exists(), index=False)


def ddp_bucketed_benchmark(rank: int, world_size: int, backend: str, output_path: str, warmup_steps: int = 5, benchmark_steps: int = 20) -> None:
    setup(rank, world_size, backend)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    device = torch.device('cuda' if backend == "nccl" else 'cpu')
    model = TransformerLM(**configs['xl'], vocab_size=10000, context_length=256, rope_theta=10000).to(device)

    for bucket_size_mb in [1, 10, 100, 1000]:
        if rank == 0:
            print(f"Running DDPBucketed with bucket size: {bucket_size_mb} MB")
        ddp_model = DDPBucketed(model, bucket_size_mb=bucket_size_mb)

        if backend == "nccl":
            torch.cuda.synchronize()

        train_step_times = []
        sync_times = []
        for i in range(warmup_steps + benchmark_steps):
            batch = get_random_batch(10000, 4, 256, device=device)
            t0 = timeit.default_timer()
            logits = ddp_model(batch[0])
            loss = cross_entropy(logits.view(-1, logits.size(-1)), batch[1].view(-1))
            loss.backward()
            if backend == "nccl":
                torch.cuda.synchronize()
            t_sync = timeit.default_timer()
            ddp_model.finish_gradient_synchronization()
            if backend == "nccl":
                torch.cuda.synchronize()
            t1 = timeit.default_timer()
            elapsed = t1 - t0
            elapsed_sync = t1 - t_sync
            if i >= warmup_steps:
                train_step_times.append(elapsed)
                sync_times.append(elapsed_sync)
        if rank == 0:
            if type(output_path) is str:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({
                "model_size": ["xl"],
                "backend": [backend],
                "bucket_size_mb": [bucket_size_mb],
                "time_per_step": [np.mean(train_step_times)],
                "std_time_step": [np.std(train_step_times)],
                "time_per_sync": [np.mean(sync_times)],
                "std_time_sync": [np.std(sync_times)],
                "fraction_sync": [np.mean(sync_times) / np.mean(train_step_times)],
            })
            df.to_csv(output_path, mode='a', header=not output_path.exists(), index=False)


def run_all_reduce_benchmark() -> None:
    """
    Run the all-reduce benchmark across multiple processes.
    This function is intended to be run in a distributed setting.
    """
    world_size = 4  # Number of processes
    backend: Literal["gloo", "nccl"] = "gloo"  # Change to "nccl" for GPU-based all-reduce
    output_path = "benchmarks/all_reduce_benchmark.csv"
    for backend in ["gloo", "nccl"]:
        print(f"Running all-reduce benchmark with backend: {backend}")
        mp.spawn(all_reduce_benchmark_process, args=(world_size, backend, output_path), nprocs=world_size, join=True)
    print(pd.read_csv(output_path).to_markdown())
    
def run_ddp_individual_parameters_benchmark() -> None:
    """
    Run the DDPIndividualParameters benchmark across multiple processes.
    This function is intended to be run in a distributed setting.
    """
    world_size = 1  # Number of processes
    backend: Literal["gloo", "nccl"] = "nccl"
    output_path = "benchmarks/ddp_individual_parameters_benchmark.csv"
    print(f"Running DDPIndividualParameters benchmark with backend: {backend}")
    mp.spawn(ddp_individual_parameters_benchmark, args=(world_size, backend, output_path), nprocs=world_size, join=True)
    print(pd.read_csv(output_path).to_markdown())
    

def run_ddp_bucketed_benchmark() -> None:
    """
    Run the DDPBucketed benchmark across multiple processes.
    This function is intended to be run in a distributed setting.
    """
    world_size = 1  # Number of processes
    backend: Literal["gloo", "nccl"] = "nccl"
    output_path = "benchmarks/ddp_bucketed_benchmark.csv"
    print(f"Running DDPBucketed benchmark with backend: {backend}")
    mp.spawn(ddp_bucketed_benchmark, args=(world_size, backend, output_path), nprocs=world_size, join=True)
    print(pd.read_csv(output_path).to_markdown())

if __name__ == "__main__":
    run_all_reduce_benchmark()
    print("All-reduce benchmark completed.")
    
    run_ddp_individual_parameters_benchmark()
    print("DDPIndividualParameters benchmark completed.")
    
    run_ddp_bucketed_benchmark()
    print("DDPBucketed benchmark completed.")