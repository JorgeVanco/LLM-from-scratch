import torch
import torch.cuda.nvtx as nvtx
import numpy as np
import pandas as pd
from pathlib import Path
import timeit
import argparse
from contextlib import nullcontext

from src.model import TransformerLM
from src.utils import cross_entropy

@nvtx.range("Initialize model")
def initialize_model(args, device='cuda') -> TransformerLM:
    model = TransformerLM(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        vocab_size=10000,  # Example vocab size
        context_length=128,  # Example sequence length,
        rope_theta=10000.0,  # Example rope theta
    ).to(device)
    return model

def get_random_batch(vocab_size=10000, batch_size=4, seq_length=128, device='cuda') -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch_size, seq_length)).to(device), torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)

def benchmark_model(model, warmup_steps=5, benchmark_steps=10, mixed_precision=False) -> tuple[tuple[np.float32, np.float32], tuple[np.float32, np.float32]]:
    
    mp = torch.autocast("cuda", dtype=torch.bfloat16) if mixed_precision else nullcontext()
    with mp: 
        for _ in range(warmup_steps):
            batch = get_random_batch()
            model(batch[0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        times_forward = np.zeros(benchmark_steps)
        times_backward = np.zeros(benchmark_steps)
        print("Starting benchmarking...")
        for i in range(benchmark_steps):
            batch = get_random_batch()
            start_time = timeit.default_timer()
            logits = model(batch[0])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = timeit.default_timer() - start_time
            times_forward[i] = elapsed
            
            loss = cross_entropy(logits.view(-1, logits.size(-1)), batch[1].view(-1))
            start_time = timeit.default_timer()
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = timeit.default_timer() - start_time
            times_backward[i] = elapsed
        
        
    return (times_forward.mean(), times_forward.std()), (times_backward.mean(), times_backward.std())

def benchmark_model_nsys(model, warmup_steps=5, benchmark_steps=10, mixed_precision = False) -> None:
    mp = torch.autocast("cuda", dtype=torch.bfloat16) if mixed_precision else nullcontext()
    with mp:
        with nvtx.range("Warmup"):
            for _ in range(warmup_steps):
                batch = get_random_batch()
                model(batch[0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()

        for i in range(benchmark_steps):
            nvtx.range_push(f"Benchmark step {i+1}")
            with nvtx.range("Get random batch"):
                batch = get_random_batch()

            with nvtx.range("Forward pass"):
                logits = model(batch[0])

            with nvtx.range("Computing loss"):
                loss = cross_entropy(logits.view(-1, logits.size(-1)), batch[1].view(-1))

            with nvtx.range("Backward pass"):
                loss.backward()

            nvtx.range_pop()

def simple_benchmark(args, output_path='benchmark.csv') -> None:
    for model_args in [(768, 3072, 12, 12), (1024, 4096, 24, 16)]:#, (1280, 5120, 36, 20), (1600, 6400, 48, 25), (2560, 10240, 32, 32)]:
        args.d_model, args.d_ff, args.num_layers, args.num_heads = model_args
        print(f"Benchmarking model with d_model={args.d_model}, d_ff={args.d_ff}, num_heads={args.num_heads}, num_layers={args.num_layers}")

        model = initialize_model(args)

        (mean_time, std_time), (mean_time_backward, std_time_backward) = benchmark_model(model, args.warmup_steps, args.benchmark_steps)
        print(f"Mean time (forward): {mean_time:.4f} seconds")
        print(f"Standard deviation (forward): {std_time:.4f} seconds")
        print(f"Mean time (backward): {mean_time_backward:.4f} seconds")
        print(f"Standard deviation (backward): {std_time_backward:.4f} seconds")


        
        df = pd.DataFrame({
            "d_model": [args.d_model],
            "d_ff": [args.d_ff],
            "num_heads": [args.num_heads],
            "num_layers": [args.num_layers],
            "warmup_steps": [args.warmup_steps],
            "benchmark_steps": [args.benchmark_steps],
            "mean_time_forward": [mean_time],
            "std_time_forward": [std_time],
            "mean_time_backward": [mean_time_backward],
            "std_time_backward": [std_time_backward]
        })
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, mode='a', header=not output_path.exists(), index=False)

        df = pd.read_csv(output_path, index_col=False)
        print(df.to_markdown())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking script for BasicsTransformerLM")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--benchmark_steps", type=int, default=10)
    parser.add_argument("--mixed_precision", type=bool, default=False, help="Use mixed precision for benchmarking")
    
    args = parser.parse_args()
    output_path = 'profiler_output/benchmark.csv'
    simple_benchmark(args, output_path)
    