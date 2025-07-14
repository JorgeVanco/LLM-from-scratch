import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import pandas as pd
from pathlib import Path
import timeit
import argparse
from contextlib import nullcontext

from src.model import TransformerLM
from src.model.models import scaled_dot_product_attention
from src.utils import cross_entropy

configs = {
    'small': {'d_model': 768, 'num_layers': 12, 'num_heads': 12},
    'medium': {'d_model': 1024, 'num_layers': 24, 'num_heads': 16},
    'large': {'d_model': 1280, 'num_layers': 36, 'num_heads': 20},
    'xl': {'d_model': 1600, 'num_layers': 48, 'num_heads': 25},
    '2.7B': {'d_model': 2560, 'num_layers': 32, 'num_heads': 32},
}


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
    if args.compile:
        model = torch.compile(model, fullgraph=True, dynamic=True)
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
        
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("dtype after fc1:", x.dtype)
        x = self.ln(x)
        print("dtype after LayerNorm:", x.dtype)
        x = self.fc2(x)
        print("dtype after fc2:", x.dtype)
        return x        

def test_autocast() -> None:
    model = ToyModel(10, 5).to('cuda')
    x = torch.randn(2, 10).to('cuda')
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(x)
        cross_entropy = F.cross_entropy(output, torch.tensor([1, 2], device='cuda', dtype=torch.int64))
        cross_entropy.backward()
        print("Output:", output.dtype)
        print("Loss:", cross_entropy.dtype)
        for name, param in model.named_parameters():
            print(f"Parameter {name} has dtype {param.dtype} and gradient dtype {param.grad.dtype}")
        
def memory_profile(model, warmup_steps=5, benchmark_steps=10, mixed_precision=False, backward=True) -> None:
    output_path = Path("profiler_output/memory_snapshot.pickle")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = 'cuda:0'
    
    grad = nullcontext() if backward else torch.no_grad()
    mp = torch.autocast("cuda", dtype=torch.bfloat16) if mixed_precision else nullcontext()
    if backward:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with grad:
        with mp: 
            with profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=0, warmup=warmup_steps, active=benchmark_steps, repeat=1),
                experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as p:
                for i in range(benchmark_steps + warmup_steps):
                    if i == warmup_steps:
                        # Start recording memory history.
                        torch.cuda.memory._record_memory_history(max_entries=1000000)
                    if backward:
                        optimizer.zero_grad()
                        
                    batch = get_random_batch()
                    logits = model(batch[0])
                    
                    if backward:
                        loss = cross_entropy(logits.view(-1, logits.size(-1)), batch[1].view(-1))
                        loss.backward()
                        
                        optimizer.step()
                    p.step()
        
    # Save a pickle file to be loaded by PyTorch's online tool.
    torch.cuda.memory._dump_snapshot(output_path)
    
    # Stop recording history.
    torch.cuda.memory._record_memory_history(enabled=None)
    p.export_memory_timeline("profiler_output/memory_timeline.html", device=device)
    
    print(f"torch.cuda.memory_allocated(0): {torch.cuda.memory_allocated(0)/ (1024**2)} MiB")
    print(f"torch.cuda.max_memory_allocated(0): {torch.cuda.max_memory_allocated(0)/ (1024**3)} GiB")

def profile_attention(compile: bool = False) -> None:
    if compile:
        scaled_dot_product_attention = torch.compile(scaled_dot_product_attention, fullgraph=True, dynamic=True)
    output_path = Path("profiler_output/attention_profile.csv")
    for d_head in [16, 32, 64, 128]:
        for seq_len in [256, 1024, 4096, 8192, 16384]:
            Q = torch.randn(8, seq_len, d_head, device='cuda', requires_grad=True) # Batch size, sequence length, head dimension
            K = torch.randn(8, seq_len, d_head, device='cuda', requires_grad=True)
            V = torch.randn(8, seq_len, d_head, device='cuda', requires_grad=True)
            try:
                # Warmup iterations
                for _ in range(10):
                    output = scaled_dot_product_attention(Q, K, V, ~torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool())

                del output
                torch.cuda.reset_max_memory_allocated(0)
                torch.cuda.synchronize()  # Ensure all operations are complete before timing
                
                forward_times = []
                backward_times = []
                for _ in range(100):
                    start_time = timeit.default_timer()
                    output = scaled_dot_product_attention(Q, K, V,~torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool())
                    torch.cuda.synchronize()
                    elapsed = timeit.default_timer() - start_time
                    forward_times.append(elapsed)
                max_memory = torch.cuda.max_memory_allocated(0) / (1024**2)  # Convert to MiB
                loss = output.sum()
                for _ in range(100):
                    for param in [Q, K, V]:
                        param.grad = None  # Clear gradients
                    torch.cuda.synchronize()
                    start_time = timeit.default_timer()
                    loss.backward(retain_graph=True)
                    torch.cuda.synchronize()
                    elapsed = timeit.default_timer() - start_time
                    backward_times.append(elapsed)
                forward_mean = np.mean(forward_times)
                forward_std = np.std(forward_times)
                backward_mean = np.mean(backward_times)
                backward_std = np.std(backward_times)
            except torch.cuda.OutOfMemoryError:
                forward_mean = np.nan
                forward_std = np.nan
                backward_mean = np.nan
                backward_std = np.nan
                max_memory = torch.cuda.max_memory_allocated(0) / (1024**2)
            
            df = pd.DataFrame({
                "compile": [compile],
                "d_head": [d_head],
                "seq_len": [seq_len],
                "forward_mean": [forward_mean],
                "forward_std": [forward_std],
                "backward_mean": [backward_mean],
                "backward_std": [backward_std],
                "max_memory (MiB)": [max_memory]
            })
            df.to_csv(output_path, mode='a', header=not output_path.exists(), index=False)
            print(f"Profiled attention with d_head={d_head}, seq_len={seq_len}")
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
    parser.add_argument("--mixed_precision",action="store_true", help="Use mixed precision for benchmarking")
    parser.add_argument("--no_backward", action="store_true", help="Benchmark backward pass if True")
    parser.add_argument("--compile", action="store_true", help="Compile the model with torch.compile")
    args = parser.parse_args()
    output_path = 'profiler_output/benchmark.csv'
    # simple_benchmark(args, output_path)

    # model = initialize_model(args)
    # memory_profile(model, args.warmup_steps, args.benchmark_steps, args.mixed_precision, not args.no_backward)
    
    profile_attention(args.compile)
    