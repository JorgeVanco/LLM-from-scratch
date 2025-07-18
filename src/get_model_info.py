import torch
import itertools
import argparse

from src.data_loading.data_loading import load_dataset
from src.model import TransformerLM
from src.config import ConfigManager
from src.utils import cross_entropy, gradient_clipping
from src.train import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLM from scratch")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config file")
    args = parser.parse_args()
    config = ConfigManager.load_config(args.config)
    config.logging.use_wandb = False
    config.training.compile_model = False
    trainer = Trainer(config)
    config = ConfigManager._config_to_dict(config)
    

    model = TransformerLM(**config['model'])
    # model = torch.compile(model)
    print("Model Info:")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    vocab = config['model']['vocab_size']
    seq = config['model']['context_length']
    d_model = config['model']['d_model']
    d_ff = config['model']['d_ff']
    num_layers = config['model']['num_layers']
    num_heads = config['model']['num_heads']
    
    num_params = 2*vocab*d_model + num_layers * (2*d_model + 4*d_model**2 + 3*d_model*d_ff) + d_model
    print(f"Total parameters in the model: {num_params}")
    print(f"Total memory usage in the model: {num_params * (4 if trainer.dtype == torch.float32 else 2) / (1024**2)} MiB")

    gradients = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total memory usage of gradients in the model: {gradients * 4 / (1024**2)} MiB")
    adamw = 2 * num_params
    print(f"Total memory usage of AdamW parameters: {adamw * 4 / (1024**2)} MiB")

    batch_size = config['training']['batch_size']
    activations = (num_layers * (7*seq*d_model + 3 * d_model * d_ff + 2*num_heads * seq**2) + seq * d_model + seq * vocab) * batch_size
    print(f"Total memory usage of activations: {activations * 4 / (1024**3)} GiB")

    print(f"Total memory usage in the model: {num_params * (4 if trainer.dtype == torch.float32 else 2) / (1024**2)} MiB")
    print(f"Total memory usage in training: {(num_params * (4 if trainer.dtype == torch.float32 else 2) + (gradients + adamw + activations) * 4) / (1024**3)} GiB")

    print(f"FLOPs in forward pass: ")
    
    data = load_dataset(
        dataset_path=config['training']['train_data_path'],
        batch_size=config['training']['batch_size'],
        context_length=config['model']['context_length'],
        shuffle=True,
        num_workers=0,
        generator=None,
        pin_memory=False,
        pin_memory_device="",
        drop_last=False
    )
    
    # x, y = next(iter(data))
    # x = x.to('cuda')
    # data = iter(data)
    
    # # model = torch.nn.Sequential(torch.nn.Embedding(vocab, d_model), torch.nn.Linear(d_model, d_model), torch.nn.Linear(d_model, d_model), torch.nn.Linear(d_model, d_model), torch.nn.Linear(d_model, d_model), torch.nn.Linear(d_model, d_model), torch.nn.Linear(d_model, d_model), torch.nn.Linear(d_model, vocab)).to('cuda')
    # model = torch.compile(model)
    # model= model.to('cuda')
    
    
    # # model = torch.nn.Embedding(vocab, vocab).to('cuda')
    
    # optimizer = AdamW(
    #     model.parameters(),
    #     lr=config['optimizer']['lr'],
    #     betas=config['optimizer']['betas'],
    #     eps=config['optimizer']['eps'],
    #     weight_decay=config['optimizer']['weight_decay'],
    # )
    
    
    
    torch.cuda.memory._record_memory_history()
    # from typing import TypedDict, List


    # class Frame(TypedDict):
    #     filename: str
    #     line: int
    #     name: str
        
    # class History(TypedDict):
    #     addr: int
    #     frames : List[Frame] # stack trace when address was last allocated
    #                         # most recent frame first
    #     real_size: int # unrounded size requested from the allocator
    # class Block(TypedDict):
    #     size: int
    #     state: str # 'active_allocated', used by a tensor
    #             # 'active_awaiting_free', we are waiting for another stream to finish using
    #             #                         this, then it will become free
    #             # 'inactive', free for reuse
    #     history: List[History]
    # class Segment(TypedDict):
    #     address: int
    #     total_size: int #  cudaMalloc'd size of segment
    #     stream: int
    #     segment_type: str # 'large' (>1MB) or 'small'
    #     allocated_size: int # size of memory in use
    #     active_size: int # size of memory in use or in active_awaiting_free state
    #     blocks : List[Block]
    # class Snapshot(TypedDict):
    #     segments : List[Segment]
        
    # for _ in range(20):
    #     x, y = next(data)
    #     x = x.to('cuda')
    #     logits = model(x)
    #     loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    #     # Backward pass
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    

    # return
    
    # model = torch.compile(model, mode='reduce-overhead', fullgraph=True, dynamic=True)
    torch.cuda.reset_max_memory_allocated(0)
    max_iters: int = trainer.get_iters()
    trainer.config.training.max_iters = max_iters
    trainer.model.train()
    train_iter = itertools.cycle(trainer.train_data)
    
    # warmup
    x, y = next(train_iter)
    x, y = x.to(trainer.device), y.to(trainer.device)

    # Forward pass
    with torch.autocast(device_type=trainer.device.type, dtype=trainer.dtype):
        logits = trainer.model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    del x, y, logits  # Free memory
    
    # Backward pass
    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,  # Profile CPU activities
            torch.profiler.ProfilerActivity.CUDA,  # Profile CUDA activities
        ],
        
        profile_memory = True,  # Enable memory profiling
        with_stack=True,  # Include stack traces in the profiling results
        record_shapes=True,
        # Define a schedule for the profiler
        schedule=torch.profiler.schedule(
            wait=0,      # Wait for 1 iteration before starting to profile
            warmup=0,    # Warm up for 4 iterations to stabilize performance
            active=1,    # Profile for 6 active iterations
            repeat=1,    # Repeat the profiling schedule once
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_output'),
        
    ) as p:
        # optimizer = AdamW(
        #     model.parameters(),
        #     lr=config['optimizer']['lr'],
        #     betas=config['optimizer']['betas'],
        #     eps=config['optimizer']['eps'],
        #     weight_decay=config['optimizer']['weight_decay'],
        # )
        
        # x, y = next(data)
        # x = x.to('cuda')
        # logits = model(x)
        # loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        # # Backward pass
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # p.step()
        for iteration in range(3):
            
            # Update learning rate
            lr = trainer.get_learning_rate(iteration)
            trainer.update_learning_rate(lr, trainer.config.scheduler.use_multiplier)
            
            # Get batch
            x, y = next(train_iter)
            x, y = x.to(trainer.device), y.to(trainer.device)

            # Forward pass
            with torch.autocast(device_type=trainer.device.type, dtype=trainer.dtype):
                logits = trainer.model(x)
                loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            del x, y, logits  # Free memory
            
            # Backward pass
            trainer.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if trainer.config.training.gradient_clip_val > 0:
                gradient_clipping(
                    trainer.model.parameters(), trainer.config.training.gradient_clip_val
                )

            trainer.optimizer.step()
        p.step()
        
    print(f"torch.cuda.memory_allocated(0): {torch.cuda.memory_allocated(0)/ (1024**2)} MiB")
    print(f"torch.cuda.max_memory_allocated(0): {torch.cuda.max_memory_allocated(0)/ (1024**3)} GiB")
            

            # print(optimizer.state_dict())

        # Print a table of the profiling results, sorted by total CUDA time, limited to the top 10 entries
        # print(p.key_averages().table(sort_by="cuda_time_total", row_limit=8))
    print(p.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
    p.export_memory_timeline("profiler_output/memory_timeline.html", device="cuda:0")
    
    snapshot = torch.cuda.memory._snapshot()
    # print(snapshot['segments'])

    from pickle import dump
    dump(snapshot, open('profiler_output/snapshot.pickle', 'wb'))