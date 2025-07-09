# LLM from Scratch

A complete implementation of Large Language Model training from scratch, including tokenizer training, model pretraining, and post-training phases.

Based on CS336 Language Modeling from Scratch Stanford course.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Download Data](#download-data)
- [Tokenizer](#tokenizer)
  - [Training a Tokenizer](#training-a-tokenizer)
  - [Tokenizing Datasets](#tokenizing-datasets)
  - [Supported Features](#supported-features)
  - [Usage Examples](#usage-examples)
- [Model Architecture](#model-architecture)
  - [Transformer Implementation](#transformer-implementation)
  - [Components](#components)
  - [Model Configuration](#model-configuration)
  - [Architecture Examples](#architecture-examples)
- [Optimizers](#optimizers)
  - [SGD with Learning Rate Scaling](#sgd-with-learning-rate-scaling)
  - [AdamW Implementation](#adamw-implementation)
- [Learning Rate Schedulers](#learning-rate-schedulers)
  - [Cosine Annealing with Warmup](#cosine-annealing-with-warmup)
- [Data Loading](#data-loading)

## Overview

This project implements a complete pipeline for training Large Language Models from scratch, featuring:

- **Custom BPE Tokenizer**: Byte Pair Encoding tokenizer with parallel processing support
- **Transformer Architecture**: Complete transformer implementation with modern architectural choices
- **Custom Optimizers**: SGD and AdamW implementations with proper weight decay
- **Learning Rate Scheduling**: Cosine annealing with warmup support
- **Model Pretraining**: Full training loop with efficient data loading
- **Post-training**: Fine-tuning and alignment capabilities

## Installation

```bash
# Clone the repository
git clone https://github.com/JorgeVanco/LLM-from-scratch.git
cd LLM-from-scratch

# Install dependencies using uv
uv sync
```

## Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Tokenizer

The tokenizer implementation uses Byte Pair Encoding (BPE) with support for special tokens and parallel processing. It follows the GPT-style tokenization approach with regex-based pre-tokenization.

### Training a Tokenizer

Train a new BPE tokenizer on your text data:

```bash
# Train with default settings
uv run -m src.train_tokenizer --data-path=data/your_text_file.txt --output-dir=tokenizer/your_tokenizer --vocab-size=10000

# Train with custom parameters
uv run -m src.train_tokenizer \
  --data-path=data/owt_train.txt \
  --output-dir=tokenizer/owt/32000 \
  --vocab-size=32000 \
  --special-tokens "<|endoftext|>" "<|pad|>" \
  --num-processes=8
```

**Parameters:**
- `--data-path`: Path to the text file for training
- `--output-dir`: Directory to save tokenizer files (`vocab.json` and `merges.txt`)
- `--vocab-size`: Target vocabulary size (default: 10,000)
- `--special-tokens`: List of special tokens to include (default: `["<|endoftext|>"]`)
- `--num-processes`: Number of parallel processes for training (default: auto-detect)

### Tokenizing Datasets

Convert text datasets to tokenized binary format for efficient training:

```bash
# Sequential tokenization (single process)
uv run -m src.tokenize_dataset \
  --dataset-path=data/TinyStoriesV2-GPT4-train.txt \
  --output-path=data/tokenized/TinyStoriesV2-GPT4/10000/train.npy \
  --tokenizer-dir=tokenizer/tiny-stories/10000 \
  --num-processes=1

# Parallel tokenization (multiple processes)
uv run -m src.tokenize_dataset \
  --dataset-path=data/TinyStoriesV2-GPT4-train.txt \
  --output-path=data/tokenized/TinyStoriesV2-GPT4/32000/train.npy \
  --tokenizer-dir=tokenizer/tiny-stories/32000 \
  --queue-size=5000 \
  --num-processes=8
```

**Parameters:**
- `--dataset-path`: Path to the input text file
- `--output-path`: Path for the output tokenized binary file (`.npy` format)
- `--tokenizer-dir`: Directory containing the trained tokenizer files
- `--special-tokens`: Special tokens to use during tokenization
- `--num-processes`: Number of parallel processes (1 for sequential, >1 for parallel)
- `--queue-size`: Size of processing queues for parallel tokenization
- `--chunk-size`: Number of tokens to process in each chunk (default: 1,000,000)

### Supported Features

#### BPE Tokenizer Features
- **Regex-based Pre-tokenization**: Uses GPT-style regex pattern for consistent tokenization
- **Parallel Training**: Multi-process support for faster tokenizer training on large datasets
- **Special Token Support**: Handles special tokens like `<|endoftext|>` seamlessly
- **Caching**: Built-in caching mechanism for improved encoding performance
- **Memory Efficient**: Processes large files without loading everything into memory

#### Dataset Tokenization Features
- **Sequential Processing**: Single-threaded tokenization for smaller datasets
- **Parallel Processing**: Multi-process tokenization with ordered output for large datasets
- **Progress Tracking**: Real-time progress bars for both training and tokenization
- **Binary Output**: Saves tokens as `uint16` numpy arrays for efficient storage and loading
- **Chunk Processing**: Processes large datasets in manageable chunks

#### File Format Support
- **Input**: Plain text files (UTF-8 encoded)
- **Output**: 
  - Tokenizer: `vocab.json` and `merges.txt` files
  - Datasets: NumPy binary files (`.npy`) with `uint16` dtype

### Usage Examples

#### Complete Tokenization Pipeline

```bash
# 1. Train tokenizer on OpenWebText
uv run -m src.train_tokenizer \
  --data-path=data/owt_train.txt \
  --output-dir=tokenizer/owt/32000 \
  --vocab-size=32000

# 2. Tokenize training data
uv run -m src.tokenize_dataset \
  --dataset-path=data/owt_train.txt \
  --output-path=data/tokenized/owt/32000/train.npy \
  --tokenizer-dir=tokenizer/owt/32000 \
  --num-processes=8

# 3. Tokenize validation data
uv run -m src.tokenize_dataset \
  --dataset-path=data/owt_valid.txt \
  --output-path=data/tokenized/owt/32000/valid.npy \
  --tokenizer-dir=tokenizer/owt/32000 \
  --num-processes=8
```

#### Working with Different Datasets

```bash
# TinyStories dataset with smaller vocabulary
uv run -m src.tokenize_dataset \
  --dataset-path=data/TinyStoriesV2-GPT4-train.txt \
  --output-path=data/tokenized/TinyStoriesV2-GPT4/10000/train.npy \
  --tokenizer-dir=tokenizer/tiny-stories/10000 \
  --num-processes=1

uv run -m src.tokenize_dataset \
  --dataset-path=data/TinyStoriesV2-GPT4-valid.txt \
  --output-path=data/tokenized/TinyStoriesV2-GPT4/10000/valid.npy \
  --tokenizer-dir=tokenizer/tiny-stories/10000 \
  --num-processes=1
```

#### Performance Considerations

- **Memory Usage**: The tokenizer processes files in chunks to minimize memory usage
- **Parallel Processing**: Use multiple processes for large datasets (>1GB)
- **Vocabulary Size**: Larger vocabularies (32K) provide better compression but require more memory
- **Queue Size**: Adjust `--queue-size` based on available RAM (larger = faster but more memory)

#### Output Verification

After tokenization, the script provides useful statistics:

```
Tokenized dataset saved to data/tokenized/owt/32000/train.npy
Total tokens in the file: 9,035,582,198
First 10 tokens: [15496  11  314  481  655  257  1643  6621  284  262]
Last 10 tokens: [262  1110  286  616  1204  290  262  835  286  262]
Tokenization completed in 1847.32 seconds
```

This information helps verify that tokenization completed successfully and provides insights into the dataset size and token distribution.

## Model Architecture

The model implementation provides a complete transformer architecture with modern design choices and flexible configuration options. The architecture follows contemporary LLM design patterns with support for different normalization schemes and activation functions.

### Transformer Implementation

The `TransformerLM` class implements a decoder-only transformer language model with the following key features:

- **RMSNorm**: Root Mean Square Layer Normalization for improved training stability
- **Rotary Position Embedding (RoPE)**: Relative position encoding that maintains better length extrapolation
- **Multi-Head Self-Attention**: Standard scaled dot-product attention with causal masking
- **SwiGLU/SiLU FFN**: Choice between SwiGLU and SiLU activation functions in feed-forward networks
- **Flexible Normalization**: Support for pre-norm, post-norm, or no normalization configurations

### Components

#### Core Modules

**Linear Layer**
- Custom linear transformation with truncated normal initialization
- Optimized weight initialization: `std = sqrt(2 / (in_features + out_features))`
- Uses `einops` for efficient tensor operations

**Embedding Layer**
- Token embedding with truncated normal initialization
- Standard deviation of 1.0 with clipping at ±3σ

**RMSNorm**
- Root Mean Square Layer Normalization
- More stable than LayerNorm, especially for large models
- Configurable epsilon for numerical stability (default: 1e-5)

**Rotary Position Embedding**
- Implements RoPE for relative position encoding
- Configurable base frequency (theta) parameter
- Supports arbitrary sequence lengths up to maximum context length

#### Attention Mechanism

**Multi-Head Self-Attention**
- Scaled dot-product attention with causal masking
- Configurable number of attention heads
- Optional RoPE integration for position-aware attention
- Efficient implementation using `einops` for tensor reshaping

**Attention Features**
- Causal masking for autoregressive generation
- Proper scaling by `sqrt(d_k)` for attention weights
- Support for different head dimensions

#### Feed-Forward Networks

**SwiGLU**
- Gated Linear Unit with SiLU activation
- Recommended hidden dimension: `d_ff = 8/3 * d_model` (rounded to multiple of 64)
- Two linear projections with element-wise gating

**SiLU FFN**
- Simple feed-forward network with SiLU activation
- Recommended hidden dimension: `d_ff = 4 * d_model`
- Single gated transformation

#### Transformer Block

**Flexible Architecture**
- Configurable normalization placement (pre-norm, post-norm, or none)
- Residual connections around attention and FFN layers
- Choice between SwiGLU and SiLU feed-forward networks

### Model Configuration

#### Basic Configuration

```python
from src.model import TransformerLM

# Small model for experimentation
model = TransformerLM(
    vocab_size=10000,
    context_length=1024,
    num_layers=12,
    d_model=768,
    num_heads=12,
    d_ff=3072,
    rope_theta=10000.0,
    post_norm=False,      # Use pre-norm (default)
    ffn_type="swiglu"     # Use SwiGLU activation
)
```

#### Advanced Configuration

```python
# Larger model with custom settings
model = TransformerLM(
    vocab_size=32000,
    context_length=2048,
    num_layers=24,
    d_model=1024,
    num_heads=16,
    d_ff=4096,
    rope_theta=10000.0,
    post_norm=None,       # No normalization (not recommended haha)
    ffn_type="silu"       # Use SiLU FFN
)
```

**Parameters:**
- `vocab_size`: Size of the vocabulary (number of tokens)
- `context_length`: Maximum sequence length the model can handle
- `num_layers`: Number of transformer blocks
- `d_model`: Model dimensionality (embedding size)
- `num_heads`: Number of attention heads (must divide `d_model`)
- `d_ff`: Feed-forward network hidden dimension ($\frac{8}{3} \cdot$ `d_model` for SwiGLU, $4\cdot$ `d_model` for SiLU)
- `rope_theta`: Base frequency for RoPE (set to `None` to disable RoPE)
- `post_norm`: Normalization placement (`False` for pre-norm, `True` for post-norm, `None` for no norm)
- `ffn_type`: Feed-forward network type (`"swiglu"` or `"silu"`)

### Architecture Examples

#### GPT-2 Style Model

```python
# GPT-2 Small configuration
model = TransformerLM(
    vocab_size=50257,
    context_length=1024,
    num_layers=12,
    d_model=768,
    num_heads=12,
    d_ff=3072,
    rope_theta=None,      # GPT-2 uses learned positional embeddings
    post_norm=False,
    ffn_type="silu"
)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
# Output: Parameters: 124,439,808
```

#### Modern LLM Configuration

```python
# Modern architecture with RoPE and SwiGLU
model = TransformerLM(
    vocab_size=32000,
    context_length=2048,
    num_layers=32,
    d_model=4096,
    num_heads=32,
    d_ff=11008,           # 8/3 * 4096, rounded to multiple of 64
    rope_theta=10000.0,
    post_norm=False,
    ffn_type="swiglu"
)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

#### Model Usage

```python
import torch
from src.model import TransformerLM

# Initialize model
model = TransformerLM(
    vocab_size=10000,
    context_length=512,
    num_layers=6,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    rope_theta=10000.0
)

# Forward pass
batch_size = 4
seq_len = 256
input_ids = torch.randint(0, 10000, (batch_size, seq_len))

# Get logits for next token prediction
logits = model(input_ids)  # Shape: (batch_size, seq_len, vocab_size)

# For training, compute loss against targets
targets = torch.randint(0, 10000, (batch_size, seq_len))
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
```

## Optimizers

The implementation includes custom optimizers designed for training large language models. Both optimizers support proper weight decay and are optimized for transformer architectures.

### SGD with Learning Rate Scaling

A custom SGD implementation with learning rate scaling that decreases the learning rate by `1/sqrt(t+1)` where `t` is the iteration number.

#### Features
- **Adaptive Learning Rate**: Automatically scales learning rate based on iteration number
- **Memory Efficient**: Minimal state storage (only iteration counter)
- **Stable Training**: Square root scaling prevents exploding gradients in early training

#### Implementation Details

```python
from src.optimizers import SGD

# Initialize SGD optimizer
optimizer = SGD(model.parameters(), lr=0.1)

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()  # Applies lr / sqrt(t + 1) scaling automatically
```

The learning rate at step `t` is computed as: `lr_effective = lr / sqrt(t + 1)`

### AdamW Implementation

A complete AdamW optimizer implementation with bias correction and decoupled weight decay.

#### Features
- **Bias Correction**: Proper bias correction for momentum terms
- **Decoupled Weight Decay**: Weight decay applied separately from gradient updates
- **Stable Updates**: Epsilon parameter prevents division by zero
- **Configurable Betas**: Customizable momentum parameters

#### Implementation Details

```python
from src.optimizers import AdamW

# Initialize AdamW with recommended settings for LLM training
optimizer = AdamW(
    model.parameters(),
    lr=6e-4,                    # Peak learning rate
    betas=(0.9, 0.95),          # Momentum parameters
    eps=1e-8,                   # Numerical stability
    weight_decay=0.1            # Weight decay strength
)
```

**Parameters:**
- `lr`: Learning rate (default: 1e-3)
- `betas`: Coefficients for momentum terms (default: (0.9, 0.999))
- `eps`: Epsilon for numerical stability (default: 1e-8)
- `weight_decay`: Weight decay coefficient (default: 0.01)

#### AdamW Algorithm

The optimizer implements the AdamW algorithm with proper bias correction:

1. **Momentum Update**: $m_t = β₁ * m_{t-1} + (1 - β₁) * g_t$
2. **Variance Update**: $v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²$
3. **Bias Correction**: $m̂_t = m_t / (1 - β₁ᵗ)$, $v̂_t = v_t / (1 - β₂ᵗ)$
4. **Parameter Update**: $θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε) - lr * λ * θ_{t-1}$

Where $λ$ is the weight decay coefficient and weight decay is applied directly to parameters.


## Learning Rate Schedulers

The implementation includes a cosine annealing learning rate scheduler with linear warmup, which is essential for stable training of large language models.

### Cosine Annealing with Warmup

The `learning_rate_cosine` function implements a three-phase learning rate schedule:

1. **Linear Warmup**: Gradually increase learning rate from 0 to maximum
2. **Cosine Decay**: Smoothly decay learning rate following cosine curve
3. **Constant Minimum**: Maintain minimum learning rate after decay

#### Implementation Details

```python
from src.schedulers import learning_rate_cosine

# Compute learning rate for step t
lr = learning_rate_cosine(
    t=current_step,
    max_learning_rate=6e-4,
    min_learning_rate=6e-5,
    warmup_iters=2000,
    cosine_cycle_iters=100000
)
```

**Parameters:**
- `t`: Current training step
- `max_learning_rate`: Peak learning rate after warmup
- `min_learning_rate`: Minimum learning rate (maintained after decay)
- `warmup_iters`: Number of steps for linear warmup
- `cosine_cycle_iters`: Total steps for cosine decay (including warmup)

#### Schedule Phases

**Phase 1 - Linear Warmup (0 ≤ t < warmup_iters)**
```
lr = t * max_learning_rate / warmup_iters
```

**Phase 2 - Cosine Decay (warmup_iters ≤ t ≤ cosine_cycle_iters)**
```
progress = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
lr = min_learning_rate + 0.5 * (1 + cos(π * progress)) * (max_learning_rate - min_learning_rate)
```

**Phase 3 - Constant Minimum (t > cosine_cycle_iters)**
```
lr = min_learning_rate
```

## Data Loading

The data loading system provides memory-efficient dataset handling with support for large-scale training. It uses memory-mapped files to avoid loading entire datasets into memory, making it suitable for training on datasets that exceed available RAM.

### Memory-Mapped Dataset Loading

The core data loading functionality uses `numpy.memmap` to efficiently access tokenized datasets stored as binary files without loading them entirely into memory.

#### Key Features

- **Memory Efficiency**: Uses memory-mapped files to access large datasets without full memory loading
- **Random Sampling**: Implements random batch sampling for better training dynamics
- **GPU-Ready**: Automatic tensor creation with device placement for efficient GPU training

#### Basic Usage

```python
from src.data_loading import load_dataset

# Load dataset with memory mapping
dataloader = load_dataset(
    dataset_path="data/tokenized/owt/32000/train.npy",
    batch_size=32,
    context_length=1024
)
```