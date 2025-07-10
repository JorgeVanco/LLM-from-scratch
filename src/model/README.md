# Model Architecture

The model implementation provides a complete transformer architecture with modern design choices and flexible configuration options. The architecture follows contemporary LLM design patterns with support for different normalization schemes and activation functions.

## Table of Contents

- [Transformer Implementation](#transformer-implementation)
- [Components](#components)
- [Model Configuration](#model-configuration)
- [Architecture Examples](#architecture-examples)
- [Hyperparameter Sweeps, Ablations and Architecture modifications](#hyperparameter-sweeps-ablations-and-architecture-modifications)

## Transformer Implementation

The `TransformerLM` class implements a decoder-only transformer language model with the following key features:

- **RMSNorm**: Root Mean Square Layer Normalization for improved training stability
- **Rotary Position Embedding (RoPE)**: Relative position encoding that maintains better length extrapolation
- **Multi-Head Self-Attention**: Standard scaled dot-product attention with causal masking
- **SwiGLU/SiLU FFN**: Choice between SwiGLU and SiLU activation functions in feed-forward networks
- **Flexible Normalization**: Support for pre-norm, post-norm, or no normalization configurations

## Components

### Core Modules

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

### Attention Mechanism

**Multi-Head Self-Attention**
- Scaled dot-product attention with causal masking
- Configurable number of attention heads
- Optional RoPE integration for position-aware attention
- Efficient implementation using `einops` for tensor reshaping

**Attention Features**
- Causal masking for autoregressive generation
- Proper scaling by `sqrt(d_k)` for attention weights
- Support for different head dimensions

### Feed-Forward Networks

**SwiGLU**
- Gated Linear Unit with SiLU activation
- Recommended hidden dimension: `d_ff = 8/3 * d_model` (rounded to multiple of 64)
- Two linear projections with element-wise gating

**SiLU FFN**
- Simple feed-forward network with SiLU activation
- Recommended hidden dimension: `d_ff = 4 * d_model`
- Single gated transformation

### Transformer Block

**Flexible Architecture**
- Configurable normalization placement (pre-norm, post-norm, or none)
- Residual connections around attention and FFN layers
- Choice between SwiGLU and SiLU feed-forward networks

## Model Configuration

### Basic Configuration

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

### Advanced Configuration

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

## Architecture Examples

### GPT-2 Style Model

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

### Modern LLM Configuration

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

### Model Usage

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



## Hyperparameter Sweeps, Ablations and Architecture modifications

### Learning Rate Sweep
[Weights and biases report](https://api.wandb.ai/links/jorgev/0lxnzp8x)
![imagen](https://github.com/user-attachments/assets/e4afa957-7428-4fb9-b908-a285770597e7)
![imagen](https://github.com/user-attachments/assets/9d50d2d2-f411-47a7-9910-36b7f5d2f823)


### Batch Size Sweep
[Weights and biases report](https://api.wandb.ai/links/jorgev/mt2xp56c)
![imagen](https://github.com/user-attachments/assets/2a99d917-f510-41ba-9b3d-245f09f81bf4)
![imagen](https://github.com/user-attachments/assets/476021ca-1ede-4049-9884-9f51da0a0912)

### Ablation 1: Layer Normalization
Comparison between Pre-Norm, Post-Norm and No Norm.
[Weights and biases report](https://api.wandb.ai/links/jorgev/fkc6udf9)
![imagen](https://github.com/user-attachments/assets/9777a51a-6ed5-41f3-b954-ab308ecbdea9)

### Ablation 2: Position Embeddings
RoPE vs. NoPE (No Position Embeddings)
[Weights and biases report](https://api.wandb.ai/links/jorgev/k5yi4it7)
![imagen](https://github.com/user-attachments/assets/d4cf9fb3-383a-471c-91ee-7941a882dd23)
![imagen](https://github.com/user-attachments/assets/490d5fa0-2e83-4fe7-b36f-b4ff6170190b)


### Ablation 3: SwiGLU vs. SiLU
[Weights and biases report](https://api.wandb.ai/links/jorgev/xzroxzij)
![imagen](https://github.com/user-attachments/assets/1ef6cf7c-b328-4579-989a-e39735cee348)


