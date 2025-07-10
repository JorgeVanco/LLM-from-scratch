# Optimizers

The implementation includes custom optimizers designed for training large language models. Both optimizers support proper weight decay and are optimized for transformer architectures.

## Table of Contents
- [SGD with Learning Rate Scaling](#sgd-with-learning-rate-scaling)
- [AdamW Implementation](#adamw-implementation)

## SGD with Learning Rate Scaling

A custom SGD implementation with learning rate scaling that decreases the learning rate by `1/sqrt(t+1)` where `t` is the iteration number.

### Features
- **Adaptive Learning Rate**: Automatically scales learning rate based on iteration number
- **Memory Efficient**: Minimal state storage (only iteration counter)
- **Stable Training**: Square root scaling prevents exploding gradients in early training

### Implementation Details

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

## AdamW Implementation

A complete AdamW optimizer implementation with bias correction and decoupled weight decay.

### Features
- **Bias Correction**: Proper bias correction for momentum terms
- **Decoupled Weight Decay**: Weight decay applied separately from gradient updates
- **Stable Updates**: Epsilon parameter prevents division by zero
- **Configurable Betas**: Customizable momentum parameters

### Implementation Details

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

### AdamW Algorithm

The optimizer implements the AdamW algorithm with proper bias correction:

1. **Momentum Update**: $m_t = β₁ * m_{t-1} + (1 - β₁) * g_t$
2. **Variance Update**: $v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²$
3. **Bias Correction**: $m̂_t = m_t / (1 - β₁ᵗ)$, $v̂_t = v_t / (1 - β₂ᵗ)$
4. **Parameter Update**: $θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε) - lr * λ * θ_{t-1}$

Where $λ$ is the weight decay coefficient and weight decay is applied directly to parameters.