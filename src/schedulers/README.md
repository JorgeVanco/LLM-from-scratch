# Learning Rate Schedulers

The implementation includes a cosine annealing learning rate scheduler with linear warmup, which is essential for stable training of large language models.

## Table of Contents
- [Cosine Annealing with Warmup](#cosine-annealing-with-warmup)

## Cosine Annealing with Warmup

The `learning_rate_cosine` function implements a three-phase learning rate schedule:

1. **Linear Warmup**: Gradually increase learning rate from 0 to maximum
2. **Cosine Decay**: Smoothly decay learning rate following cosine curve
3. **Constant Minimum**: Maintain minimum learning rate after decay

### Implementation Details

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

### Schedule Phases

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