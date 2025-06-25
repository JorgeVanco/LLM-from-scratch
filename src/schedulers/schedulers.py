import math


def learning_rate_cosine(t, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters) -> float:
    if t < warmup_iters:
        learning_rate = t * max_learning_rate / warmup_iters
    elif warmup_iters <= t <= cosine_cycle_iters:
        learning_rate = min_learning_rate + 0.5 * (1 + math.cos((t - warmup_iters)/(cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        learning_rate = min_learning_rate
        
    return learning_rate