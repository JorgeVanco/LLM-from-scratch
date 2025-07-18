import math


def learning_rate_cosine(t: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if t < warmup_iters:
        learning_rate = t * max_learning_rate / warmup_iters
    elif warmup_iters <= t <= cosine_cycle_iters:
        learning_rate = min_learning_rate + 0.5 * (1 + math.cos((t - warmup_iters)/(cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        learning_rate = min_learning_rate
        
    return learning_rate


def learning_rate_warmup_stable_decay(t: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, stable_iters: int, decay_iters: int) -> float:
    if t < warmup_iters:
        learning_rate = t * max_learning_rate / warmup_iters
    elif warmup_iters <= t <= stable_iters:
        learning_rate = max_learning_rate
    elif t <= decay_iters:
        learning_rate = max_learning_rate + (t - stable_iters) * (min_learning_rate - max_learning_rate) / (decay_iters - stable_iters)
    else:
        learning_rate = min_learning_rate
    return learning_rate