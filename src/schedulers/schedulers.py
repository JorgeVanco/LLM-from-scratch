import math


def learning_rate_cosine(t: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if t < warmup_iters:
        learning_rate = t * max_learning_rate / warmup_iters
    elif warmup_iters <= t <= cosine_cycle_iters:
        learning_rate = min_learning_rate + 0.5 * (1 + math.cos((t - warmup_iters)/(cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        learning_rate = min_learning_rate
        
    return learning_rate


def learning_rate_multiplier_cosine(t: int, max_t: int, warmup_frac: float, cosine_cycle_frac: float) -> float:
    x = t / max_t
    assert 0 <= x < 1
    if x < warmup_frac:
        multiplier = x / warmup_frac
    elif warmup_frac <= x <= warmup_frac + cosine_cycle_frac:
        multiplier = 0.1 + 0.5 * (1 + math.cos((x - warmup_frac)/(cosine_cycle_frac) * math.pi)) * 0.9
    else:
        multiplier = 0.1
        
    return multiplier


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


def learning_rate_multiplier_warmup_stable_decay(t: int, max_t: int, warmup_frac: float, decay_frac: float) -> float:
    x = t / max_t
    assert 0 <= x < 1
    if x < warmup_frac:
        return x / warmup_frac
    elif x < 1 - decay_frac:
        return 1.0
    else:
        w = (1 - x) / decay_frac
        return w * 1.0 + (1 - w) * 0.1