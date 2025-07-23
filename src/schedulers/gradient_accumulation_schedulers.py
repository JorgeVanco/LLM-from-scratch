def augment_gradient_acc(t: int, max_t: int) -> int:
    x = t / max_t
    if x < 0.55:
        return 1
    elif x < 0.85:
        return 2
    else:
        return 3