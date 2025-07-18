from functools import partial

from src.schedulers import (
    learning_rate_cosine,
    learning_rate_multiplier_cosine,
    learning_rate_multiplier_warmup_stable_decay,
    learning_rate_warmup_stable_decay,
)
from src.config import ExperimentConfig

def get_scheduler(config: ExperimentConfig) -> partial:
    name: str = config.scheduler.name.lower()
    if name not in ("wsd", "warmup_stable_decay", "cosine"):
        raise ValueError(f"Unknown scheduler: {config.scheduler.name} without multiplier")
    
    if config.scheduler.use_multiplier:
        if name == "cosine":
            return partial(
                learning_rate_multiplier_cosine,
                max_t=config.training.max_iters,
                warmup_frac=config.scheduler.warmup_frac,
                cosine_cycle_frac=config.scheduler.cosine_cycle_frac
            )
        elif name in ("wsd", "warmup_stable_decay"):
            return partial(
                learning_rate_multiplier_warmup_stable_decay,
                max_t=config.training.max_iters,
                warmup_frac=config.scheduler.warmup_frac,
                decay_frac=config.scheduler.decay_frac,
            )
    else:
        if name == "cosine":
            # Automatically extend cosine cycle iters to end of training
            if config.scheduler.cosine_cycle_iters is None:
                config.scheduler.cosine_cycle_iters = config.training.max_iters
            return partial(
                learning_rate_cosine,
                max_learning_rate=config.scheduler.max_learning_rate,
                min_learning_rate=config.scheduler.min_learning_rate,
                warmup_iters=config.scheduler.warmup_iters,
                cosine_cycle_iters=config.scheduler.cosine_cycle_iters,
            )
        elif name in ("wsd", "warmup_stable_decay"):
            return partial(
                learning_rate_warmup_stable_decay,
                max_learning_rate=config.scheduler.max_learning_rate,
                min_learning_rate=config.scheduler.min_learning_rate,
                warmup_iters=config.scheduler.warmup_iters,
                stable_iters=config.scheduler.stable_iters,
                decay_iters=config.scheduler.decay_iters
            )