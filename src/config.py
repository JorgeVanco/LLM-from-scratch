"""
Training configuration system for LLM from scratch.
This module defines the configuration structure and provides utilities for loading configs.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Literal
import yaml
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    vocab_size: int = 50257
    context_length: int = 1024
    num_layers: int = 12
    d_model: int = 768
    num_heads: int = 12
    d_ff: int = 3072
    qk_norm: bool = True
    rope_theta: float | None = 10000.0
    post_norm: bool | None = False
    ffn_type: Literal["swiglu", "silu"] = "swiglu"


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""
    name: str = "AdamW"  # "AdamW" or "SGD" or "Muon"
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    
    # Muon fields
    muon_lr: float = 0.05
    muon_momentum: float = 0.95
    
    def __post_init__(self) -> None:
        self.lr = float(self.lr)
        self.eps = float(self.eps)
        self.weight_decay = float(self.weight_decay)
        self.muon_lr = float(self.muon_lr)
        self.muon_momentum = float(self.muon_momentum)
        


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduling."""
    use_scheduler: bool = True
    name: str = "warmup_stable_decay" # cosine, warmup_stable_decay or wsd
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 2000
    cosine_cycle_iters: int | None = None
    stable_iters: int | None = None
    decay_iters: int = 4000
    
    use_multiplier: bool = False
    # Multiplier schedulers settings
    warmup_frac: float = 0.0
    cosine_cycle_frac: float = 0.8
    decay_frac: float = 0.2
    
    def __post_init__(self) -> None:
        self.max_learning_rate = float(self.max_learning_rate)
        self.min_learning_rate = float(self.min_learning_rate)
        self.warmup_frac = float(self.warmup_frac)
        self.cosine_cycle_frac = float(self.cosine_cycle_frac)
        self.decay_frac = float(self.decay_frac)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Required fields
    train_data_path: str
    
    batch_size: int = 16
    max_iters: int | None = None
    max_tokens: int | None = None
    eval_interval: int = 1000
    eval_iters: int = 200
    log_interval: int = 100
    save_interval: int = 5000
    gradient_clip_val: float = 1.0
    compile_model: bool = False  # torch.compile
    
    # Data loading
    val_data_path: Optional[str] = ""
    
    # Device settings
    device: str = "auto"  # "auto", "cuda", "cpu"
    dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer."""
    tokenizer_path: str
    special_tokens: list[str] = field(default_factory=lambda: ["<|endoftext|>"])


@dataclass
class LoggingConfig:
    """Configuration for logging and checkpointing."""
    project_name: str = "llm-from-scratch"
    run_name: Optional[str] = None
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    tags: list[str] = field(default_factory=lambda: [])


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Experiment metadata
    experiment_name: str = "baseline"
    description: str = "Baseline training run"
    seed: int = 42


class ConfigManager:
    """Utility class for loading and saving configurations."""
    
    @staticmethod
    def load_config(config_path: str) -> ExperimentConfig:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return ConfigManager._dict_to_config(config_dict)
    
    @staticmethod
    def save_config(config: ExperimentConfig, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = ConfigManager._config_to_dict(config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig."""
        # Create nested configs
        model_config = ModelConfig(**config_dict.get('model', {}))
        optimizer_config = OptimizerConfig(**config_dict.get('optimizer', {}))
        scheduler_config = SchedulerConfig(**config_dict.get('scheduler', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        tokenizer_config = TokenizerConfig(**config_dict.get('tokenizer', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        # Extract top-level experiment fields
        experiment_fields = {
            'experiment_name': config_dict.get('experiment_name', 'baseline'),
            'description': config_dict.get('description', 'Baseline training run'),
            'seed': config_dict.get('seed', 42)
        }
        
        return ExperimentConfig(
            model=model_config,
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            training=training_config,
            tokenizer=tokenizer_config,
            logging=logging_config,
            **experiment_fields
        )
    
    @staticmethod
    def _config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
        """Convert ExperimentConfig to dictionary."""
        return asdict(config)