"""
Training configuration system for LLM from scratch.
This module defines the configuration structure and provides utilities for loading configs.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
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
    rope_theta: float = 10000.0


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""
    name: str = "AdamW"  # "AdamW" or "SGD"
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    
    def __post_init__(self) -> None:
        self.lr = float(self.lr)
        self.eps = float(self.eps)
        self.weight_decay = float(self.weight_decay)
        


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduling."""
    use_scheduler: bool = True
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 2000
    cosine_cycle_iters: int = 50000
    
    def __post_init__(self):
        self.max_learning_rate = float(self.max_learning_rate)
        self.min_learning_rate = float(self.min_learning_rate)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Required fields
    train_data_path: str
    
    batch_size: int = 16
    max_iters: int = 100000
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
        return {
            'experiment_name': config.experiment_name,
            'description': config.description,
            'seed': config.seed,
            'model': {
                'vocab_size': config.model.vocab_size,
                'context_length': config.model.context_length,
                'num_layers': config.model.num_layers,
                'd_model': config.model.d_model,
                'num_heads': config.model.num_heads,
                'd_ff': config.model.d_ff,
                'rope_theta': config.model.rope_theta,
            },
            'optimizer': {
                'name': config.optimizer.name,
                'lr': config.optimizer.lr,
                'betas': list(config.optimizer.betas),
                'eps': config.optimizer.eps,
                'weight_decay': config.optimizer.weight_decay,
            },
            'scheduler': {
                'use_scheduler': config.scheduler.use_scheduler,
                'max_learning_rate': config.scheduler.max_learning_rate,
                'min_learning_rate': config.scheduler.min_learning_rate,
                'warmup_iters': config.scheduler.warmup_iters,
                'cosine_cycle_iters': config.scheduler.cosine_cycle_iters,
            },
            'training': {
                'batch_size': config.training.batch_size,
                'max_iters': config.training.max_iters,
                'eval_interval': config.training.eval_interval,
                'eval_iters': config.training.eval_iters,
                'log_interval': config.training.log_interval,
                'save_interval': config.training.save_interval,
                'gradient_clip_val': config.training.gradient_clip_val,
                'compile_model': config.training.compile_model,
                'train_data_path': config.training.train_data_path,
                'val_data_path': config.training.val_data_path,
                'device': config.training.device,
                'dtype': config.training.dtype,
            },
            'tokenizer': {
                'tokenizer_path': config.tokenizer.tokenizer_path,
                'special_tokens': config.tokenizer.special_tokens,
            },
            'logging': {
                'project_name': config.logging.project_name,
                'run_name': config.logging.run_name,
                'log_dir': config.logging.log_dir,
                'checkpoint_dir': config.logging.checkpoint_dir,
                'use_wandb': config.logging.use_wandb,
                'wandb_project': config.logging.wandb_project,
            }
        }