import time
import random
import argparse
from pathlib import Path
from typing import Any
import numpy as np
import torch
from tqdm import tqdm
import itertools
import json
from dataclasses import asdict

from src.model import TransformerLM
from src.data_loading import load_dataset
from src.optimizers import SGD, AdamW, Muon, MuonWithAuxAdam
from src.schedulers import get_scheduler
from src.tokenizer import Tokenizer, load_tokenizer
from src.utils import (
    cross_entropy,
    gradient_clipping,
    softmax,
    generate_text,
    load_checkpoint,
    save_checkpoint,
)
from src.config import ExperimentConfig, ConfigManager

torch.set_float32_matmul_precision('high')
class Trainer:
    """Main trainer class for the LLM."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.setup_environment()
        self.setup_logging()
        self.load_data()
        self.setup_tokenizer()
        self.setup_model()
        self.setup_optimizer()
        self.setup_scheduler()
        self.current_iter = 0
        self.best_val_loss = float("inf")

    def setup_environment(self) -> None:
        """Setup random seeds and device."""
        # Set random seeds
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)

        # Set device
        if self.config.training.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.training.device)

        # Set dtype
        self.dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = self.dtype_map[self.config.training.dtype]

        print(f"Using device: {self.device}")
        print(f"Using dtype: {self.dtype}")

    def setup_logging(self) -> None:
        """Setup logging and checkpointing directories."""
        self.log_dir = Path(self.config.logging.log_dir) / self.config.experiment_name
        self.checkpoint_dir = (
            Path(self.config.logging.checkpoint_dir) / self.config.experiment_name
        )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup wandb if requested
        if self.config.logging.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=self.config.logging.wandb_project
                    or self.config.logging.project_name,
                    name=self.config.logging.run_name or self.config.experiment_name,
                    tags=self.config.logging.tags,
                    config=self.config.__dict__,
                )
                self.use_wandb = True
            except ImportError:
                print("Warning: wandb not installed, skipping wandb logging")
                self.use_wandb = False
        else:
            self.use_wandb = False

    def load_data(self) -> None:
        """Load training and validation data."""
        print("Loading training data...")
        self.train_data = load_dataset(
            self.config.training.train_data_path,
            self.config.training.batch_size,
            self.config.model.context_length,
            num_workers=0,
        )
        if (
            self.config.training.val_data_path
            and Path(self.config.training.val_data_path).exists()
        ):
            self.val_data = load_dataset(
                self.config.training.val_data_path,
                self.config.training.batch_size,
                self.config.model.context_length,
                num_workers=0,
            )
        else:
            self.val_data = None

        print(f"Training batches: {len(self.train_data):,}")
        if self.val_data is not None:
            print(f"Validation batches: {len(self.val_data):,}")

    def setup_tokenizer(self) -> None:
        """Setup the tokenizer."""
        if self.config.tokenizer.tokenizer_path:
            print("Loading tokenizer from files...")
            self.tokenizer = load_tokenizer(
                tokenizer_dir=self.config.tokenizer.tokenizer_path,
                special_tokens=self.config.tokenizer.special_tokens,
            )
        else:
            print(
                "Warning: No tokenizer files provided, using character-level tokenizer"
            )
            # Create a simple character-level tokenizer
            vocab = {i: bytes([i]) for i in range(256)}
            self.tokenizer = Tokenizer(vocab, [], self.config.tokenizer.special_tokens)

        print(f"Vocabulary size: {len(self.tokenizer.vocab)}")

    def setup_model(self) -> None:
        """Setup the model."""
        print("Setting up model...")

        # Update vocab size if needed
        actual_vocab_size = len(self.tokenizer.vocab)
        if self.config.model.vocab_size != actual_vocab_size:
            print(
                f"Updating vocab size from {self.config.model.vocab_size} to {actual_vocab_size}"
            )
            self.config.model.vocab_size = actual_vocab_size

        self.model = TransformerLM(
            **asdict(self.config.model)
        )

        self.model = self.model.to(self.device)

        # Compile model if requested
        if self.config.training.compile_model:
            print("Compiling model...")
            self.model = torch.compile(self.model)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    def setup_optimizer(self) -> None:
        """Setup optimizer and scheduler."""
        print(f"Setting up {self.config.optimizer.name} optimizer...")

        if self.config.optimizer.name.lower() == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.optimizer.lr,
                betas=self.config.optimizer.betas,
                eps=self.config.optimizer.eps,
                weight_decay=self.config.optimizer.weight_decay,
            )
        elif self.config.optimizer.name.lower() == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=self.config.optimizer.lr)
        elif self.config.optimizer.name.lower() == "muon":          
            hidden_matrix_params = [p for n, p in self.model.layers.named_parameters() if p.ndim >= 2 and "embed" not in n]
            embed_params = [p for n, p in self.model.named_parameters() if "embed" in n]
            scalar_params = [p for p in self.model.parameters() if p.ndim < 2]
            head_params = [self.model.lm_head.weight]

            adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
            adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
            muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
            param_groups = [*adam_groups, muon_group]
            self.optimizer = MuonWithAuxAdam(param_groups)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer.name}")
        
        for group in self.optimizer.param_groups:
            group["initial_lr"] = group["lr"]
            
    def setup_scheduler(self) -> None:
        if not self.config.scheduler.use_scheduler:
            self.scheduler = None
        else:
            print(f"Setting up {self.config.scheduler.name} scheduler...")
            self.scheduler = get_scheduler(self.config)

    def get_learning_rate(self, iteration: int) -> float:
        """Get learning rate for current iteration."""
        if not self.scheduler:
            return self.config.optimizer.lr
        
        return self.scheduler(iteration)

    def update_learning_rate(self, lr: float, use_multiplier: bool) -> None:
        """Update optimizer learning rate."""
        for param_group in self.optimizer.param_groups:
            if use_multiplier:
                param_group["lr"] = param_group["initial_lr"] * lr
            else:
                param_group["lr"] = lr

    @torch.no_grad()
    def estimate_loss(self, use_whole_dataset: bool = False) -> dict[str, float]:
        """Estimate loss on train and validation sets."""
        self.model.eval()
        losses: dict = {}
        total_loss: float = 0.0

        iters = len(self.val_data) if use_whole_dataset else min(len(self.val_data), self.config.training.eval_iters)
        iter_dataloader = iter(self.val_data)
        for _ in tqdm(range(iters), desc="Estimating loss", position=1, leave=False):
            x, y = next(iter_dataloader)
            x, y = x.to(self.device), y.to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                logits = self.model(x)
                loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            total_loss += loss.item()

        losses["val"] = total_loss / iters

        self.model.train()
        return losses

    def save_checkpoint(self, iteration: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{iteration}.pt"

        save_checkpoint(self.model, self.optimizer, iteration, checkpoint_path)
        # checkpoint = {
        #     'iteration': iteration,
        #     'model_state_dict': self.model.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     'config': self.config,
        #     'best_val_loss': self.best_val_loss,
        # }

        # torch.save(checkpoint, checkpoint_path)

        # # Save latest checkpoint
        # latest_path = self.checkpoint_dir / "latest.pt"
        # torch.save(checkpoint, latest_path)

        # # Save best checkpoint
        # if is_best:
        #     best_path = self.checkpoint_dir / "best.pt"
        #     torch.save(checkpoint, best_path)

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        self.current_iter = load_checkpoint(checkpoint_path, self.model, self.optimizer)
        print(f"Loaded checkpoint from iteration {self.current_iter}")

    def log_metrics(self, metrics: dict[str, Any], iteration: int) -> None:
        """Log metrics to wandb."""

        # Log to wandb
        if self.use_wandb:
            import wandb

            wandb.log(metrics, step=iteration)

    def generate_sample(self, prompt: str = "The", max_tokens: int = 100) -> str:
        """Generate a sample text for monitoring."""
        try:
            return generate_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                context_length=self.config.model.context_length,
                max_tokens=max_tokens,
                temperature=0.8,
                top_k=50,
                device=self.device,
            )
        except Exception as e:
            return f"Generation failed: {str(e)}"

    def get_iters(self) -> int:

        assert self.config.training.max_iters is not None or self.config.training.max_tokens is not None, "Must declare max_iters or max_tokens for training"

        if self.config.training.max_iters is None:
            return self.config.training.max_tokens // (self.config.training.batch_size * self.config.model.context_length)
        
        elif self.config.training.max_tokens is None:
            return self.config.training.max_iters
        
        else:
            return min(self.config.training.max_iters, self.config.training.max_tokens // (self.config.training.batch_size * self.config.model.context_length))


    def train(self) -> None:
        """Main training loop."""
        print("Starting training...")

        max_iters: int = self.get_iters()
        self.config.training.max_iters = max_iters

        print(f"Training for {max_iters:,} iterations")
        print(
            f"Total tokens: {max_iters * self.config.training.batch_size * self.config.model.context_length:,}"
        )

        self.model.train()
        start_time = time.time()

        train_iter = itertools.cycle(self.train_data)

        for iteration in tqdm(
            range(self.current_iter, max_iters), position=0, leave=True
        ):
            self.current_iter = iteration

            # Update learning rate
            lr = self.get_learning_rate(iteration)
            self.update_learning_rate(lr, self.config.scheduler.use_multiplier)

            # Get batch
            x, y = next(train_iter)
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                logits = self.model(x)
                loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            del x, y, logits  # Free memory
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.training.gradient_clip_val > 0:
                gradient_clipping(
                    self.model.parameters(), self.config.training.gradient_clip_val
                )

            self.optimizer.step()

            # Logging
            if iteration % self.config.training.log_interval == 0:
                elapsed = time.time() - start_time
                tokens_processed = (
                    (iteration + 1)
                    * self.config.training.batch_size
                    * self.config.model.context_length
                )
                tokens_per_sec = tokens_processed / elapsed

                metrics = {
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                }

                self.log_metrics(metrics, tokens_processed)

                print(
                    f"Iter {iteration:6d} | Loss: {loss.item():.4f} | "
                    f"LR: {lr:.2e} | Tokens/sec: {tokens_per_sec:.0f} | "
                    f"Tokens processed: {tokens_processed:,} | Elapsed: {elapsed:.2f}s"
                )

            # Evaluation
            if (
                iteration % self.config.training.eval_interval == 0
                and iteration > 0
                or iteration == max_iters - 1
            ) and self.val_data is not None:
                losses = self.estimate_loss(use_whole_dataset= False)

                # Check if this is the best model
                is_best = losses["val"] < self.best_val_loss
                if is_best:
                    self.best_val_loss = losses["val"]

                metrics = {
                    "eval/val_loss": losses["val"],
                    "eval/best_val_loss": self.best_val_loss,
                }

                self.log_metrics(metrics, tokens_processed)

                print(
                    f"Iter {iteration:6d} | "
                    f"Val Loss: {losses['val']:.4f} | Best: {self.best_val_loss:.4f} | "
                    f"Tokens/sec: {tokens_per_sec:.0f} | Elapsed: {elapsed:.2f}s"
                )

                # Generate sample text
                sample = self.generate_sample()
                print(f"Sample: {sample}")

                # Save checkpoint if best
                if is_best:
                    self.save_checkpoint(iteration, is_best=True)

            # Save checkpoint
            if iteration % self.config.training.save_interval == 0 and iteration > 0:
                self.save_checkpoint(iteration)

        print("Training completed!")
        self.save_checkpoint(max_iters)

        results = {
            "final_loss": float(losses["val"]),
            "best_val_loss": float(self.best_val_loss),
            "status": "completed"
        }

        if self.use_wandb:
            import wandb
            wandb.finish()

        print(f"RESULTS_JSON:{json.dumps(results)}", flush=True)

def parse_cli_overrides(argv) -> dict:
    overrides = {}
    for arg in argv:
        if "=" not in arg:
            continue
        key, val = arg.split("=", 1)
        if key not in ("config", "resume"):
            try:
                val = eval(val)  # convert to correct type (int, float, bool, etc.)
            except:
                pass
            overrides[key] = val
    return overrides


def apply_override(cfg_obj: Any, key_path: str, value: Any) -> None:
    """Recursively apply override like 'model.d_model=1024'."""
    keys = key_path.split(".")
    current = cfg_obj
    for key in keys[:-1]:
        current = getattr(current, key)
    setattr(current, keys[-1], value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LLM from scratch")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    args, overrides = parser.parse_known_args()
    
    # Load configuration
    config = ConfigManager.load_config(args.config)
    for key_path, value in parse_cli_overrides(overrides).items():
        apply_override(config, key_path, value)

    # Create trainer
    trainer = Trainer(config)

    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
