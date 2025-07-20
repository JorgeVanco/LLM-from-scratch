import argparse

from src.model import TransformerLM
from src.optimizers import AdamW
from src.utils import generate_text
from src.tokenizer import Tokenizer, load_tokenizer
from src.utils import load_checkpoint
from src.config import ConfigManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLM from scratch")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    args = parser.parse_args()
    config = ConfigManager.load_config(args.config)
    model = TransformerLM(**config.model.__dict__)
    load_checkpoint(args.checkpoint_path, model)
    tokenizer: Tokenizer = load_tokenizer(
                tokenizer_dir=config.tokenizer.tokenizer_path,
                special_tokens=config.tokenizer.special_tokens,
            )
    print(generate_text(model, tokenizer, "Lily and Max were playing", config.model.context_length, args.max_tokens))