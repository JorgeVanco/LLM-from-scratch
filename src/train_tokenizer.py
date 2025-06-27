import argparse
import sys
from pathlib import Path
from src.tokenizer import train_bpe

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a BPE (Byte Pair Encoding) tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to text file for training"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tokenizer",
        help="Directory to save the trained tokenizer files"
    )
    
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
        help="Size of the vocabulary to learn"
    )
    
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="List of special tokens to include in vocabulary"
    )
    
    parser.add_argument(
        "--num-processes",
        type=int,
        default=4,
        help="Number of processes for parallelization"
    )
    
    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Input path '{args.data_path}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not data_path.is_file():
        print(f"Error: Input path '{args.data_path}' is not a file", file=sys.stderr)
        sys.exit(1)
    
    if args.vocab_size <= 0:
        print(f"Error: Vocabulary size must be positive, got {args.vocab_size}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

def main() -> None:
    args = parse_args()
    validate_args(args)
    
    print("="*50)
    print("BPE Tokenizer Training")
    print("="*50)
    print(f"Input file: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Vocabulary size: {args.vocab_size:,}")
    print(f"Special tokens: {args.special_tokens}")
    print("="*50)
    
    try:
        train_bpe(
            data_path=args.data_path,
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
            num_processes=args.num_processes
        )
        print("\nTraining completed successfully!")
        print(f"Tokenizer saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"\nError during training: {e}", file=sys.stderr)
        sys.exit(1)
    
    
if __name__ == "__main__":
    main() 