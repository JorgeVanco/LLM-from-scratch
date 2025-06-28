import numpy as np
import argparse
import sys
from pathlib import Path
import os
from src.tokenizer import Tokenizer, load_tokenizer
from tqdm import tqdm

def tokenize_dataset(tokenizer: Tokenizer, dataset_path: str, output_path: str, chunk_size: int = 1_000_000) -> None:
    buffer = []
    total_tokens = 0

    with open(dataset_path, "r", encoding="utf-8") as file, open(output_path, "wb") as bin_file:
        for token in tqdm(tokenizer.encode_iterable(file), desc="Tokenizing", unit="tokens"):
            buffer.append(token)
            if len(buffer) >= chunk_size:
                arr = np.array(buffer, dtype=np.uint16)  # use np.uint32 if your vocab is large
                arr.tofile(bin_file)
                total_tokens += len(buffer)
                buffer = []

        # Write remaining tokens
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            arr.tofile(bin_file)
            total_tokens += len(buffer)
    
    print(f"Total tokens written: {total_tokens}")
    
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a BPE (Byte Pair Encoding) tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset file to tokenize"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the tokenized output"
    )
    
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        required=True,
        help="Directory containing the tokenizer files (vocab.json and merges.txt)"
    )
    
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="List of special tokens in vocabulary"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of tokens to process in each chunk"
    )

    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> None:
    data_path = Path(args.dataset_path)
    if not data_path.exists():
        print(f"Error: Input path '{args.dataset_path}' does not exist", file=sys.stderr)
        sys.exit(1)
        
    if not data_path.is_file():
        print(f"Error: Input path '{args.dataset_path}' is not a file", file=sys.stderr)
        sys.exit(1)
        
    if args.chunk_size <= 0:
        print("Error: Chunk size must be a positive integer", file=sys.stderr)
        sys.exit(1)
        
    if not args.tokenizer_dir:
        print("Error: Tokenizer directory must be specified", file=sys.stderr)
        sys.exit(1)
        
    # Create output directory if it doesn't exist
    output_dir = Path(os.path.dirname(args.output_path))
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    
    print("="*50)
    print("Tokenizing Dataset")
    print("="*50)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output path: {args.output_path}")
    print(f"Tokenizer directory: {args.tokenizer_dir}")
    print(f"Special tokens: {args.special_tokens}")
    print(f"Chunk size: {args.chunk_size}")
    print("="*50)
    
    tokenizer: Tokenizer = load_tokenizer(args.tokenizer_dir, args.special_tokens)

    dataset_path = args.dataset_path
    output_path = args.output_path

    try:
        tokenize_dataset(tokenizer, dataset_path, output_path, args.chunk_size)
        read = np.memmap(output_path, dtype=np.uint16)
        print(f"Tokenized dataset saved to {output_path}")
        print(f"Total tokens in the file: {len(read)}")
        print(f"First 10 tokens: {read[:10]}")
        print(f"Last 10 tokens: {read[-10:]}")
        
    except Exception as e:
        print(f"Error during tokenization: {e}", file=sys.stderr)
        sys.exit(1)

    
if __name__ == "__main__":
    main()