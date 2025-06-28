import time
import numpy as np
import argparse
import sys
from pathlib import Path
import os
from src.tokenizer import Tokenizer, load_tokenizer
from tqdm import tqdm
import multiprocessing as mp

def tokenize_dataset(tokenizer: Tokenizer, dataset_path: str, output_path: str, chunk_size: int = 1_000_000) -> None:
    """Tokenizes a dataset file and saves the tokens to a binary file.
    Simple sequential implementation

    Args:
        tokenizer (Tokenizer): The tokenizer to use for encoding.
        dataset_path (str): The path to the dataset file.
        output_path (str): The path to the output binary file.
        chunk_size (int, optional): The number of tokens to process at once. Defaults to 1_000_000.
    """
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
    

def worker_process(input_queue: mp.Queue, output_queue: mp.Queue, tokenizer, worker_id: int):
    """
    Worker process that tokenizes lines and sends results back with line_id
    """
    processed_count = 0
    
    while True:
        try:
            item = input_queue.get(timeout=1)
            if item is None:  # Poison pill to stop worker
                break
            line_id, line_text = item
            
            # Tokenize the line
            tokens = tokenizer.encode(line_text)
            
            # Send result back with line_id
            output_queue.put((line_id, tokens))
            processed_count += 1
            
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            break
    
    print(f"Worker {worker_id} processed {processed_count} lines")

def writer_process(output_queue: mp.Queue, output_path: str, total_lines: int, 
                  buffer_size: int = 1_000_000):
    """
    Writer process that saves tokenized lines in order
    """
    buffer = []
    total_tokens = 0
    next_expected_line_id = 0
    pending_results: dict[int, list] = {}  # Store out-of-order results
    
    with open(output_path, "wb") as bin_file:
        with tqdm(total=total_lines, desc="Writing tokens", position=2) as pbar:
            while next_expected_line_id < total_lines:
                try:
                    # Get result from queue
                    line_id, tokens = output_queue.get(block=True, timeout=1)
                    pending_results[line_id] = tokens
                    # print(f"Received line {line_id} with {len(tokens)} tokens")
                    # Process all consecutive lines we have
                    while next_expected_line_id in pending_results:
                        tokens = pending_results.pop(next_expected_line_id)
                        # print(f"Processing line {next_expected_line_id} with {len(tokens)} tokens")
                        buffer.extend(tokens)
                        
                        # Write buffer when it gets large enough
                        if len(buffer) >= buffer_size:
                            arr = np.array(buffer, dtype=np.uint16)
                            arr.tofile(bin_file)
                            total_tokens += len(buffer)
                            buffer = []
                        
                        next_expected_line_id += 1
                        pbar.update(1)
                        
                        # Optional: flush to disk periodically
                        if next_expected_line_id % 1000 == 0:
                            bin_file.flush()
                
                except:
                    # Timeout - continue checking for results
                    # print("Writer process timeout, checking for results...")
                    continue
        
        # Write remaining tokens in buffer
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            arr.tofile(bin_file)
            total_tokens += len(buffer)
    
    print(f"Writer finished. Total tokens written: {total_tokens}")
    return total_tokens

def tokenize_dataset_parallel(tokenizer, dataset_path: str, output_path: str,
                                 num_workers: int = None, buffer_size: int = 1_000_000,
                                 queue_size: int = 1000) -> None:
    """
    Your proposed approach: ordered queue-based parallel tokenization
    
    Args:
        tokenizer (Tokenizer): The tokenizer to use for encoding.
        dataset_path (str): The path to the dataset file.
        output_path (str): The path to the output binary file.
        num_workers (int, optional): The number of worker processes to use. Defaults to CPU count - 1.
        buffer_size (int, optional): The buffer size for writing tokens. Defaults to 1_000_000.
        queue_size (int, optional): The maximum size of input/output queues. Defaults to 1000.
    """
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)  # Leave one CPU for writer
    
    print(f"Starting tokenization with {num_workers} workers + 1 writer process")
    
    # Count total lines for progress tracking
    print("Counting lines...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    print(f"Total lines to process: {total_lines}")
    
    # Create queues
    input_queue = mp.Queue(maxsize=queue_size)
    output_queue = mp.Queue(maxsize=2 * queue_size)
    
    # Start writer process
    writer = mp.Process(
        target=writer_process,
        args=(output_queue, output_path, total_lines, buffer_size)
    )
    writer.start()
    
    # Start worker processes
    workers = []
    for i in range(num_workers):
        worker = mp.Process(
            target=worker_process,
            args=(input_queue, output_queue, tokenizer, i)
        )
        worker.start()
        workers.append(worker)
    
    # Feed lines to workers
    print("Feeding lines to workers...")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_id, line in enumerate(tqdm(f, total=total_lines, desc="Queuing lines", position=0)):
                input_queue.put((line_id, line))
        
        # Send poison pills to workers
        for _ in range(num_workers):
            input_queue.put(None)
        
        # Wait for all workers to finish
        for worker in workers:
            worker.join()
        
        # Wait for writer to finish
        writer.join()
        
    except KeyboardInterrupt:
        print("Interrupted! Cleaning up...")
        # Terminate all processes
        for worker in workers:
            worker.terminate()
        writer.terminate()
        raise
    
    print("Tokenization completed successfully!")
    
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
        "--num-processes",
        type=int,
        default=None,
        help="Number of processes for parallelization"
    )
    
    parser.add_argument(
        "--queue-size",
        type=int,
        default=2000,
        help="Maximum size of input/output queues"
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
    print(f"Number of processes: {args.num_processes}")
    print(f"Chunk size: {args.chunk_size}")
    print("="*50)
    
    tokenizer: Tokenizer = load_tokenizer(args.tokenizer_dir, args.special_tokens)

    dataset_path = args.dataset_path
    output_path = args.output_path

    try:
        t0 = time.time()
        if args.num_processes == 1:
            tokenize_dataset(tokenizer, dataset_path, output_path, args.chunk_size)
        else:
            tokenize_dataset_parallel(tokenizer, dataset_path, output_path, args.num_processes, args.chunk_size, args.queue_size)
        t1 = time.time()
        read = np.memmap(output_path, dtype=np.uint16)
        print(f"Tokenized dataset saved to {output_path}")
        print(f"Total tokens in the file: {len(read)}")
        print(f"First 10 tokens: {read[:10]}")
        print(f"Last 10 tokens: {read[-10:]}")
        print(f"Tokenization completed in {t1 - t0:.2f} seconds")
    except Exception as e:
        print(f"Error during tokenization: {e}", file=sys.stderr)
        sys.exit(1)

    
if __name__ == "__main__":
    main()