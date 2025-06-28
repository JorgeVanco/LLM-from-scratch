import json
import os
import regex as re
from collections import defaultdict, Counter
from .tokenizer_utils import find_chunk_boundaries, pretokenize_chunk, save_bpe_vocab
import multiprocessing as mp
from multiprocessing import Queue, Process, Manager
import time
from tqdm import tqdm
import signal
import atexit

class PersistentWorkerBPE:
    def __init__(self) -> None:
        self.PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.workers = []
        self.task_queues = []
        self.result_queue = None
        self.manager = None
        
    def _get_counts(self, pretoken_counts: defaultdict) -> dict[tuple[bytes], int]:
        counts: defaultdict = defaultdict(int)
        
        for pretoken, count in pretoken_counts.items():
            for i, j in zip(pretoken, pretoken[1:]):
                counts[(i, j)] += count
                
        return counts
        
    def _get_max_pair(self, counts: dict[tuple[bytes], int]) -> tuple[bytes, bytes]:
        return max(counts.items(), key=lambda x: (x[1], x[0]))[0]

    def _apply_merges(self, pretoken_counts: defaultdict, merge: tuple[bytes], pair_counts: dict[tuple[bytes], int]) -> list[bytes]:
        a, b = merge
        merged = a + b
        
        del pair_counts[merge]
        
        for word, word_count in list(pretoken_counts.items()):
            
            if a not in word or b not in word:
                continue
            
            new_word = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and word[j] == a and word[j + 1] == b:
                    new_word.append(merged)
                    
                    # Update counts
                    if j + 1 < len(word) - 1:
                        pair_counts[(b, word[j + 2])] -= word_count
                        pair_counts[(merged, word[j + 2])] += word_count 
                        
                    if j >= 1:
                        pair_counts[(word[j - 1], a)] -= word_count
                        pair_counts[(word[j - 1], merged)] += word_count
                    
                    j += 2  # Skip both merged bytes
                else:
                    new_word.append(word[j])
                    j += 1
            
            del pretoken_counts[word]
            if len(tuple(new_word)) > 1:
                pretoken_counts[tuple(new_word)] = word_count
            
    
    def _get_pretoken_counts_parallel(self, data_path: str, special_tokens: list[str], num_processes: int = None):
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()
            
        with open(data_path, "rb") as file:
            chunk_boundaries = find_chunk_boundaries(file, num_processes, "<|endoftext|>".encode("utf-8"))
            
        # Prepare arguments for parallel processing
        chunk_args = []
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            chunk_args.append((data_path, start, end, special_tokens))
            
        # Process chunks in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            chunk_results = pool.map(pretokenize_chunk, chunk_args)
            
        # Combine results
        total_counts = defaultdict(int)
        for chunk_count in chunk_results:
            for pretoken, count in chunk_count.items():
                total_counts[pretoken] += count
        
        return total_counts
    
    def train(self, data_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        
        assert vocab_size > 256, "Vocabulary size must be greater than 256"
        print("Starting pre-tokenization...")
        pretoken_counts = self._get_pretoken_counts_parallel(data_path, special_tokens, num_processes)

        vocab: dict[int,bytes] = {i: bytes([i]) for i in range(256)}
        current_vocab_size: int = len(vocab)
        merges: list[tuple[bytes, bytes]] = list()
        
        # Create balanced chunks for workers
        total_words = len(byte_text)
        chunk_size = max(1, total_words // num_processes)
        byte_text_chunks = [
            byte_text[i:i + chunk_size] 
            for i in range(0, total_words, chunk_size)
        ]
        
        # Initial count
        counts = self._get_counts(pretoken_counts)
        
        for _ in tqdm(range(num_merges), desc="Merges"):
            
            max_pair = self._get_max_pair(counts)
            
            vocab[current_vocab_size] = b"".join(max_pair)
            merges.append(max_pair)
            
            current_vocab_size += 1
            
            if current_vocab_size < vocab_size:
                self._apply_merges(pretoken_counts, max_pair, counts)
        
        try:
            # Initialize vocabulary and merges
            vocab = {i: bytes([i]) for i in range(256)}
            current_vocab_size = len(vocab)
            merges = []
            
            num_merges = vocab_size - current_vocab_size - len(special_tokens)
            
            print("Computing initial counts...")
            counts = self._collect_counts(num_processes)
            
            print(f"Starting {num_merges} merges...")
            
            # Track performance
            merge_times = []
            
            for merge_idx in tqdm(range(num_merges), desc="BPE Merges"):
                merge_start = time.time()
                
                # Find most frequent pair
                if not counts:
                    print(f"No pairs remaining at merge {merge_idx}")
                    break
                
                max_pair = max(counts.items(), key=lambda x: (x[1], x[0]))[0]
                
                if counts[max_pair] < 2:
                    print(f"No frequent pairs (count < 2) at merge {merge_idx}")
                    break
                
                # Add to vocabulary
                vocab[current_vocab_size] = b"".join(max_pair)
                merges.append(max_pair)
                current_vocab_size += 1
                
                # Apply merge using persistent workers
                count_changes, words_changed = self._apply_merge_to_workers(max_pair, num_processes)
                
                # Update global counts
                for pair, change in count_changes.items():
                    if change != 0:
                        counts[pair] = counts.get(pair, 0) + change
                        if counts[pair] <= 0:
                            counts.pop(pair, None)
                
                merge_time = time.time() - merge_start
                merge_times.append(merge_time)
                
                # Progress reporting
                if merge_idx % 100 == 0 and merge_idx > 0:
                    avg_time = sum(merge_times[-100:]) / min(100, len(merge_times))
                    total_words, total_tokens = self._get_worker_stats(num_processes)
                    print(f"Merge {merge_idx}: {words_changed} words changed, "
                          f"avg time/merge: {avg_time:.4f}s, "
                          f"compression: {total_tokens/total_words:.2f} tokens/word")
            
            # Add special tokens
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in vocab.values():
                    vocab[current_vocab_size] = token_bytes
                    current_vocab_size += 1
            
            print(f"Training completed. Final vocab size: {len(vocab)}")
            
            if merge_times:
                print(f"Average merge time: {sum(merge_times)/len(merge_times):.4f}s")
                print(f"Total merge time: {sum(merge_times):.2f}s")
            
            return vocab, merges
            
        finally:
            # Cleanup workers
            self._cleanup_workers()

def train_bpe(data_path: str, output_dir: str, vocab_size: int, special_tokens: list[str], num_processes: int = 4):
    """Train BPE using persistent worker processes"""
    
    bpe = PersistentWorkerBPE()
    
    try:
        vocab, merges = bpe.train_with_persistent_workers(data_path, vocab_size, special_tokens, num_processes)
        save_bpe_vocab(output_dir, vocab, merges)
        return vocab, merges
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        bpe._cleanup_workers()
        raise
    except Exception as e:
        print(f"Training failed: {e}")
        bpe._cleanup_workers()
        raise

if __name__ == "__main__":
    import time
    
    # Performance test with persistent workers
    start_time = time.time()
    train_bpe("data/tinystories_sample_5M.txt", "tokenizer/tiny-stories-sample/1000", 1000, ["<|endoftext|>"])
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    # bpe = BPE()
    # import time
    # import cProfile
    # with cProfile.Profile() as pr:
    #     pr.enable()
    #     # Train the BPE tokenizer on a sample corpus
    #     # Adjust the path to your corpus file as needed
    #     # "data/tinystories_sample_5M.txt"
    #     vocab, merges = bpe.train("data/tinystories_sample_5M.txt", 1000, ["<|endoftext|>"])
    # end_time = time.time()
    # pr.disable()
    # pr.print_stats(sort='time')
    # print(f"Training completed in {end_time - start_time:.2f} seconds")
    # # Print the vocabulary and merges
    # # print("Vocabulary:", vocab)
    # # print("Merges:", merges)