from collections import defaultdict
from .tokenizer_utils import find_chunk_boundaries, pretokenize_chunk, save_bpe_vocab
import multiprocessing
from tqdm import tqdm

class BPE:
    def __init__(self) -> None:
        self.PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
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
            print(f"No specific number of processes provided. Using {num_processes} processes for parallel pre-tokenization.")

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
        
        num_merges: int = vocab_size - current_vocab_size - len(special_tokens)
        
        # Initial count
        counts = self._get_counts(pretoken_counts)
        
        for _ in tqdm(range(num_merges), desc="Merges"):
            
            max_pair = self._get_max_pair(counts)
            
            vocab[current_vocab_size] = b"".join(max_pair)
            merges.append(max_pair)
            
            current_vocab_size += 1
            
            if current_vocab_size < vocab_size:
                self._apply_merges(pretoken_counts, max_pair, counts)
        
        # Add special tokens to the vocabulary
        for token in special_tokens:
            if token not in vocab.values():
                vocab[current_vocab_size] = token.encode("utf-8")
                current_vocab_size += 1
        
        return vocab, merges

    
def train_bpe(data_path: str, output_dir: str, vocab_size: int, special_tokens: list[str], num_processes: int = 4) -> None:
    bpe = BPE()
    vocab, merges = bpe.train(data_path, vocab_size, special_tokens, num_processes)
    save_bpe_vocab(output_dir, vocab, merges)
    
if __name__ == "__main__":
    import time
    start_time = time.time()
    train_bpe("data/tinystories_sample_5M.txt", "tokenizer/tiny-stories-sample/1000", 1000, ["<|endoftext|>"])
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    # bpe = BPE()
    # import time
    # import cProfile
    # start_time = time.time()
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