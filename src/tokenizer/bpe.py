from collections import defaultdict
from .tokenizer_utils import find_chunk_boundaries, pretokenize_chunk, save_bpe_vocab
import multiprocessing
from tqdm import tqdm

class BPE:
    def __init__(self) -> None:
        self.PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
    def _get_counts(self, byte_text: list[list[bytes]]) -> dict[tuple[bytes], int]:
        counts: defaultdict = defaultdict(int)

        for word in byte_text:
            for i, j in zip(word, word[1:]):
                counts[(i, j)] += 1
                
        return counts
        
    def _get_max_pair(self, counts: dict[tuple[bytes], int]) -> tuple[bytes, bytes]:
        return max(counts.items(), key=lambda x: (x[1], x[0]))[0]

    def _apply_merges(self, byte_text: list[list[bytes]], merge: tuple[bytes], counts: dict[tuple[bytes], int]) -> list[bytes]:
        a, b = merge
        merged = a + b
        
        del counts[merge]
        
        for i in range(len(byte_text)):
            word = byte_text[i]
            
            if a not in word or b not in word:
                continue
            
            new_word = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and word[j] == a and word[j + 1] == b:
                    new_word.append(merged)
                    
                    # Update counts
                    if j + 1 < len(word) - 1:
                        counts[(b, word[j + 2])] -= 1
                        counts[(merged, word[j + 2])] += 1 
                        
                    if j >= 1:
                        counts[(word[j - 1], a)] -= 1
                        counts[(word[j - 1], merged)] += 1
                    
                    j += 2  # Skip both merged bytes
                else:
                    new_word.append(word[j])
                    j += 1
            
            byte_text[i] = tuple(new_word)
            
        return byte_text
    
    
    def train(self, data_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 4) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        
        assert vocab_size > 256, "Vocabulary size must be greater than 256"
        with open(data_path, "rb") as file:
            chunk_boundaries = find_chunk_boundaries(file, num_processes, "<|endoftext|>".encode("utf-8"))
            
        chunks = list(zip(chunk_boundaries[:-1], chunk_boundaries[1:]))
        args = [(data_path, start, end, special_tokens, self.PATTERN) for start, end in chunks]

        with multiprocessing.Pool(len(args)) as pool:
            pretokenized_texts = pool.starmap(pretokenize_chunk, args)

        byte_text = []
        for pretokenized_text in pretokenized_texts:
            byte_text.extend(pretokenized_text)

        vocab: dict[int,bytes] = {i: bytes([i]) for i in range(256)}
        current_vocab_size: int = len(vocab)
        merges: list[tuple[bytes, bytes]] = list()
        
        num_merges: int = vocab_size - current_vocab_size - len(special_tokens)
        
        # Initial count
        counts = self._get_counts(byte_text)
        
        for _ in tqdm(range(num_merges), desc="Merges"):
            
            max_pair = self._get_max_pair(counts)
            
            vocab[current_vocab_size] = b"".join(max_pair)
            merges.append(max_pair)
            
            current_vocab_size += 1
            
            if current_vocab_size < vocab_size:
                byte_text = self._apply_merges(byte_text, max_pair, counts)
        
        # Add special tokens to the vocabulary
        for token in special_tokens:
            if token not in vocab.values():
                vocab[current_vocab_size] = token.encode("utf-8")
                current_vocab_size += 1
        
        return vocab, merges

    
def train_bpe(data_path: str, output_dir: str, vocab_size: int, special_tokens: list[str]) -> None:
    bpe = BPE()
    vocab, merges = bpe.train(data_path, vocab_size, special_tokens)
    save_bpe_vocab(output_dir, vocab, merges)
    
if __name__ == "__main__":
    import time
    start_time = time.time()
    train_bpe("data/test.txt", "tokenizer/corpus/500", 289, ["<|endoftext|>"])
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
    #     vocab, merges = bpe.train("data/tinystories_sample_5M.txt", 500, ["<|endoftext|>"])
    # end_time = time.time()
    # pr.disable()
    # pr.print_stats(sort='time')
    # print(f"Training completed in {end_time - start_time:.2f} seconds")
    # # Print the vocabulary and merges
    # # print("Vocabulary:", vocab)
    # # print("Merges:", merges)