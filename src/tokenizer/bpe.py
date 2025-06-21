import regex as re
from collections import defaultdict

class BPE:
    def __init__(self) -> None:
        self.PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
    def _get_counts(self, byte_text: list[list[bytes]]) -> dict[tuple[bytes], int]:
        counts: dict[tuple[bytes], int] = dict()
        counts: defaultdict = defaultdict(int)

        for word in byte_text:
            for i, j in zip(word, word[1:]):
                counts[(i, j)] += 1
                
        return counts
        
    def _get_max_pair(self, byte_text: list[list[bytes]]) -> tuple[bytes, bytes]:
        counts: dict[tuple[bytes], int] = self._get_counts(byte_text)
                
        max_pair = max(counts.items(), key=lambda x: (x[1], x[0]))[0]
        return max_pair

    def _apply_merges(self, byte_text: list[list[bytes]], merge: tuple[bytes]) -> list[bytes]:
        for i in range(len(byte_text)):
            if merge[0] not in byte_text[i] or merge[1] not in byte_text[i]:
                continue
            
            j = 0
            while j < len(byte_text[i]) - 1:    
                word = byte_text[i]
                a, b = word[j], word[j + 1]
                if (a, b) == merge:
                    byte_text[i] = tuple(word[:j] + (a + b, ) + word[j+2:])
                j += 1
        return byte_text

    def train(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        
        assert vocab_size > 256, "Vocabulary size must be greater than 256"
        with open(input_path, "r", encoding="utf-8") as file:
            # chunk_boundaries = find_chunk_boundaries(file, desired_num_chunks, split_special_tokens)
            text = file.read()

        pretokenized_text = re.finditer(self.PATTERN, text)
        byte_text: list[list[bytes]] = [tuple(c.encode("utf-8") for c in pretoken.group()) for pretoken in pretokenized_text]

        vocab: dict[int,bytes] = {i: bytes([i]) for i in range(256)}
        current_vocab_size: int = len(vocab)
        merges: list[tuple[bytes, bytes]] = list()
        
        num_merges = vocab_size - current_vocab_size - len(special_tokens)
        
        for _ in range(num_merges):
            
            max_pair = self._get_max_pair(byte_text)
            
            vocab[current_vocab_size] = b"".join(max_pair)
            merges.append(max_pair)
            
            current_vocab_size += 1
            
            if current_vocab_size < vocab_size:
                byte_text = self._apply_merges(byte_text, max_pair)
        
        # Add special tokens to the vocabulary
        for token in special_tokens:
            if token not in vocab.values():
                vocab[current_vocab_size] = token.encode("utf-8")
                current_vocab_size += 1
        
        return vocab, merges
    
if __name__ == "__main__":
    bpe = BPE()
    import time
    import cProfile
    start_time = time.time()
    with cProfile.Profile() as pr:
        pr.enable()
        # Train the BPE tokenizer on a sample corpus
        # Adjust the path to your corpus file as needed
        vocab, merges = bpe.train("data/corpus.en", 500, ["<|endoftext|>"])
    end_time = time.time()
    pr.disable()
    pr.print_stats(sort='time')
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    # Print the vocabulary and merges
    # print("Vocabulary:", vocab)
    # print("Merges:", merges)