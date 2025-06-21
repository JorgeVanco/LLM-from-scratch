import regex as re

class BPE:
    def __init__(self) -> None:
        self.PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def _get_lexicographic_greater(self, pair1:tuple[bytes, bytes], pair2:tuple[bytes, bytes]) -> tuple[bytes, bytes]:
        bytes1 = b"".join(pair1)
        bytes2 = b"".join(pair2)
        for i in range(max(len(bytes1), len(bytes2))):
            a = ord(bytes1[i:i+1]) if i < len(bytes1) else -1
            b = ord(bytes2[i:i+1]) if i < len(bytes2) else -1
            
            if a > b:
                return pair1
            elif b > a:
                return pair2
        return pair2
        
        
    def _get_max_pair(self, byte_text: list[list[bytes]]) -> tuple[bytes, bytes]:
        counts: dict[tuple[bytes], int] = dict()
        max_pair = None
        for word in byte_text:
            for i, j in zip(word, word[1:]):
                if (i, j) not in counts:
                    counts[(i, j)] = 0
                counts[(i, j)] += 1
                
                if max_pair is None or counts[(i, j)] >= counts[max_pair]:
                    if max_pair is None or counts[(i, j)] > counts[max_pair]:
                        max_pair = (i, j)
                    else:
                        max_pair = self._get_lexicographic_greater((i,j), max_pair)
                  
        return max_pair

    def _apply_merges(self, byte_text: list[list[bytes]], merge: tuple[bytes]) -> list[bytes]:
        for i, word in enumerate(byte_text):
            for j, (a, b) in enumerate(zip(word, word[1:])):
                if (a, b) == merge:
                    byte_text[i] = tuple(word[:j] + (a + b, ) + word[j+2:])
                    break
        return byte_text

    def train(self, input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        file = open(input_path, "r", encoding="utf-8")
        # chunk_boundaries = find_chunk_boundaries(file, desired_num_chunks, split_special_tokens)
        text = file.read()
        file.close()
        
        pretokenized_text = re.findall(self.PATTERN, text)
        vocab: dict[int,bytes] = {i: bytes(chr(i), "utf-8") for i in range(256)}
        byte_text: list[list[bytes]] = [tuple([bytes(chr(c), "utf-8") for c in bytes(pretoken, "utf-8")]) for pretoken in pretokenized_text]
        current_vocab_size: int = 256
        merges: list[tuple[bytes, bytes]] = list()
        while current_vocab_size < vocab_size:
            
            max_pair = self._get_max_pair(byte_text)
            
            vocab[current_vocab_size] = b"".join(max_pair)
            merges.append(max_pair)
            
            current_vocab_size += 1
            
            if current_vocab_size < vocab_size:
                byte_text = self._apply_merges(byte_text, max_pair)
        
        return vocab, merges
    
if __name__ == "__main__":
    bpe = BPE()
    vocab, merges = bpe.train("data/bpe_test.txt", 259, [])
    print("Vocabulary:", vocab)
    print("Merges:", merges)