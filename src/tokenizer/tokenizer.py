from typing import Iterable, Iterator
import regex as re
import json
from .tokenizer_utils import parse_escaped_str

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: None | list = None) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []
        
        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")
            if special_token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = special_token_bytes
                
        self.encoding_vocab = {v: k for k, v in self.vocab.items()}
                
        self.PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @staticmethod
    def _load_vocab(vocab_filepath: str) -> dict[int, bytes]:
        with open(vocab_filepath, "r", encoding="utf-8") as file:
            vocab = json.load(file)

        return Tokenizer._convert_vocab_to_bytes(vocab)
    
    @staticmethod
    def _convert_vocab_to_bytes(vocab: dict[str, int]) -> dict[int, bytes]:
        """
        Convert string-based vocab back to bytes format.
        
        Args:
            vocab: Dictionary mapping token strings to IDs
            
        Returns:
            Dictionary mapping token IDs to byte sequences
        """
        byte_vocab = {}
        for token_str, token_id in vocab.items():     
            byte_vocab[token_id] = parse_escaped_str(token_str)
        
        return byte_vocab
    
    @staticmethod
    def _load_merges(merges_filepath: str) -> list[tuple[bytes, bytes]]:
        merges: list[tuple[str, str]] = []
        with open(merges_filepath, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))
        return Tokenizer._convert_merges_to_bytes(merges)
    
    @staticmethod
    def _convert_merges_to_bytes(merges: list[tuple[str, str]]) -> list[tuple[bytes, bytes]]:
        """
        Convert string-based merges back to bytes format.
        
        Args:
            merges: List of merge rules as (token1_str, token2_str) pairs
            
        Returns:
            List of merge rules as (token1_bytes, token2_bytes) pairs
        """
        byte_merges = []
        for a_str, b_str in merges:
            a_bytes = parse_escaped_str(a_str)
            b_bytes = parse_escaped_str(b_str)
            byte_merges.append((a_bytes, b_bytes))
        
        return byte_merges
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> None:
        return cls(
            vocab=cls._load_vocab(vocab_filepath),
            merges=cls._load_merges(merges_filepath),
            special_tokens=special_tokens
        )
        
    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            splitted_text = re.split("(" + "|".join(re.escape(t) for t in self.special_tokens)+")", text)
        else:
            splitted_text = [text]
            
        tokenized_text: list[int] = []
        for split_text in splitted_text:
            if split_text not in self.special_tokens:
                pretokenized_text = re.finditer(self.PATTERN, split_text)
                for pretoken in pretokenized_text:
                    pretoken = tuple(bytes([c]) for c in pretoken.group().encode("utf-8"))
                    if len(pretoken) >= 2:
                        for merge in self.merges:
                            if merge[0] not in pretoken or merge[1] not in pretoken:
                                continue
                            j = 0
                            while j < len(pretoken) - 1:
                                a, b = pretoken[j], pretoken[j + 1]
                                if (a, b) == merge:
                                    pretoken = tuple(pretoken[:j] + (a + b, ) + pretoken[j+2:])
                                j += 1
                    tokenized_text.extend(self.encoding_vocab[t] for t in pretoken)
            else:
                tokenized_text.append(self.encoding_vocab[split_text.encode("utf-8")])

        return tokenized_text

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            for token in self.encode(line):
                yield token
    
    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return decoded_bytes.decode("utf-8", errors="replace")