from typing import Iterable, Iterator
import regex as re
import json

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

        return {int(k): v.encode("utf-8") for k, v in vocab.items()}
    
    @staticmethod
    def _load_merges(merges_filepath: str) -> list[tuple[bytes, bytes]]:
        with open(merges_filepath, "r", encoding="utf-8") as file:
            merges = [tuple(line.rstrip().split(" ")) for line in file.readlines()]
        return [(a.encode("utf-8"), b.encode("utf-8")) for a, b in merges]
    
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
        pass
    
    def decode(self, ids: list[int]) -> str:
        decoded_bytes = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return decoded_bytes.decode("utf-8", errors="replace")