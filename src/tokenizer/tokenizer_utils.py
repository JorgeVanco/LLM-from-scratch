import os
import json
from typing import BinaryIO
import regex as re
from collections import defaultdict

PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize_chunk(args) -> list[tuple[bytes]]:
    input_path: str
    start: int
    end: int
    special_tokens: list[str]
    
    input_path, start, end, special_tokens = args
    
    pretoken_counts = defaultdict(int)
    
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")

        if special_tokens:
            text_segments = re.split("|".join(re.escape(t) for t in special_tokens), chunk)
        else:
            text_segments = [chunk]

        for segment in text_segments:
            # if segment:
            pretokenized_segment = re.finditer(PATTERN, segment)
            for pretoken_match in pretokenized_segment:
                pretoken = tuple(bytes([b]) for b in pretoken_match.group().encode("utf-8"))
                pretoken_counts[pretoken] += 1
            
    return pretoken_counts


def save_bpe_vocab(output_dir: str, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "merges.txt"), "w", encoding="utf-8") as file:
        for a, b in merges:
            a_str = bytes_to_escaped_str(a)
            b_str = bytes_to_escaped_str(b)
            file.write(f"{a_str} {b_str}\n")
    
    # Save vocab.json
    vocab_dict = {}
    for token_id, token_bytes in vocab.items():
        # Convert bytes to string representation for JSON
        token_str = bytes_to_escaped_str(token_bytes)
        
        vocab_dict[token_str] = token_id
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as file:
        json.dump(vocab_dict, file, ensure_ascii=False, indent=2)

    return vocab_dict, merges


def bytes_to_escaped_str(b: bytes) -> str:
        return ''.join(f'\\x{byte:02x}' if byte < 32 or byte > 126 else chr(byte) for byte in b).replace(' ', 'Ġ')
    

def parse_escaped_str(token_str: str) -> bytes:
    
    # Convert back from Ġ to space
    token_str = token_str.replace('Ġ', ' ')
    
    # Handle byte escape sequences
    if '\\x' in token_str:
        # Parse hex escape sequences
        token_bytes = bytearray()
        i = 0
        while i < len(token_str):
            if i < len(token_str) - 3 and token_str[i:i+2] == '\\x':
                hex_str = token_str[i+2:i+4]
                token_bytes.append(int(hex_str, 16))
                i += 4
            else:
                token_bytes.append(ord(token_str[i]))
                i += 1
        return bytes(token_bytes)
    else:
        # Regular UTF-8 encoding
        return token_str.encode('utf-8')
    