import os
from typing import BinaryIO
import regex as re

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

def pretokenize_chunk(input_path: str, start: int, end: int, special_tokens: list[str], pattern: str) -> list[tuple[bytes]]:
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")

    if special_tokens:
        splitted_text = re.split("|".join(re.escape(t) for t in special_tokens), chunk)
    else:
        splitted_text = [chunk]

    byte_text = []
    for t in splitted_text:
        pretokenized_text = re.finditer(pattern, t)
        byte_text.extend(tuple(c.encode("utf-8") for c in m.group()) for m in pretokenized_text)
    return byte_text

# ## Usage
# num_processes = 4  # Number of processes to use for parallel processing
# with open("data/corpus.en", "rb") as f:
#     boundaries = find_chunk_boundaries(
#         f, num_processes, "<|endoftext|>".encode("utf-8"))
#     print("Chunk boundaries:", boundaries)
#     # The following is a serial implementation, but you can parallelize this 
#     # by sending each start/end pair to a set of processes.
#     for start, end in zip(boundaries[:-1], boundaries[1:]):
#         f.seek(start)
#         chunk = f.read(end - start).decode("utf-8", errors="ignore")
#         # Run pre-tokenization on your chunk and store the counts for each pre-token