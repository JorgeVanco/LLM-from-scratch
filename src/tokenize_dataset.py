from src.tokenizer import Tokenizer
import numpy as np

def tokenize_dataset(tokenizer: Tokenizer, dataset_path: str, output_path: str, chunk_size: int = 1_000_000) -> None:
    buffer = []
    total_tokens = 0

    with open(dataset_path, "r", encoding="utf-8") as file, open(output_path, "wb") as bin_file:
        for token in tokenizer.encode_iterable(file):
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
            
if __name__ == "__main__":

    dataset_path = "data/TinyStoriesV2-GPT4-valid.txt"
    output_path = "data/TinyStoriesV2-GPT4-valid.npy"
    vocab = {i: bytes([i]) for i in range(256)}
    
    tokenizer = Tokenizer(vocab, [], ["<|endoftext|>"]) #Tokenizer.from_files(vocab_filepath, merges_filepath, ["<|endoftext|>"])

    tokenize_dataset(tokenizer, dataset_path, output_path, 10)

    read = np.memmap(output_path, dtype=np.uint16)
    print(read)
    print(len(read))