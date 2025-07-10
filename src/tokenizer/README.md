# Tokenizer

The tokenizer implementation uses Byte Pair Encoding (BPE) with support for special tokens and parallel processing. It follows the GPT-style tokenization approach with regex-based pre-tokenization.

## Table of Contents
  - [Training a Tokenizer](#training-a-tokenizer)
  - [Tokenizing Datasets](#tokenizing-datasets)
  - [Supported Features](#supported-features)
  - [Usage Examples](#usage-examples)

## Training a Tokenizer

Train a new BPE tokenizer on your text data:

```bash
# Train with default settings
uv run -m src.train_tokenizer --data-path=data/your_text_file.txt --output-dir=tokenizer/your_tokenizer --vocab-size=10000

# Train with custom parameters
uv run -m src.train_tokenizer \
  --data-path=data/owt_train.txt \
  --output-dir=tokenizer/owt/32000 \
  --vocab-size=32000 \
  --special-tokens "<|endoftext|>" "<|pad|>" \
  --num-processes=8
```

**Parameters:**
- `--data-path`: Path to the text file for training
- `--output-dir`: Directory to save tokenizer files (`vocab.json` and `merges.txt`)
- `--vocab-size`: Target vocabulary size (default: 10,000)
- `--special-tokens`: List of special tokens to include (default: `["<|endoftext|>"]`)
- `--num-processes`: Number of parallel processes for training (default: auto-detect)

## Tokenizing Datasets

Convert text datasets to tokenized binary format for efficient training:

```bash
# Sequential tokenization (single process)
uv run -m src.tokenize_dataset \
  --dataset-path=data/TinyStoriesV2-GPT4-train.txt \
  --output-path=data/tokenized/TinyStoriesV2-GPT4/10000/train.npy \
  --tokenizer-dir=tokenizer/tiny-stories/10000 \
  --num-processes=1

# Parallel tokenization (multiple processes)
uv run -m src.tokenize_dataset \
  --dataset-path=data/TinyStoriesV2-GPT4-train.txt \
  --output-path=data/tokenized/TinyStoriesV2-GPT4/32000/train.npy \
  --tokenizer-dir=tokenizer/tiny-stories/32000 \
  --queue-size=5000 \
  --num-processes=8
```

**Parameters:**
- `--dataset-path`: Path to the input text file
- `--output-path`: Path for the output tokenized binary file (`.npy` format)
- `--tokenizer-dir`: Directory containing the trained tokenizer files
- `--special-tokens`: Special tokens to use during tokenization
- `--num-processes`: Number of parallel processes (1 for sequential, >1 for parallel)
- `--queue-size`: Size of processing queues for parallel tokenization
- `--chunk-size`: Number of tokens to process in each chunk (default: 1,000,000)

## Supported Features

### BPE Tokenizer Features
- **Regex-based Pre-tokenization**: Uses GPT-style regex pattern for consistent tokenization
- **Parallel Training**: Multi-process support for faster tokenizer training on large datasets
- **Special Token Support**: Handles special tokens like `<|endoftext|>` seamlessly
- **Caching**: Built-in caching mechanism for improved encoding performance
- **Memory Efficient**: Processes large files without loading everything into memory

### Dataset Tokenization Features
- **Sequential Processing**: Single-threaded tokenization for smaller datasets
- **Parallel Processing**: Multi-process tokenization with ordered output for large datasets
- **Progress Tracking**: Real-time progress bars for both training and tokenization
- **Binary Output**: Saves tokens as `uint16` numpy arrays for efficient storage and loading
- **Chunk Processing**: Processes large datasets in manageable chunks

### File Format Support
- **Input**: Plain text files (UTF-8 encoded)
- **Output**: 
  - Tokenizer: `vocab.json` and `merges.txt` files
  - Datasets: NumPy binary files (`.npy`) with `uint16` dtype

## Usage Examples

### Complete Tokenization Pipeline

```bash
# 1. Train tokenizer on OpenWebText
uv run -m src.train_tokenizer \
  --data-path=data/owt_train.txt \
  --output-dir=tokenizer/owt/32000 \
  --vocab-size=32000

# 2. Tokenize training data
uv run -m src.tokenize_dataset \
  --dataset-path=data/owt_train.txt \
  --output-path=data/tokenized/owt/32000/train.npy \
  --tokenizer-dir=tokenizer/owt/32000 \
  --num-processes=8

# 3. Tokenize validation data
uv run -m src.tokenize_dataset \
  --dataset-path=data/owt_valid.txt \
  --output-path=data/tokenized/owt/32000/valid.npy \
  --tokenizer-dir=tokenizer/owt/32000 \
  --num-processes=8
```

### Working with Different Datasets

```bash
# TinyStories dataset with smaller vocabulary
uv run -m src.tokenize_dataset \
  --dataset-path=data/TinyStoriesV2-GPT4-train.txt \
  --output-path=data/tokenized/TinyStoriesV2-GPT4/10000/train.npy \
  --tokenizer-dir=tokenizer/tiny-stories/10000 \
  --num-processes=1

uv run -m src.tokenize_dataset \
  --dataset-path=data/TinyStoriesV2-GPT4-valid.txt \
  --output-path=data/tokenized/TinyStoriesV2-GPT4/10000/valid.npy \
  --tokenizer-dir=tokenizer/tiny-stories/10000 \
  --num-processes=1
```

### Performance Considerations

- **Memory Usage**: The tokenizer processes files in chunks to minimize memory usage
- **Parallel Processing**: Use multiple processes for large datasets (>1GB)
- **Vocabulary Size**: Larger vocabularies (32K) provide better compression but require more memory
- **Queue Size**: Adjust `--queue-size` based on available RAM (larger = faster but more memory)

### Output Verification

After tokenization, the script provides useful statistics:

```
Tokenized dataset saved to data/tokenized/owt/32000/train.npy
Total tokens in the file: 9,035,582,198
First 10 tokens: [15496  11  314  481  655  257  1643  6621  284  262]
Last 10 tokens: [262  1110  286  616  1204  290  262  835  286  262]
Tokenization completed in 1847.32 seconds
```

This information helps verify that tokenization completed successfully and provides insights into the dataset size and token distribution.
