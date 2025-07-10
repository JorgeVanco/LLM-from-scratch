# LLM from Scratch

A complete implementation of Large Language Model training from scratch, including tokenizer training, model pretraining, and post-training phases.

Based on CS336 Language Modeling from Scratch Stanford course.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Download Data](#download-data)
- [Project Components](#project-components)


## Overview

This project implements a complete pipeline for training Large Language Models from scratch, featuring:

- **Custom BPE Tokenizer**: Byte Pair Encoding tokenizer with parallel processing support
- **Transformer Architecture**: Complete transformer implementation with modern architectural choices
- **Custom Optimizers**: SGD and AdamW implementations with proper weight decay
- **Learning Rate Scheduling**: Cosine annealing with warmup support
- **Model Pretraining**: Full training loop with efficient data loading
- **Post-training**: Fine-tuning and alignment capabilities

## Installation

```bash
# Clone the repository
git clone https://github.com/JorgeVanco/LLM-from-scratch.git
cd LLM-from-scratch

# Install dependencies using uv
uv sync
```

## Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Project Components
- [Tokenizer](src/tokenizer/README.md) - BPE tokenization with parallel processing
- [Model](src/model/README.md) - Transformer architecture implementation
- [Data Loading](src/data_loading/README.md) - Memory-efficient dataset handling
- [Optimizers](src/optimizers/README.md) - SGD and AdamW implementations
- [Schedulers](src/schedulers/README.md) - Learning rate scheduling

