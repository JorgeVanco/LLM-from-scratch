# Data Loading

The data loading system provides memory-efficient dataset handling with support for large-scale training. It uses memory-mapped files to avoid loading entire datasets into memory, making it suitable for training on datasets that exceed available RAM.

## Memory-Mapped Dataset Loading

The core data loading functionality uses `numpy.memmap` to efficiently access tokenized datasets stored as binary files without loading them entirely into memory.

### Key Features

- **Memory Efficiency**: Uses memory-mapped files to access large datasets without full memory loading
- **Random Sampling**: Implements random batch sampling for better training dynamics
- **GPU-Ready**: Automatic tensor creation with device placement for efficient GPU training

### Basic Usage

```python
from src.data_loading import load_dataset

# Load dataset with memory mapping
dataloader = load_dataset(
    dataset_path="data/tokenized/owt/32000/train.npy",
    batch_size=32,
    context_length=1024
)
```
