curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
bash ./scripts/download_data.sh
uv sync
uv run -m src.tokenize_dataset \
    --dataset-path=data/TinyStoriesV2-GPT4-train.txt \
    --output-path=data/tokenized/TinyStoriesV2-GPT4/32000/train.npy \
    --tokenizer-dir=tokenizer/tiny-stories/32000 \
    --queue-size=5000 \
    --num-processes=8

uv run -m src.train --config configs/baseline.yaml