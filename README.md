'''bash
python -m src.tokenize_dataset --dataset-path=data/TinyStoriesV2-GPT4-valid.txt --output-path=data/tokenized/TinyStoriesV2-GPT4/10000/valid.npy --tokenizer-dir=tokenizer/tiny-stories/10000 --num-processes=1

python -m src.tokenize_dataset --dataset-path=data/TinyStoriesV2-GPT4-train.txt --output-path=data/tokenized/TinyStoriesV2-GPT4/10000/train.npy --tokenizer-dir=tokenizer/tiny-stories/10000 --queue-size=5000 --num-processes=1

python -m src.tokenize_dataset --dataset-path=data/TinyStoriesV2-GPT4-train.txt --output-path=data/tokenized/TinyStoriesV2-GPT4/32000/train.npy --tokenizer-dir=tokenizer/tiny-stories/32000 --queue-size=5000 --num-processes=1

python -m src.tokenize_dataset --dataset-path=data/TinyStoriesV2-GPT4-valid.txt --output-path=data/tokenized/TinyStoriesV2-GPT4/32000/valid.npy --tokenizer-dir=tokenizer/tiny-stories/32000 --num-processes=1

uv run -m src.train_tokenizer --data-path=data/owt_train.txt --output-dir=data/tokenized/owt/32000/train.npy --vocab-size=32000

uv run -m src.train_tokenizer --data-path=data/owt_train.txt --output-dir=data/tokenized/owt/10000/train.npy --vocab-size=10000

python -m src.tokenize_dataset --dataset-path=data/owt_train.txt --output-path=data/tokenized/owt/32000/train.npy --tokenizer-dir=tokenizer/owt/32000 --num-processes=1
python -m src.tokenize_dataset --dataset-path=data/owt_valid.txt --output-path=data/tokenized/owt/32000/valid.npy --tokenizer-dir=tokenizer/owt/32000 --num-processes=1

python -m src.tokenize_dataset --dataset-path=data/owt_train.txt --output-path=data/tokenized/owt/10000/train.npy --tokenizer-dir=tokenizer/owt/10000 --num-processes=1
python -m src.tokenize_dataset --dataset-path=data/owt_valid.txt --output-path=data/tokenized/owt/10000/valid.npy --tokenizer-dir=tokenizer/owt/10000 --num-processes=1

python -m src.tokenize_dataset --dataset-path=data/TinyStoriesV2-GPT4-train.txt --output-path=data/tokenized/TinyStoriesV2-GPT4/10000/train.npy --tokenizer-dir=tokenizer/tiny-stories/10000 --num-processes=1
'''