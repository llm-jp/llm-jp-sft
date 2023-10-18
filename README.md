# Supervised Fine-tuning for LLMs

This repository contains the code for fine-tuning language models on labeled datasets.

## Requirements

- Python: 3.10.12
- [trl](https://huggingface.co/docs/trl/index): 0.7.2
- [transformers](https://huggingface.co/docs/transformers/index): 4.34.0
- [tokenizers](https://huggingface.co/docs/tokenizers/index): 0.14.0
- Others: see [requirements.txt](requirements.txt).

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

A sample dataset is provided in `data/`. The dataset is in a jsonl file, and here is its structure:

```json
{"text": "### 指示：以下の質問に答えなさい。 ### 質問：日本で一番高い山は？ ### 回答：富士山"}
```

During the training phase, the loss is computed only on tokens after the "### 回答：" segment. In this case, the loss will be computed on "富士山".

## Training

- For 1.3B models:
```bash
accelerate launch --config_file accelerate_config_zero2.yaml train.py \
    --num_train_epochs 2 \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 80 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler cosine \
    --bf16 \
    --save_steps 50000 \
    --logging_steps 1 \
    --report_to wandb \
    --data_files <path to tuning/jamp.json> <path to tuning/janli.json> ... \
    --eval_data_files <path to develop_small/jamp.json> <path to develop_small/janli.json> ... \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --model_name_or_path <path to HF model> \
    --tokenizer_name_or_path <path to HF tokenizer> \
    --output_dir <path to output directory>
```

- For 13B models:
```bash
accelerate launch --config_file accelerate_config_zero3.yaml train.py \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler cosine \
    --bf16 \
    --save_steps 50000 \
    --logging_steps 1 \
    --report_to wandb \
    --data_files <path to tuning/jamp.json> <path to tuning/janli.json> ... \
    --eval_data_files <path to develop_small/jamp.json> <path to develop_small/janli.json> ... \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --model_name_or_path <path to HF model> \
    --tokenizer_name_or_path <path to HF tokenizer> \
    --output_dir <path to output directory>
```
