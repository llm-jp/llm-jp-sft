# LLM-jp SFT

This repository contains the code for supervised fine-tuning.

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

During the training phase, the loss is computed only on tokens after the `### 回答：` segment. In this case, the loss will be computed on "富士山".

## Training

This is an example of training a model on the sample dataset.

```bash
python train.py \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler cosine \
    --data_files data/example.jsonl \
    --model_name_or_path llm-jp/llm-jp-1.3b-v1.0 \
    --output_dir results/
```

## To Reproduce LLM-jp Models

This section will be updated upon all the resources are publicly available.
