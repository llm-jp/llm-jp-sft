# LLM-jp SFT (Supervised Fine-Tuning)

This repository contains the code for supervised fine-tuning of LLM-jp models.

## Requirements

- Python: 3.10.12
- [trl](https://huggingface.co/docs/trl/index): 0.7.2
- [transformers](https://huggingface.co/docs/transformers/index): 4.34.0
- [tokenizers](https://huggingface.co/docs/tokenizers/index): 0.14.0

## Installation

Install the necessary packages using `pip`:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

A sample dataset is provided in `data/`. A training example is structured as follows:

```json
{"text": "### 指示：以下の質問に答えなさい。 ### 質問：日本で一番高い山は？ ### 回答：富士山"}
```

During training, loss calculation is focused on tokens post the "### 回答：" segment. For the above example, the loss will be based on "富士山".

## Training

Here is the command to train a model on the sample dataset.

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

### Development Environment

We fine-tuned models on a node equipped with 8 A100 40GB GPUs.

### Datasets

We used the following datasets for fine-tuning.

- Jaster: A collection of automatically transformed data from the existing Japanese NLP datasets.
- Dolly: A Japanese translation of [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k).
- OpenAssistant: A Japanese translation of [OpenAssistant Conversations Dataset](https://huggingface.co/datasets/OpenAssistant/oasst1).

**NOTE**: The datasets mentioned above are not public as of now. We're in the process of making them accessible. Stay tuned for updates.

### Fine-tuning

#### For the 1.3B model

```bash
accelerate launch --config_file accelerate_config_zero3.yaml \
    train.py \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler cosine \
    --bf16 \
    --max_seq_length 2048 \
    --data_files jamp.json janli.json jcommonsenseqa.json jemhopqa.json jnli.json jsem.json jsick.json jsquad.json jsts.json niilc.json dolly_deepl.json oasst_deepl.json \
    --model_name_or_path llm-jp/llm-jp-1.3b-v1.0 \
    --output_dir results/llm-jp-1.3b-v1.0_jaster-dolly-oasst
```

#### For the 13B model

```bash
accelerate launch --config_file accelerate_config_zero3.yaml \
    train.py \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler cosine \
    --bf16 \
    --max_seq_length 2048 \
    --gradient_checkpointing \
    --data_files jamp.json janli.json jcommonsenseqa.json jemhopqa.json jnli.json jsem.json jsick.json jsquad.json jsts.json niilc.json dolly_deepl.json oasst_deepl.json \
    --model_name_or_path llm-jp/llm-jp-13b-v1.0 \
    --output_dir results/llm-jp-13b-v1.0_jaster-dolly-oasst
```

### Fine-tuning with PEFT

#### For the 1.3B model

```bash
accelerate launch --config_file accelerate_config_zero3.yaml \
    train.py \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler cosine \
    --bf16 \
    --max_seq_length 2048 \
    --data_files jamp.json janli.json jcommonsenseqa.json jemhopqa.json jnli.json jsem.json jsick.json jsquad.json jsts.json niilc.json dolly_deepl.json oasst_deepl.json \
    --use_peft \
    --model_name_or_path llm-jp/llm-jp-1.3b-v1.0 \
    --output_dir results/llm-jp-1.3b-v1.0_jaster-dolly-oasst
```

#### For the 13B model

```bash
accelerate launch --config_file accelerate_config_zero3.yaml \
    train.py \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler cosine \
    --bf16 \
    --max_seq_length 2048 \
    --gradient_checkpointing \
    --data_files jamp.json janli.json jcommonsenseqa.json jemhopqa.json jnli.json jsem.json jsick.json jsquad.json jsts.json niilc.json dolly_deepl.json oasst_deepl.json \
    --use_peft \
    --model_name_or_path llm-jp/llm-jp-13b-v1.0 \
    --output_dir results/llm-jp-13b-v1.0_jaster-dolly-oasst
```
