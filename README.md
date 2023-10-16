# Supervised Fine-tuning for LLMs

## Requirements

- Python: 3.10.12

## Installation

```bash
pip install -r requirements.txt
```

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
accelerate launch --config_file accelerate_config.yaml train.py \
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

## Path

- Dataset
  - train: `llm-jp:/model/llm-jp-eval/dataset/tuning/`
  - eval: `llm-jp:/model/llm-jp-eval/dataset/develop_small/`
- Models
  - 13B: `llm-jp:/model/checkpoint_HF/13B/ds_gpt_v101_fattn_nfs_0825_refined-data-gpt_13B_refined_gpu96_node12_lr0.00008533_gbs1536_mbs1_nwk2_zero1_pp8/global_step96657/`
  - 1.3B: `llm-jp:/model/checkpoint_HF/1.3B/ds_gpt_v101_fattn_nfs_0825_refined-data-gpt_1.3B_refined_gpu96_node12_lr0.0001708_gbs1536_mbs4_nwk2_zero1_pp1/global_step96173/`
- Tokenizer
  - from Hugging Face Hub: `llm-jp/hf-fast-tokenizer-v21b3`
