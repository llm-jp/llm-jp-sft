# Supervised Fine-tuning for LLMs

## Requirements

- Python: 3.10.12

## Installation

```bash
pip install -r requirements.txt
```

## Training

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
    --data_files <path to jamp.json> <path to janli.json> ... \
    --model_name_or_path <path to HF model> \
    --tokenizer_name_or_path <path to HF tokenizer> \
    --output_dir <path to output directory>
```

## Path

- Dataset: `llm-jp:/model/llm-jp-eval/dataset/`
- Models
  - 13B: `llm-jp:/model/checkpoint_HF/13B/ds_gpt_v101_fattn_nfs_0825_refined-data-gpt_13B_refined_gpu96_node12_lr0.00008533_gbs1536_mbs1_nwk2_zero1_pp8/global_step96657/`
  - 1.3B: `/model/checkpoint_HF/1.3B/ds_gpt_v101_fattn_nfs_0825_refined-data-gpt_1.3B_refined_gpu96_node12_lr0.0001708_gbs1536_mbs4_nwk2_zero1_pp1/global_step96173/`
- Tokenizer: `llm-jp:/model/checkpoint_HF/tokenizer/code10k_en20k_ja30k.ver2.1_hf_fast_saved`
