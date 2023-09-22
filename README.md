# Supervised Fine-tuning for LLMs

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --data_path /data/llmjp0/eval/instruction_dataset/mdx_format --model_name_or_path /data/llmjp0/model_cache_dir/outputs/checkpoint_HF/1.3B/ds_gpt_v101_fattn_nfs_0825_fold-gpt_1.3B_fold09_gpu96_node12_lr2.0e-4_gbs1536_mbs4_nwk1_zero1_pp1/global_step87430/ --tokenizer_name_or_path /model/kodama/tokenizer/code10k_en20k_ja30k.ver2.1_hf_fast_saved --output_dir /model/kiyomaru/tuning/sft/sandbox
```

## Reference

### Pretrained models

- LLM-jp 1.3B: `/data/llmjp0/model_cache_dir/outputs/checkpoint_HF/1.3B/ds_gpt_v101_fattn_nfs_0825_fold-gpt_1.3B_fold09_gpu96_node12_lr2.0e-4_gbs1536_mbs4_nwk1_zero1_pp1/global_step87430/`

### Pretrained tokenizsers

- LLM-jp v2.1: `/model/kodama/tokenizer/code10k_en20k_ja30k.ver2.1_hf_fast/tokenizer.json`

### Datasets

- LLM-jp instructions: `/data/llmjp0/eval/instruction_dataset/`
