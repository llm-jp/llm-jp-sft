# Supervised Fine-tuning for LLMs

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
# Setup HF accelerator properly
accelerate config

# Launch the training script
accelerate launch train.py \
    --data_files <path to jamp.json> <path to janli.json> ... \
    --model_name_or_path <path to HF model> \
    --tokenizer_name_or_path <path to HF tokenizer> \
    --output_dir <path to output directory>
