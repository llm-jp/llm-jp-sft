#!/bin/sh
# 1.3B
accelerate launch --config_file accelerate_config_zero2.yaml \
    train.py \
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
    --data_files dataset/tuning/jamp.json dataset/tuning/janli.json dataset/tuning/jcommonsenseqa.json dataset/tuning/jemhopqa.json dataset/tuning/jnli.json dataset/tuning/jsem.json dataset/tuning/jsick.json dataset/tuning/jsquad.json dataset/tuning/jsts.json dataset/tuning/niilc.json \
    --eval_data_files dataset/develop_small/jamp.json dataset/develop_small/janli.json dataset/develop_small/jcommonsenseqa.json dataset/develop_small/jemhopqa.json dataset/develop_small/jnli.json dataset/develop_small/jsem.json dataset/develop_small/jsick.json dataset/develop_small/jsquad.json dataset/develop_small/jsts.json dataset/develop_small/niilc.json \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --model_name_or_path models/llmjp-1.3b-refined \
    --tokenizer_name_or_path llm-jp/hf-fast-tokenizer-v21b3 \
    --output_dir /model/kiyomaru/sft/results/llmjp-1.3b-refined.js
# 13B
accelerate launch --config_file accelerate_config.yaml \
    train.py \
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
    --data_files dataset/tuning/jamp.json dataset/tuning/janli.json dataset/tuning/jcommonsenseqa.json dataset/tuning/jemhopqa.json dataset/tuning/jnli.json dataset/tuning/jsem.json dataset/tuning/jsick.json dataset/tuning/jsquad.json dataset/tuning/jsts.json dataset/tuning/niilc.json \
    --eval_data_files dataset/develop_small/jamp.json dataset/develop_small/janli.json dataset/develop_small/jcommonsenseqa.json dataset/develop_small/jemhopqa.json dataset/develop_small/jnli.json dataset/develop_small/jsem.json dataset/develop_small/jsick.json dataset/develop_small/jsquad.json dataset/develop_small/jsts.json dataset/develop_small/niilc.json \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --model_name_or_path models/llmjp-13b-refined \
    --tokenizer_name_or_path llm-jp/hf-fast-tokenizer-v21b3 \
    --output_dir /model/kiyomaru/sft/results/llmjp-13b-refined.js
