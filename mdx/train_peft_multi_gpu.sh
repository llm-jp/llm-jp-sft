#!/bin/sh
model_name_or_path=$1
tokenizer_name_or_path=$2
dataset_path=$3
dataset_sh=$4
num_train_epochs=$5
output_dir=$6
per_device_train_batch_size=$7
gradient_accumulation_steps=$8
accelerate launch --config_file configs/accelerate_config_zero1.yaml \
    train.py \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --use_peft \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --lr_scheduler cosine \
    --bf16 \
    --max_seq_length 2048 \
    --logging_steps 10 \
    --report_to wandb \
    --data_files `$dataset_sh $dataset_path/tuning` \
    --eval_data_files `$dataset_sh $dataset_path/develop_small` \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --output_dir $output_dir
