#!/bin/sh
config_file=$1
model_name_or_path=$2
tokenizer_name_or_path=$3
dataset_path=$4
dataset_sh=$5
num_train_epochs=$6
output_dir=$7
per_device_train_batch_size=$8
gradient_accumulation_steps=$9
accelerate launch --config_file $config_file \
    train.py \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler cosine \
    --bf16 \
    --max_seq_length 2048 \
    --logging_steps 1 \
    --report_to wandb \
    --data_files `./$dataset_sh $dataset_path/tuning` \
    --eval_data_files `./$dataset_sh $dataset_path/develop_small` \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --output_dir $output_dir
