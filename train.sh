#!/bin/sh
# 1.3B
accelerate launch train.py \
    --data_files dataset/tuning/jamp.json dataset/tuning/janli.json dataset/tuning/jcommonsenseqa.json dataset/tuning/jemhopqa.json dataset/tuning/jnli.json dataset/tuning/jsem.json dataset/tuning/jsick.json dataset/tuning/jsquad.json dataset/tuning/jsts.json dataset/tuning/niilc.json \
    --model_name_or_path models/llmjp-1.3b-refined \
    --tokenizer_name_or_path models/llmjp-tokenizer-2.1 \
    --output_dir /model/kiyomaru/sft/results/llmjp-1.3b-refined.js
# 13B
accelerate launch train.py \
    --data_files dataset/tuning/jamp.json dataset/tuning/janli.json dataset/tuning/jcommonsenseqa.json dataset/tuning/jemhopqa.json dataset/tuning/jnli.json dataset/tuning/jsem.json dataset/tuning/jsick.json dataset/tuning/jsquad.json dataset/tuning/jsts.json dataset/tuning/niilc.json \
    --model_name_or_path models/llmjp-13b-refined \
    --tokenizer_name_or_path models/llmjp-tokenizer-2.1 \
    --output_dir /model/kiyomaru/sft/results/llmjp-13b-refined.js
