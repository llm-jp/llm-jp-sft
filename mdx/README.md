# llm-jp-sft MDX environment

## Set Up

Create directories and symbolic links for the datasets and the models:

```bash
$ mkdir datasets models results
$ ln -s absolute_path_for_dataset_directory datasets/
$ ln -s absolute_path_for_model_directory models/
```

## Run Training Scripts

From root directory of this repository, run `mdx/train_*.sh` as described in the following sections.
Examples are using `jaster` as dataset and `llm-jp-1.3b-v1.0` as base model.

### Full Parameter Supervised Fine-tuning

#### Single-node Multi-GPU Training
- script:
  - 1.3B: `mdx/train_full_single_node.sh`
  - 13B: `mdx/train_full_single_node_gradient_checkpointing.sh`
- args:
  - config_file
  - model_name_or_path
  - tokenizer_name_or_path
  - dataset_path
  - dataset_sh
  - num_train_epochs
  - output_dir
  - per_device_train_batch_size
  - gradient_accumulation_steps
- examples:
  - 1.3B model on A100 40GB 1node_8gpu with `accelerate_config_zero1.yaml`
    - `$ mdx/train_full_single_node.sh accelerate_config_zero1.yaml llm-jp/llm-jp-1.3b-v1.0 llm-jp/llm-jp-1.3b-v1.0 datasets/v1.1.1 dataset_jaster.sh 2 results/llm-jp-1.3b-instruct-full-jaster-v1.0 8 8`
  - 13B model on A100 40GB 1node_8gpu with `accelerate_config_zero3.yaml`
    - `$ mdx/train_full_single_node_gradient_checkpointing.sh accelerate_config_zero3.yaml llm-jp/llm-jp-13b-v1.0 llm-jp/llm-jp-13b-v1.0 datasets/v1.1.1 dataset_jaster.sh 2 results/llm-jp-13b-instruct-full-jaster-v1.0 1 32`

#### Multi-node Multi-GPU Training
For multi-node training, you need to specify the IP address of the network used for inter-node communication of the rank 0 (master) node as `main_process_ip` arguments.
In addition, you need to run `mdx/train_full_multi_node.sh` from all the nodes with specifying the sequential number from 0 to 7 assigned to each node as `machine_rank` argument.
- script:
  - `mdx/train_full_multi_node.sh`
- args:
  - config_file
  - model_name_or_path
  - tokenizer_name_or_path
  - dataset_path
  - dataset_sh
  - num_train_epochs
  - output_dir
  - per_device_train_batch_size
  - gradient_accumulation_steps
  - main_process_ip
  - machine_rank
- example:
  - 13B model on A100 40GB 8node_64gpu with `accelerate_config_zero2.8node.yaml`
    - `$ mdx/train_full_multi_node.sh accelerate_config_zero2.8node.yaml llm-jp/llm-jp-13b-v1.0 llm-jp/llm-jp-13b-v1.0 datasets/v1.1.1 dataset_jaster.sh 2 results/llm-jp-13b-instruct-full-jaster-v1.0 3 6 10.???.???.??? [0-7]`

### Fine-tuning with PEFT

#### Single-GPU Training
- script:
  - 1.3B: `mdx/train_peft_single_gpu.sh`
  - 13B: `mdx/train_peft_single_gpu_gradient_checkpointing.sh`
- args:
  - model_name_or_path
  - tokenizer_name_or_path
  - dataset_path
  - dataset_sh
  - num_train_epochs
  - output_dir
  - per_device_train_batch_size
  - gradient_accumulation_steps
- examples:
  - 1.3B model on single A100 40GB
    - `$ CUDA_VISIBLE_DEVICES=0 mdx/train_peft_single_gpu.sh llm-jp/llm-jp-1.3b-v1.0 llm-jp/llm-jp-1.3b-v1.0 datasets/v1.1.1 dataset_jaster.sh 5 results/llm-jp-1.3b-instruct-lora-jaster-v1.0 8 4`
  - 13B model on single A100 40GB
    - `$ CUDA_VISIBLE_DEVICES=0 mdx/train_peft_single_gpu_gradient_checkpointing.sh llm-jp/llm-jp-13b-v1.0 llm-jp/llm-jp-13b-v1.0 datasets/v1.1.1 dataset_jaster.sh 5 results/llm-jp-13b-instruct-lora-jaster-v1.0 1 32`

#### Single-node Multi-GPU Training
- script:
  - `mdx/train_peft_multi_gpu.sh`
- args:
  - model_name_or_path
  - tokenizer_name_or_path
  - dataset_path
  - dataset_sh
  - num_train_epochs
  - output_dir
  - per_device_train_batch_size
  - gradient_accumulation_steps
- examples:
  - 1.3B model on A100 40GB 1node_8gpu with `accelerate_config_zero1.yaml`
    - `$ mdx/train_peft_multi_gpu.sh llm-jp/llm-jp-1.3b-v1.0 llm-jp/llm-jp-1.3b-v1.0 datasets/v1.1.1 dataset_jaster.sh 5 results/llm-jp-1.3b-instruct-lora-jaster-v1.0 8 8`
  - 13B model on A100 40GB 1node_8gpu with `accelerate_config_zero1.yaml`
    - `$ mdx/train_peft_multi_gpu.sh llm-jp/llm-jp-13b-v1.0 llm-jp/llm-jp-13b-v1.0 datasets/v1.1.1 dataset_jaster.sh 5 results/llm-jp-13b-instruct-lora-jaster-v1.0 1 16`
