import logging
from dataclasses import dataclass
from typing import Optional

from peft import PeftModel
import torch
import wandb
from peft import LoraConfig
from datasets import disable_caching, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

disable_caching()

logger = logging.getLogger(__name__)

@dataclass
class SFTTrainingArguments:
    model_name_or_path: str
    data_files: list[str]
    eval_data_files: list[str] = None
    tokenizer_name_or_path: Optional[str] = None
    use_fast: bool = True
    additional_special_tokens: list[str] = None
    max_seq_length: int = 2048
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: bool = False
    use_peft: bool = False
    peft_target_model: Optional[str] = "llm-jp"
    peft_target_modules: Optional[list[str]] = None
    peft_lora_r: int = 8
    peft_lora_alpha: int = 32
    peft_lora_dropout: float = 0.05
    use_model_wandb_artifacts: bool = True
    use_dataset_wandb_artifacts: bool = True
    training_dataset_wandb_artifacts_filepath: str = None
    eval_dataset_wandb_artifacts_filepath: str = None
    model_artifact_name : str = None

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("load_in_8bit and load_in_4bit are mutually exclusive")
        if self.peft_target_model and self.peft_target_modules is None:
            if self.peft_target_model == "llm-jp":
                self.peft_target_modules = ["c_attn", "c_proj", "c_fc"]
            elif self.peft_target_model == "llama":
                # https://github.com/serp-ai/LLaMA-8bit-LoRA/blob/main/finetune_peft_8bit.py
                self.peft_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            elif self.peft_target_model == "llama-all":
                # https://note.com/kan_hatakeyama/n/ncd09c52d26c7
                self.peft_target_modules = [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",
                    "embed_tokens",
                ]
            else:
                logger.warning(
                    f"peft_target_model '{self.peft_target_model}' is not supported, "
                    f"so peft_target_modules is set to None."
                )

    def from_pretrained_kwargs(self, training_args):
        if self.load_in_8bit:
            kwargs = {"load_in_8bit": True}
        elif self.load_in_4bit:
            kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            }
        elif training_args.bf16:
            kwargs = {"torch_dtype": torch.bfloat16}
        else:
            kwargs = {"torch_dtype": torch.float16}
        kwargs["use_flash_attention_2"] = self.use_flash_attention_2
        return kwargs
    
def load_datasets(data_files):
    datasets = []
    for data_file in data_files:
        dataset = load_dataset("json", data_files=data_file)
        dataset = dataset["train"]
        dataset = dataset.select_columns("text")
        datasets.append(dataset)
    return concatenate_datasets(datasets)
   
    
def main() -> None:
    with wandb.init() as run:
        parser = HfArgumentParser((TrainingArguments, SFTTrainingArguments))
        training_args, sft_training_args = parser.parse_args_into_dataclasses()

        if sft_training_args.use_model_wandb_artifacts:
            model_artifact = run.use_artifact(sft_training_args.model_name_or_path)
            model_path = model_artifact.download()
        else:
            model_path = sft_training_args.model_name_or_path

        tokenizer_name_or_path: str = (
            sft_training_args.tokenizer_name_or_path or model_path
        )
        logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_fast=sft_training_args.use_fast,
            additional_special_tokens=sft_training_args.additional_special_tokens,
            trust_remote_code=True,
        )

        logger.info("Loading data")
        
        train_dataset = []
        if sft_training_args.use_dataset_wandb_artifacts:
            artifact_dir = run.use_artifact(sft_training_args.data_files).download()
            data_file = artifact_dir+sft_training_args.training_dataset_wandb_artifacts_filepath
            dataset = load_dataset("json", data_files=data_file)
            dataset = dataset["train"]
            dataset = dataset.select_columns("text")
            train_dataset.append(dataset)
            train_dataset = concatenate_datasets(train_dataset)
        else:
            load_datasets(sft_training_args.data_files)
        
        eval_dataset = []
        if sft_training_args.use_dataset_wandb_artifacts:
            artifact_dir = run.use_artifact(sft_training_args.data_files).download()
            data_file = artifact_dir+sft_training_args.eval_dataset_wandb_artifacts_filepath
            dataset = load_dataset("json", data_files=data_file)
            dataset = dataset["train"]
            dataset = dataset.select_columns("text")
            eval_dataset.append(dataset)
            eval_dataset = concatenate_datasets(eval_dataset)
            training_args.do_eval = True
        else:
            load_datasets(sft_training_args.data_files)
            training_args.do_eval = True

        logger.info("Formatting prompts")
        instruction_ids = tokenizer.encode("USER: ", add_special_tokens=False)[1:]
        response_ids = tokenizer.encode("ASSISTANT: ", add_special_tokens=False)[1:]
        collator = DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_ids,
            response_template=response_ids,
            tokenizer=tokenizer,
        )

        logger.info(f"Loading model from {sft_training_args.model_name_or_path}")
        kwargs = sft_training_args.from_pretrained_kwargs(training_args)
        logger.debug(
            f"AutoModelForCausalLM.from_pretrained({sft_training_args.model_name_or_path}, trust_remote_code=True, **kwargs={kwargs})"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **kwargs,
        )

        peft_config: Optional[LoraConfig] = None
        if sft_training_args.use_peft:
            logger.info("Setting up LoRA")
            peft_config = LoraConfig(
                r=sft_training_args.peft_lora_r,
                target_modules=sft_training_args.peft_target_modules,
                lora_alpha=sft_training_args.peft_lora_alpha,
                lora_dropout=sft_training_args.peft_lora_dropout,
                fan_in_fan_out=True,
                bias="none",
                task_type="CAUSAL_LM",
            )
            if training_args.gradient_checkpointing:
                for param in model.parameters():
                    param.requires_grad = False
                    if param.ndim == 1:
                        param.data = param.data.to(torch.float32)
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
        

        logger.info("Setting up trainer")
        training_args.report_to=["wandb"]
        trainer = SFTTrainer(
            model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            data_collator=collator,
            peft_config=peft_config,
            max_seq_length=sft_training_args.max_seq_length,
        )

        logger.info("Training")
        trainer.train()

        logger.info("Saving model")
        trainer.save_model()
        
        base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
        model = PeftModel.from_pretrained(base_model, "./output")
        model_finetuned = model.merge_and_unload()
        
        model_finetuned.save_pretrained("finetuned_model")
        tokenizer.save_pretrained("finetuned_model")
        artifact = wandb.Artifact(sft_training_args.model_artifact_name,
                                  type="model",
                                  metadata={"training_args":training_args, 
                                            "sft_training_args":sft_training_args})
        artifact.add_dir("finetuned_model")
        run.log_artifact(artifact)
        

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
