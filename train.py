import argparse
import logging
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset, disable_caching

disable_caching()

logger = logging.getLogger(__name__)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        text = ""
        if "input" in example and example["input"][i]:
            text += f"{example['input'][i]}\n\n"
        text += f"{example['instruction'][i]} ### Response: {example['output'][i]}"
        output_texts.append(text)
    return output_texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the data directory")
    parser.add_argument("--data_files", nargs="+", help="Path to the data files")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Path to pretrained tokenizer or tokenizer identifier")
    args = parser.parse_args()

    logger.info(f"Loading model from {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)

    logger.info(f"Loading tokenizer from {args.tokenizer_name_or_path}")
    tokenizer_name_or_path: str = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    logger.info(f"Loading data from {args.data_path}")
    data_files = [os.path.join(args.data_path, data_file) for data_file in args.data_files]
    assert len(data_files) > 0, "No data files found"
    dataset = load_dataset("json", data_files=data_files)

    logger.info("Formatting prompts")
    response_template = "### Response:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    logger.info("Setting up trainer")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        fp16=True,
        ddp_backend="nccl",
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=256,
    )

    logger.info("Training")
    trainer.train()

    logger.info("Saving model")
    trainer.save_model(args.output_dir)



if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
