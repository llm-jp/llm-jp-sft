import argparse
import logging
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, get_cosine_schedule_with_warmup
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset, disable_caching

from lion import Lion

disable_caching()

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_files", nargs="+", help="Path to the data files")
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="Path to pretrained tokenizer or tokenizer identifier")
    args = parser.parse_args()

    logger.info(f"Loading model from {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")

    logger.info(f"Loading tokenizer from {args.tokenizer_name_or_path}")
    tokenizer_name_or_path: str = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    logger.info(f"Loading data")
    data_files = args.data_files
    assert len(data_files) > 0, "No data files found"
    dataset = load_dataset("json", data_files=data_files)

    logger.info("Formatting prompts")
    response_template = "回答："
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    optimizer = Lion(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    num_total_steps = len(dataset["train"]) // 32
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_total_steps // 10, num_total_steps)

    logger.info("Setting up trainer")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        # learning_rate=1e-5,
        # warmup_ratio=0.1,
        # lr_scheduler_type="cosine",
        fp16=True,
        save_steps=50_000,
        logging_steps=10,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        data_collator=collator,
        max_seq_length=512,
        optimizers=(optimizer, scheduler),
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
