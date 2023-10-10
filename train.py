import argparse
import logging

from datasets import disable_caching, load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

disable_caching()

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_files", nargs="+", help="Path to the data files")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier",
    )
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained tokenizer or tokenizer identifier",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1.0,
        help="Number of train epochs",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per GPU/TPU core/CPU for training",
    )
    parser.add_argument(
        "--global_train_batch_size",
        type=int,
        default=64,
        help="Global batch size for training",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--use_lionw", action="store_true", help="Use LionW instead of AdamW"
    )
    args = parser.parse_args()

    logger.info(f"Loading tokenizer from {args.tokenizer_name_or_path}")
    tokenizer_name_or_path: str = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    logger.info(f"Loading data")
    dataset = load_dataset("json", data_files=args.data_files)

    logger.info("Formatting prompts")
    response_template = "回答："
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    logger.info("Setting up trainer")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=(
            args.global_train_batch_size // args.per_device_train_batch_size
        ),
        optim="adamw_torch" if not args.use_lionw else "lion_32bit",
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        save_steps=50_000,
        logging_steps=10,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        args.model_name_or_path,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        data_collator=collator,
        max_seq_length=args.max_seq_length,
    )

    logger.info("Training")
    trainer.train()

    logger.info("Saving model")
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
