import argparse
import logging
import os
import glob

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the data directory")
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
    data_files = [path for path in glob.glob(os.path.join(args.data_path, "*.jsonl"))]
    assert len(data_files) > 0, "No data files found"
    dataset = load_dataset("json", data_files=data_files)

    logger.info("Formatting prompts")
    response_template = "### Response:"
    response_template_tokens = tokenizer.encode(response_template, add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLM(response_template_tokens, tokenizer=tokenizer)

    logger.info("Setting up trainer")
    trainer = SFTTrainer(
        model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        data_collator=collator,
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
