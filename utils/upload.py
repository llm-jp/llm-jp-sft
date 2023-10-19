import argparse
import logging
# import os
# os.environ['TRANSFORMERS_CACHE'] = '/model/s.sasaki/.cache'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained tokenizer or tokenizer identifier",
    )
    parser.add_argument(
        "--peft_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained peft model or model identifier",
    )
    parser.add_argument(
        "--upload_name",
        type=str,
        help="Model name for model identifier on huggingface hub",
    )
    args = parser.parse_args()

    if args.peft_model_name_or_path:
        logger.info(f"Loading from {args.peft_model_name_or_path}")
        peft_model_id = args.peft_model_name_or_path
        config = PeftConfig.from_pretrained(peft_model_id)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto", torch_dtype=torch.float16)
        model = PeftModel.from_pretrained(model, peft_model_id)
    else:
        logger.info(f"Loading from {args.model_name_or_path}")
        tokenizer_name_or_path: str = args.tokenizer_name_or_path or args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", torch_dtype=torch.float16)

    ORGANIZATION = "llm-jp"
    logger.info(f"Uploading to {ORGANIZATION}/{args.upload_name}")
    model.push_to_hub(repo_id=f"{ORGANIZATION}/{args.upload_name}", private=True, use_auth_token=True)
    tokenizer.push_to_hub(repo_id=f"{ORGANIZATION}/{args.upload_name}", private=True, use_auth_token=True)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
