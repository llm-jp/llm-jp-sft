import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
import argparse
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_name_or_path", type=str, help="model id")
    arg_parser.add_argument("--output_dir", type=str)
    arg_parser.add_argument("--dataset", type=str, default="ptb", help="dataset name")
    args = arg_parser.parse_args()
    
    model_name_or_path = args.model_name_or_path
    dataset = args.dataset
    output_dir = args.output_dir
    
    logger.info(f"model_name_or_path: {args.model_name_or_path}")
    logger.info(f"data_set: {args.dataset}")
    logger.info(f"output_dir: {args.output_dir}")

    # tokenizer setup    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # quantization config setup
    quantization_config = GPTQConfig(bits=4, group_size=128, dataset=dataset, desc_act=False)
    # load model
    quantized_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, quantization_config=quantization_config, device_map='auto')
    quantized_model.save_pretrained(model_name_or_path)
    # inference check
    text = "自然言語処理とは何か"
    tokenized_input = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(quantized_model.device)

    with torch.no_grad():
        output = quantized_model.generate(tokenized_input, max_new_tokens=100, do_sample=True, top_p=0.95, temperature=0.7)[0]
    result = tokenizer.decode(output)
    logger.info(f"Q:{text}. A:{result}")