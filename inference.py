
import argparse
import wandb
import json

import torch
from peft import PeftModel, PeftConfig
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLMモデルの推論を行うスクリプト")
    parser.add_argument(
        "--model_artifacts_path", type=str, help="推論対象のモデルのパス"
    )
    parser.add_argument(
        "--input_artifacts_path", type=str, help="推論対象の入力データのパス"
    )
    # generation config
    parser.add_argument("--do_sample", action="store_true", help="サンプリングするかどうか")
    parser.add_argument("--max_length", type=int, default=2048, help="生成するトークンの最大長")
    parser.add_argument("--temperature", type=float, default=0.7, help="サンプリング時の温度")
    parser.add_argument("--top_p", type=float, default=0.95, help="トークンの選択確率の上限")
    parser.add_argument("--repetition_penalty", type=float, default=1.05, help="トークンの繰り返しペナルティ")
    # output config
    parser.add_argument("--model_name_aka", type=str, default="", help="モデルの名前")
    parser.add_argument("--prompt_type", type=str, choices=["alpaca", "chat", "inst", "none"], default="alpaca")
    return parser.parse_args()

def main():
    with wandb.init() as run:
        args = get_args()
        print(args)

        model_artifact = run.use_artifact(args.model_artifacts_path)
        model_path = model_artifact.download()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")

        if torch.cuda.is_available():
            model.to("cuda")

        artifact_dir = run.use_artifact(args.input_artifacts_path).download()
        data_file = artifact_dir+f"/AnswerCarefullyVersion001_Test.json"
        with open(data_file, "r") as f:
            original_data = json.load(f)

        prompt_format: str
        if args.prompt_type == "alpaca":
            prompt_format = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{}\n\n### 応答:\n"
        elif args.prompt_type == "chat":
            prompt_format = "USER: {}\nASSISTANT: "
        elif args.prompt_type == "inst":
            # https://huggingface.co/docs/transformers/v4.39.1/en/chat_templating
            prompt_format = "[INST]{}[/INST]"
        elif args.prompt_type == "none":
            prompt_format = "{}"
        else:
            raise NotImplementedError()

        output_llm_outputs: list[dict[str, str]] = []
        with torch.no_grad():
            for sample in tqdm(original_data):
                input_text = prompt_format.format(sample["text"])
                input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False)
                if torch.cuda.is_available():
                    input_ids = input_ids.to("cuda")
                output = model.generate(
                    input_ids=input_ids,
                    do_sample=args.do_sample,
                    max_new_tokens=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )[0]
                output = output[input_ids.size(1):]

                output_text = tokenizer.decode(output.tolist(), skip_special_tokens=True)
                output_llm_outputs.append({
                    "ID": sample["ID"],
                    "model": args.model_name_aka if args.model_name_aka != "" else args.model_path,
                    "input-text": sample["text"],
                    "input-prompt": input_text,
                    "output": output_text,
                    "reference": sample["output"],
                })
        df = pd.DataFrame(output_llm_outputs)
        output_table = wandb.Table(dataframe=df)
        run.log({"output_to_safetydata": output_table})


        #with open(args.output_path, "w") as f:
        #    for sample in output_llm_outputs:
        #        f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()