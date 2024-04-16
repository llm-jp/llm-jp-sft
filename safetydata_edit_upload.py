import wandb
import json

data_name = "AnswerCarefullyVersion000_Dev"
project_name = "llm-safety-finetuning"
entity_name = "wandb-japan"


def transform_json(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            new_text = f"USER:以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。{entry['text']}\nASSISTANT:{entry['output']}"
            transformed_entry = {"text": new_text}
            json_string = json.dumps(transformed_entry, ensure_ascii=False)
            file.write(json_string + '\n')

input_file_path = '../AnswerCarefullyVersion000_Dev.json'
output_file_path = '../AnswerCarefullyVersion000_Dev_transformed.json'

transform_json(input_file_path, output_file_path)

with wandb.init(project=project_name, entity=entity_name) as run:
    artifact = wandb.Artifact("AnswerCarefullyVersion000_Dev", type="data")
    artifact.add_file(output_file_path)
    run.log_artifact(artifact)