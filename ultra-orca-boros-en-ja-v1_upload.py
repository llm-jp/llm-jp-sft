import wandb
import json

project_name = "llm-finetuning-with-high-quality-dataset"
entity_name = "wandb-japan"

from datasets import load_dataset
dataset = load_dataset("augmxnt/ultra-orca-boros-en-ja-v1")

def format_conversations(conversations):
    formatted_text = ""
    for conversation in conversations:
        if conversation['from'] == 'human':
            formatted_text += f"User: {conversation['value']}\n"
        elif conversation['from'] == 'gpt':
            formatted_text += f"ASSISTANT: {conversation['value']}<|endoftext|>"
    return formatted_text.strip()

with open('dataset/ultra-orca-boros-en-ja.jsonl', 'w', encoding='utf-8') as file:
    for i, data_point in enumerate(dataset['train']):
        conversations = data_point['conversations']
        formatted_text = format_conversations(conversations)
        json_data = {
            "id": i + 1,
            "text": formatted_text
        }
        json_line = json.dumps(json_data, ensure_ascii=False)
        file.write(json_line + '\n')

with wandb.init(project=project_name, entity=entity_name) as run:
    artifact = wandb.Artifact("ultra-orca-boros-en-ja", type="data")
    artifact.add_file("dataset/ultra-orca-boros-en-ja.jsonl")
    run.log_artifact(artifact)