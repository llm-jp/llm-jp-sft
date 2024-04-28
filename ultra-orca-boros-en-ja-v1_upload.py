import wandb
import pandas as pd
import json
from sklearn.model_selection import train_test_split


project_name = "llm-finetuning-with-high-quality-dataset"
entity_name = "wandb-japan"

from datasets import load_dataset
dataset = load_dataset("augmxnt/ultra-orca-boros-en-ja-v1")

def format_conversations(conversations):
    formatted_text = ""
    for conversation in conversations:
        if conversation['from'] == 'human':
            formatted_text += f"USER:{conversation['value']}\n"
        elif conversation['from'] == 'gpt':
            formatted_text += f"ASSISTANT:{conversation['value']}<|endoftext|>"
    return formatted_text.strip()

with open('data/ultra-orca-boros-en-ja.json', 'w', encoding='utf-8') as file:
    for i, data_point in enumerate(dataset['train']):
        conversations = data_point['conversations']
        formatted_text = format_conversations(conversations)
        json_data = {
            "id":data_point["id"],
            "text": formatted_text,
            "source": data_point["source"],
            "text_length":len(formatted_text)
        }
        json_line = json.dumps(json_data, ensure_ascii=False)
        file.write(json_line + '\n')

with open('data/ultra-orca-boros-en-ja.json', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    data = [json.loads(line) for line in lines]

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

with open('data/ultra-orca-boros-en-ja-train.json', 'w', encoding='utf-8') as file:
    for item in train_data:
        json_line = json.dumps({"text": item["text"]}, ensure_ascii=False)
        file.write(json_line + '\n')

with open('data/ultra-orca-boros-en-ja-val.json', 'w', encoding='utf-8') as file:
    for item in val_data:
        json_line = json.dumps({"text": item["text"]}, ensure_ascii=False)
        file.write(json_line + '\n')
        
with wandb.init(project=project_name, entity=entity_name,job_type="data-upload") as run:
    artifact = wandb.Artifact("ultra-orca-boros-en-ja-full", 
                              type="dataset")
    artifact.add_file("data/ultra-orca-boros-en-ja.json")
    artifact.add_file("data/ultra-orca-boros-en-ja-train.json")
    artifact.add_file("data/ultra-orca-boros-en-ja-val.json")
    table = wandb.Table(dataframe=pd.DataFrame(data))
    artifact.add(table,"ultra-orca-boros-en-ja_table")
    #run.log({"ultra-orca-boros-en-jp-all": table})
    run.log_artifact(artifact)