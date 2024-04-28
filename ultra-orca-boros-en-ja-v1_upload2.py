import wandb
import pandas as pd
import json
from sklearn.model_selection import train_test_split

project_name = "llm-finetuning-with-high-quality-dataset"
entity_name = "wandb-japan"
#sources=["lm-jp-eval"] # list
#sources=["airoboros","airoboros_ja_new","lm-jp-eval","slimorca","ultrafeedback"] # list
sources=["airoboros","airoboros_ja_new","slimorca","ultrafeedback"] # list


with wandb.init(project=project_name, entity=entity_name,job_type="data-upload") as run:
    artifact = run.use_artifact('wandb-japan/llm-finetuning-with-high-quality-dataset/ultra-orca-boros-en-ja-full:v0', type='dataset')
    artifact_dir = artifact.download()

    file_path = artifact_dir + '/ultra-orca-boros-en-ja.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        data = [json.loads(line) for line in lines]

    filtered_data = [item for item in data if item['source'] in sources]

    output_file_path = 'data/ultra-orca-boros-en-ja-subset.json'
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for item in filtered_data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open('data/ultra-orca-boros-en-ja-subset.json', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        data = [json.loads(line) for line in lines]

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    with open('data/ultra-orca-boros-en-ja-subset-train.json', 'w', encoding='utf-8') as file:
        for item in train_data:
            json_line = json.dumps({"text": item["text"]}, ensure_ascii=False)
            file.write(json_line + '\n')

    with open('data/ultra-orca-boros-en-ja-subset-val.json', 'w', encoding='utf-8') as file:
        for item in val_data:
            json_line = json.dumps({"text": item["text"]}, ensure_ascii=False)
            file.write(json_line + '\n')


    artifact = wandb.Artifact('ultra-orca-boros-en-ja-subset',
                                    type='dataset',
                                    metadata={"sources":sources})
    artifact.add_file(output_file_path)
    artifact.add_file("data/ultra-orca-boros-en-ja-subset.json")
    artifact.add_file("data/ultra-orca-boros-en-ja-subset-train.json")
    artifact.add_file("data/ultra-orca-boros-en-ja-subset-val.json")
    table = wandb.Table(dataframe=pd.DataFrame(data))
    artifact.add(table,"ultra-orca-boros-en-ja-subset_table")
    #run.log({"ultra-orca-boros-en-jp-all": table})
    run.log_artifact(artifact)