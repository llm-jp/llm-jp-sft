import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "cyberagent/calm2-7b-chat"
project_name = "llm-finetuning-with-high-quality-dataset"
entity_name = "wandb-japan"

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        force_download=True,
        resume_download=False
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained("model")
tokenizer.save_pretrained("model")

with wandb.init(project=project_name, entity=entity_name) as run:
    artifact = wandb.Artifact("cyberagent-calm2-7b-chat", type="model")
    artifact.add_dir("model")
    run.log_artifact(artifact)