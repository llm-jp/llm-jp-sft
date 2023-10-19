## Upload model to HuggingFace hub

### For full fine-tuning
```bash
cd llm-jp-sft/utils
python upload.py --model_name_or_path [PATH] --upload_name [UPLOAD_NAME]
```
[UPLOAD_NAME] should NOT include an organization name such as `llm-jp`.

### For PEFT
- First, set `"base_model_name_or_path"` in `adapter_config.json` to appropriate identifier like `llm-jp/llm-jp-13b-v1.0`
```bash
cd llm-jp-sft/utils
python upload.py --peft_model_name_or_path [PATH] --upload_name [UPLOAD_NAME]
```

[UPLOAD_NAME] should NOT include an organization name such as `llm-jp`.
