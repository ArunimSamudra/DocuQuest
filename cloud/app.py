import itertools

from beam import endpoint, Image, Volume, env


if env.is_remote():
    import torch
    import os
    import pandas as pd
    from tqdm import tqdm
    from datasets import load_dataset, Dataset
    from torch.utils.data import DataLoader
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

# Model parameters
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
VOLUME_PATH = "./models"


def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=VOLUME_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    print("tokenizer loaded")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=VOLUME_PATH
    )
    print("model loaded")
    return model, tokenizer


def main(context, **inputs):
    # Load model, tokenizer
    model, tokenizer = context.on_start_value

    messages = inputs.pop("messages", None)
    if not messages:
        return {"error": "Please provide messages for text generation."}

    generate_args = {
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    model_inputs = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(model_inputs, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **generate_args
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"output": output_text}


@endpoint(
    secrets=["HF_TOKEN"],
    on_start=load_models,
    name="summarizer",
    cpu=2,
    memory="32Gi",
    gpu="A100-40",
    image=Image(
        python_version="python3.10",
        python_packages=["torch", "transformers", "accelerate"],
    ),
    volumes=[
        Volume(
            name="cached_models",
            mount_path=VOLUME_PATH,
        )
    ],
)
def summarize(context, **inputs):
    return main(context, **inputs)

