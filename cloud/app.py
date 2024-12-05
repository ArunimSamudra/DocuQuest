import json
import time
from beam import endpoint, Image, Volume, env

if env.is_remote():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import psutil

# Load the LLM and tokenizer
DRAFT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
BEAM_VOLUME_PATH = "./cached_models"


def load_models():
    tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_NAME, cache_dir=BEAM_VOLUME_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=BEAM_VOLUME_PATH,
    )
    print("Model and tokenizer successfully loaded.")
    return model, tokenizer


def summarize(context, **inputs):
    model, tokenizer = context.on_start_value

    generate_args = {
        "max_length": 10000,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    prompt = inputs.pop("prompt", None)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    # Measure time and memory
    process = psutil.Process()
    start_memory = process.memory_info().rss / (1024 ** 2)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **generate_args
        )
    generated_ids = outputs[0][len(input_ids[0]):]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    # End time and memory measurement
    end_time = time.time()
    end_memory = process.memory_info().rss / (1024 ** 2)

    # Calculate metrics
    time_taken = end_time - start_time
    memory_used = end_memory - start_memory
    return {
        "summary": output_text,
        "time_taken": f"{time_taken:.2f} seconds",
        "memory_used": f"{memory_used:.2f} MB",
    }


@endpoint(
    secrets=["HF_TOKEN"],
    on_start=load_models,
    name="cloud",
    cpu=2,
    memory="32Gi",
    gpu="A100-40",
    image=Image(
        python_version="python3.9",
        python_packages=["torch", "transformers", "accelerate", "psutil"],
    ),
    volumes=[
        Volume(
            name="cached_models",
            mount_path=BEAM_VOLUME_PATH,
        )
    ],
)
def generate_text(context, **inputs):
    return summarize(context, **inputs)
