import time
from beam import endpoint, Image, Volume, env

if env.is_remote():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import psutil

DRAFT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TARGET_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
BEAM_VOLUME_PATH = "./cached_models"


def load_models():
    draft_tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_NAME, cache_dir=BEAM_VOLUME_PATH)
    draft_tokenizer.pad_token = draft_tokenizer.eos_token
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=BEAM_VOLUME_PATH,
    )
    print("Draft Model and tokenizer successfully loaded.")

    target_tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_NAME, cache_dir=BEAM_VOLUME_PATH)
    target_tokenizer.pad_token = target_tokenizer.eos_token
    target_model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_NAME, device_map="auto",
                                                        torch_dtype=torch.float16,
                                                        cache_dir=BEAM_VOLUME_PATH)
    print("Target Model and tokenizer successfully loaded.")
    return draft_model, draft_tokenizer, target_model, target_tokenizer


# Function for speculative decoding
def speculative_decoding(context, **inputs, max_length=10, alpha=1.2):
    draft_model, draft_tokenizer, target_model, target_tokenizer = context.on_start_value
    prompt = inputs.pop("prompt", None)

    # Tokenize the input
    input_ids = draft_tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    # Initialize outputs
    generated_tokens = input_ids.clone()
    # Generate with the draft model
    draft_outputs = draft_model.generate(input_ids, max_new_tokens=max_length, do_sample=True)
    draft_generated_tokens = draft_outputs[:, input_ids.shape[-1]:]  # Only new tokens


    # Verify with the verifier model
    for step, draft_token in enumerate(draft_generated_tokens[0]):
        verifier_input_ids = torch.cat([generated_tokens, draft_generated_tokens[:, :step + 1]], dim=1)
        with torch.no_grad():
            verifier_logits = target_model(verifier_input_ids).logits[:, -1, :]
            verifier_probs = torch.softmax(verifier_logits, dim=-1)

        # Acceptance step
        if torch.max(verifier_probs[:, draft_token]) > (1 / alpha):
            # Accept the draft token
            generated_tokens = torch.cat([generated_tokens, draft_generated_tokens[:, step:step + 1]], dim=1)
        else:
            # If not accepted, fallback to verifier for token generation
            with torch.no_grad():
                fallback_logits = target_tokenizer(generated_tokens).logits[:, -1, :]
                fallback_token = torch.argmax(torch.softmax(fallback_logits, dim=-1), dim=-1).unsqueeze(0)
                generated_tokens = torch.cat([generated_tokens, fallback_token], dim=1)

        # Stop if EOS token is generated
        if draft_tokenizer.eos_token_id in generated_tokens:
            break

    # Decode the final generated text
    return draft_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

@endpoint(
    secrets=["HF_TOKEN"],
    on_start=load_models,
    name="sd",
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
    return speculative_decoding(context, **inputs)
