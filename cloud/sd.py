from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the draft model (LLaMA 3.2 3B)
draft_model_name = "llama-3.2-3b"
draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name, torch_dtype=torch.float16, device_map="auto")

# Load the verifier model (LLaMA 3.1 8B)
verifier_model_name = "llama-3.1-8b"
verifier_tokenizer = AutoTokenizer.from_pretrained(verifier_model_name)
verifier_model = AutoModelForCausalLM.from_pretrained(verifier_model_name, torch_dtype=torch.float16, device_map="auto")

# Ensure the tokenizers are aligned
assert draft_tokenizer.vocab == verifier_tokenizer.vocab, "Draft and verifier tokenizers must have the same vocabulary."


# Function for speculative decoding
def speculative_decoding(prompt, max_length=50, alpha=1.2):
    """
    Implements speculative decoding with LLaMA 3.2 as the draft model
    and LLaMA 3.1 as the verifier model.

    Args:
        prompt (str): Input text prompt.
        max_length (int): Maximum number of tokens to generate.
        alpha (float): Threshold for speculative decoding (controls verifier acceptance).

    Returns:
        str: Generated text.
    """
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
            verifier_logits = verifier_model(verifier_input_ids).logits[:, -1, :]
            verifier_probs = torch.softmax(verifier_logits, dim=-1)

        # Acceptance step
        if torch.max(verifier_probs[:, draft_token]) > (1 / alpha):
            # Accept the draft token
            generated_tokens = torch.cat([generated_tokens, draft_generated_tokens[:, step:step + 1]], dim=1)
        else:
            # If not accepted, fallback to verifier for token generation
            with torch.no_grad():
                fallback_logits = verifier_model(generated_tokens).logits[:, -1, :]
                fallback_token = torch.argmax(torch.softmax(fallback_logits, dim=-1), dim=-1).unsqueeze(0)
                generated_tokens = torch.cat([generated_tokens, fallback_token], dim=1)

        # Stop if EOS token is generated
        if draft_tokenizer.eos_token_id in generated_tokens:
            break

    # Decode the final generated text
    return draft_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)


# Example usage
if __name__ == "__main__":
    prompt = "Once upon a time, in a distant kingdom"
    output = speculative_decoding(prompt, max_length=100)
    print("Generated Text:")
    print(output)
