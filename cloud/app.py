import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import app, request, jsonify

LOCAL_MODEL_DIR = "./llama_model"

# Load the llm and tokenizer
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)

print("Model and tokenizer successfully loaded.")

# Move llm to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data["text"]
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    result = tokenizer.decode(outputs[0])
    return jsonify({"generated_text": result})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080)
