import time

import psutil
from mlx_lm import generate
from main.impl.llm import LLM


class Local(LLM):

    def __init__(self, model, tokenizer):
        if model is None or tokenizer is None:
            raise ValueError("Model and tokenizer must be provided!")
        self.model, self.tokenizer = model, tokenizer
        self.generation_args = {
            "temp": 0.7,
            "repetition_penalty": 1.2,
            "repetition_context_size": 20,
            "top_p": 0.95,
        }

    def summarize_text(self, text):
        prompt = (
            "You are a highly intelligent and concise text summarization assistant. "
            "Your task is to extract the key points and summarize the given text clearly and accurately. "
            "Keep the summary informative yet brief."
        )
        conversation = [{"role": "system", "content": prompt}, {"role": "user", "content": text}]

        # Generate prompt using tokenizer
        prompt = self.tokenizer.apply_chat_template(
            conversation=conversation, tokenize=False, add_generation_prompt=True
        )

        # Measure time and memory
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024**2)
        start_time = time.time()

        # Perform text generation
        output = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            verbose=False,
            #**self.generation_args,
        )

        # End time and memory measurement
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024**2)

        # Calculate metrics
        time_taken = end_time - start_time
        memory_used = end_memory - start_memory

        return {
            "summary": output,
            "time_taken": f"{time_taken:.2f} seconds",
            "memory_used": f"{memory_used:.2f} MB",
        }

    def answer(self, context, question):
        system_prompt = (
            "You are a highly intelligent and context-aware assistant. Your role is to help the user "
            "answer questions accurately and concisely based on the provided context. Use the context "
            "carefully to generate relevant responses. If the context does not contain enough information "
            "to answer the question, let the user know by stating: 'The context does not provide enough information to answer this question.'"
        )
        text = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]

        # Generate prompt using tokenizer
        prompt = self.tokenizer.apply_chat_template(
            conversation=conversation, tokenize=False, add_generation_prompt=True
        )
        output = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            verbose=False,
            **self.generation_args,
        )
        return {"answer": output}
