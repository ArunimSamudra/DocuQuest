import time
import tracemalloc

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
        max_tokens = 1_000

        # Improved system prompt
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
        tracemalloc.start()
        start_time = time.time()

        # Perform text generation
        output = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            verbose=True,
            **self.generation_args,
        )

        # End time and memory measurement
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Calculate metrics
        time_taken = end_time - start_time
        current_memory = current / 10 ** 6  # Convert to MB
        peak_memory = peak / 10 ** 6  # Convert to MB

        # Log results
        print(f"Time taken: {time_taken:.2f} seconds")
        print(f"Memory used: {current_memory:.2f} MB")
        print(f"Peak memory: {peak_memory:.2f} MB")

        return {
            "summary": output,
            "time_taken": f"{time_taken:.2f} seconds",
            "memory_used": f"{current_memory:.2f} MB",
            "peak_memory": f"{peak_memory:.2f} MB",
        }

    def answer(self, context, question):
        raise NotImplementedError
