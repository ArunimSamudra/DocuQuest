from mlx_lm import generate, load

from src.main.config import Config
from src.main.models.llm import LLM


class Local(LLM):

    def __init__(self):
        self.model, self.tokenizer = load(path_or_hf_repo=Config.LOCAL_MODEL_PATH)
        self.generation_args = {
            "temp": 0.7,
            "repetition_penalty": 1.2,
            "repetition_context_size": 20,
            "top_p": 0.95,
        }

    def summarize_text(self, text):
        max_tokens = 1_000
        prompt = "Summarize the given text"
        conversation = [{"role": "user", "content": prompt}, {"role": "user", "content": text}]

        prompt = self.tokenizer.apply_chat_template(
            conversation=conversation, tokenize=False, add_generation_prompt=True
        )

        return generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            # max_tokens=max_tokens,
            verbose=True,
            **self.generation_args,
        )

    def answer(self, context, question):
        raise NotImplementedError
