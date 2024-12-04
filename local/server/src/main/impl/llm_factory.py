from main.impl.cloud_llm import Cloud
from main.impl.local_llm import Local


class LLMFactory:

    @staticmethod
    def create(is_cloud, model, tokenizer):
        if is_cloud:
            return Cloud()
        else:
            return Local(model, tokenizer)
