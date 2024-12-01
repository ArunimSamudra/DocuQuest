from main.models.cloud_llm import Cloud
from main.models.local_llm import Local


class LLMFactory:

    @staticmethod
    def create(is_cloud):
        if is_cloud:
            return Cloud()
        else:
            return Local()
