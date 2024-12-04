from main.impl.llm_factory import LLMFactory
from main.modules.decision_module import use_cloud_llm


class RequestHandler:

    @staticmethod
    def summarize_text(model, tokenizer, document_text):
        use_cloud = use_cloud_llm(document_text)
        llm = LLMFactory().create(use_cloud, model, tokenizer)
        return llm.summarize_text(document_text)

    @staticmethod
    def answer(content):
        return