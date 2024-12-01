from main.models.llm import LLM


class Cloud(LLM):

    def summarize_text(self, text):
        raise NotImplementedError

    def answer(self, context, question):
        raise NotImplementedError