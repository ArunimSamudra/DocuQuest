import asyncio

from main.impl.doc_retriever import DocumentRetriever
from main.impl.llm_factory import LLMFactory
from main.modules.decision_module import use_cloud_llm


class RequestHandler:

    def __init__(self):
        self.llm_factory = LLMFactory()
        self.doc_retriever = DocumentRetriever()
        self.indexing_task = None

    async def summarize_text(self, model, tokenizer, document_text):
        """
        Summarizes the document text and indexes it asynchronously.
        """
        use_cloud = use_cloud_llm(document_text)
        llm = self.llm_factory.create(use_cloud, model, tokenizer)

        # Perform asynchronous indexing
        self.indexing_task = asyncio.create_task(self.doc_retriever.index(document_text))

        return llm.summarize_text(document_text)

    async def answer(self, question, model, tokenizer):
        """
        Retrieves context and answers the question asynchronously.
        """
        # Wait for indexing to complete if it's still running
        if self.indexing_task and not self.indexing_task.done():
            print("Indexing in progress. Waiting for completion...")
            await self.indexing_task
        context = self.doc_retriever.retrieve(question)
        use_cloud = use_cloud_llm(context)
        llm = self.llm_factory.create(use_cloud, model, tokenizer)
        return llm.answer(context, question)
