import asyncio

from fastapi import HTTPException

from main.impl.doc_retriever import DocumentRetriever
from main.impl.llm_factory import LLMFactory
from main.modules.decision_module import use_cloud_llm


class RequestHandler:

    def __init__(self):
        self.llm_factory = LLMFactory()
        self.doc_retriever = DocumentRetriever()
        self.indexing_tasks = {}

    async def summarize_text(self, model, tokenizer, session_id, document_text):
        """
        Summarizes the document text and indexes it asynchronously.
        """
        use_cloud = use_cloud_llm(document_text)
        if use_cloud:
            print("Using cloud")
        else:
            print("Using local")
        llm = self.llm_factory.create(use_cloud, model, tokenizer)

        # Perform asynchronous indexing
        if session_id not in self.indexing_tasks:
            print("Creating indexing task")
            self.indexing_tasks[session_id] = asyncio.create_task(
                self.doc_retriever.index(session_id, document_text)
            )

        print("Going to summarize")
        response = llm.summarize_text(document_text)
        # Fallback to local if cloud fails
        if "error" in response is not None and use_cloud:
            llm = self.llm_factory.create(use_cloud, model, tokenizer)
            return llm.summarize_text(document_text)
        return response

    async def answer(self, model, tokenizer, session_id, question):
        """
        Retrieves context and answers the question asynchronously.
        """
        # Wait for indexing to complete if it's still running
        if session_id in self.indexing_tasks and not self.indexing_tasks[session_id].done():
            print(f"Indexing in progress for session {session_id}. Waiting for completion...")
            await self.indexing_tasks[session_id]
        context = self.doc_retriever.retrieve(session_id, question)
        use_cloud = use_cloud_llm(context)
        llm = self.llm_factory.create(use_cloud, model, tokenizer)

        answer = llm.answer(context, question)
        # Fallback to local if cloud fails
        if "error" in answer is not None and use_cloud:
            llm = self.llm_factory.create(use_cloud, model, tokenizer)
            return llm.answer(context, question)
        return answer

    async def delete_session(self, session_id):
        if session_id in self.doc_retriever.session_stores:
            del self.doc_retriever.session_stores[session_id]
            return {"message": f"Session {session_id} deleted successfully."}
        else:
            raise HTTPException(status_code=404, detail="Session not found.")
