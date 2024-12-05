from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from main.config import Config


class DocumentRetriever:

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=Config.OPENAI_API_KEY)
        self.session_stores = {}

    async def index(self, session_id, text):
        """
        Splits the given text into chunks and indexes it in the vector store asynchronously.
        """
        if session_id not in self.session_stores:
            self.session_stores[session_id] = InMemoryVectorStore(self.embeddings)
        # Simulate asynchronous behavior (e.g., chunk processing or database writes)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = text_splitter.split_text(text)

        self.session_stores[session_id].add_texts(chunks)

    def retrieve(self, session_id, question):
        # Search for relevant text
        if session_id not in self.session_stores:
            return ""
        # Perform similarity search in the session-specific vector store
        search_results = self.session_stores[session_id].similarity_search(question, k=1)
        context = " ".join([result.page_content for result in search_results])
        return context
