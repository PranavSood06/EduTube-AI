from langchain_chroma import Chroma
from ..embeddings import Embeddings
# from app.rag.embeddings.goggle_genai import Embeddings
from pathlib import Path

collection_name = "Youtube"
embedding = Embeddings()
model = embedding.get_model()
dir = Path(__file__).resolve().parent.parent.parent.parent
dir = dir/"data"

class Store:
    def __init__(self, collection_name: str):
        self.storage = Chroma(
            collection_name=collection_name,
            embedding_function=model,
            persist_directory=dir
        )

    def add_documents(self, docs):
        return self.storage.add_documents(docs)

    def add_texts(self, texts):
        return self.storage.add_texts(texts)

    def as_retriever(self,k: int = 5):
        return self.storage.as_retriever(
            k=k
        )
