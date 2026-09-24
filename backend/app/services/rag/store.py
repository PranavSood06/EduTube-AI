import logging
from pathlib import Path
from typing import List

import chromadb
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent.parent

        self.client = chromadb.PersistentClient(
            path=str(base_dir / "data" / "chroma")
        )

        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

    def create_new_collection(self, name: str):
        logger.info("Creating new collection: %s", name)

        collection = self.client.get_or_create_collection(
            name=name
        )

        return collection

    def embed_docs(self, documents: List[Document]):
        logger.info(
            "Generating embeddings for %d documents",
            len(documents)
        )

        texts = [doc.page_content for doc in documents]

        embeddings = self.embedding_model.embed_documents(texts)

        return embeddings

    def store_in_collection(
        self,
        chunk_docs: List[Document],
        name: str
    ):
        collection = self.create_new_collection(name)

        embeddings = self.embed_docs(chunk_docs)

        ids = [
            doc.metadata["chunk_id"]
            for doc in chunk_docs
        ]

        documents = [
            doc.page_content
            for doc in chunk_docs
        ]

        metadatas = [
            doc.metadata
            for doc in chunk_docs
        ]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        logger.info(
            "Stored %d documents in collection '%s'",
            len(chunk_docs),
            name
        )