import asyncio
import logging
from pathlib import Path
from typing import Sequence

import chromadb
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent.parent

        self.client = chromadb.PersistentClient(
            path=str(base_dir / "data" / "chroma")
        )

        self.embedding_model = VoyageAIEmbeddings(
            model="voyage-4"
        )

    async def create_new_collection(self, name: str):
        logger.info("Creating new collection: %s", name)
        return await asyncio.to_thread(
            self.client.get_or_create_collection, name=name
        )

    async def embed_docs(self, documents: Sequence[Document]) -> list[list[float]]:
        logger.info(
            "Generating embeddings for %d documents",
            len(documents)
        )

        texts = [doc.page_content for doc in documents]

        return await self.embedding_model.aembed_documents(texts)

    async def store_in_collection(
        self,
        chunk_docs: Sequence[Document],
        name: str
    ) -> int:
        """Embed and upsert documents without blocking the event loop."""
        collection = await self.create_new_collection(name)

        embeddings = await self.embed_docs(chunk_docs)

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

        await asyncio.to_thread(
            collection.upsert,
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
        return len(chunk_docs)

    async def get_collection(self, name: str):
        """Fetch a Chroma collection without blocking the event loop."""
        return await asyncio.to_thread(self.client.get_collection, name=name)
