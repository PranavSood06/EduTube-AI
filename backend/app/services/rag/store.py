import asyncio
import logging
import os
from typing import Sequence

import chromadb
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_voyageai import VoyageAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(self):
        api_key = os.getenv("CHROMA_API_KEY")
        tenant = os.getenv("CHROMA_TENANT")
        database = os.getenv("CHROMA_DATABASE")

        if not api_key or not tenant:
            raise RuntimeError(
                "CHROMA_API_KEY and CHROMA_TENANT must be configured for Chroma Cloud"
            )

        self.client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database,
        )

        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2"
        )

    async def get_or_create_collection(self, name: str):
        """Get a cloud collection, creating it only for a newly indexed video."""
        logger.info("Getting or creating cloud collection: %s", name)
        return await asyncio.to_thread(
            self.client.get_or_create_collection, name=name
        )

    # Retained for callers that still use the older method name.
    create_new_collection = get_or_create_collection

    async def embed_docs(
        self,
        documents: Sequence[Document],
        batch_size: int = 90,
        wait_seconds: int = 65,
    ) -> list[list[float]]:
        logger.info(
            "Generating embeddings for %d documents in batches of %d",
            len(documents),
            batch_size,
        )

        texts = [doc.page_content for doc in documents]

        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            batch_number = (start // batch_size) + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size

            logger.info(
                "Embedding batch %d/%d (%d texts)",
                batch_number,
                total_batches,
                len(batch),
            )

            embeddings = await self.embedding_model.aembed_documents(batch)

            all_embeddings.extend(embeddings)

            if start + batch_size < len(texts):
                logger.info(
                    "Batch %d complete. Waiting %d seconds before next batch...",
                    batch_number,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)

        return all_embeddings

    async def store_in_collection(
        self,
        chunk_docs: Sequence[Document],
        name: str
    ) -> int:
        """Embed and upsert documents without blocking the event loop."""
        collection = await self.get_or_create_collection(name)

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
