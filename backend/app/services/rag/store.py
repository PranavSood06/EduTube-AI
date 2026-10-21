import asyncio
import logging
import os
from typing import Sequence

import chromadb
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_voyageai import VoyageAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langsmith import trace
from .observability import RAG_TAGS, add_run_metadata, embedding_inputs, embedding_outputs, trace_async

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

    @trace_async(
        name="Embed transcript chunks",
        run_type="embedding",
        tags=[*RAG_TAGS, "indexing", "embeddings"],
        process_inputs=embedding_inputs,
        process_outputs=embedding_outputs,
    )
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
        add_run_metadata(
            embedding_model="gemini-embedding-2",
            document_count=len(documents),
            batch_size=batch_size,
        )

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

            with trace(
                "Embed transcript batch", run_type="embedding",
                inputs={"batch_number": batch_number, "document_count": len(batch)},
                tags=[*RAG_TAGS, "indexing", "embeddings"],
                metadata={"embedding_model": "gemini-embedding-2"},
            ) as run:
                embeddings = await self.embedding_model.aembed_documents(batch)
                run.end(outputs=embedding_outputs(embeddings))

            all_embeddings.extend(embeddings)

            if start + batch_size < len(texts):
                logger.info(
                    "Batch %d complete. Waiting %d seconds before next batch...",
                    batch_number,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)

        return all_embeddings

    @trace_async(
        name="Upsert transcript chunks",
        run_type="chain",
        tags=[*RAG_TAGS, "indexing", "vector-store"],
        process_inputs=embedding_inputs,
    )
    async def store_in_collection(
        self,
        chunk_docs: Sequence[Document],
        name: str
    ) -> int:
        """Embed and upsert documents without blocking the event loop."""
        collection = await self.get_or_create_collection(name)
        add_run_metadata(video_id=name, collection_name=name, chunk_count=len(chunk_docs))

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

        with trace(
            "Upsert to Chroma", run_type="tool",
            inputs={"collection_name": name, "chunk_ids": ids},
            tags=[*RAG_TAGS, "indexing", "vector-store"],
        ) as run:
            await asyncio.to_thread(
                collection.upsert,
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            run.end(outputs={"stored_chunk_count": len(ids)})

        logger.info(
            "Stored %d documents in collection '%s'",
            len(chunk_docs),
            name
        )
        return len(chunk_docs)

    async def get_collection(self, name: str):
        """Fetch a Chroma collection without blocking the event loop."""
        return await asyncio.to_thread(self.client.get_collection, name=name)
