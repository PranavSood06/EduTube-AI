import asyncio
import logging
from .store import VectorStore
from langchain_core.documents import Document
from langsmith import trace
from .observability import RAG_TAGS, add_run_metadata, retrieval_inputs, retrieval_outputs, trace_async
logger = logging.getLogger(__name__)

class Retrieval:
    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore

    @trace_async(
        name="Retrieve transcript chunks",
        run_type="retriever",
        tags=[*RAG_TAGS, "retrieval"],
        process_inputs=retrieval_inputs,
        process_outputs=retrieval_outputs,
    )
    async def retrieve(
        self,
        query: str,
        collection_name: str,
        k: int = 5
    ) -> list[Document]:
        if k <= 0:
            raise ValueError("k must be greater than 0")

        logger.info(
            "Retrieval started for %d documents",
            k
        )
        add_run_metadata(video_id=collection_name, retrieval_k=k)
        
        try:
            collection = await self.vectorstore.get_collection(collection_name)
            with trace(
                "Embed search query", run_type="embedding",
                inputs={"query": query}, tags=[*RAG_TAGS, "retrieval", "embeddings"],
                metadata={"embedding_model": "gemini-embedding-2"},
            ) as run:
                query_embedding = await self.vectorstore.embedding_model.aembed_query(query)
                run.end(outputs={"dimensions": len(query_embedding)})
            with trace(
                "Search transcript collection", run_type="retriever",
                inputs={"collection_name": collection_name, "k": k},
                tags=[*RAG_TAGS, "retrieval", "vector-store"],
            ) as run:
                results = await asyncio.to_thread(
                    collection.query,
                    query_embeddings=[query_embedding],
                    n_results=k
                )
                run.end(outputs={
                    "match_count": len(results["documents"][0]),
                    "chunk_ids": [metadata.get("chunk_id") for metadata in results["metadatas"][0] if metadata],
                })
            documents = []
            texts = results["documents"][0]
            metadatas = results["metadatas"][0]
            for text, metadata in zip(texts, metadatas):
                documents.append(
                    Document(
                        page_content=text,
                        metadata=metadata or {}
                    )
                )
            logger.info(
                "Retrieved %d documents from '%s'",
                len(documents),
                collection_name
            )
            add_run_metadata(retrieved_chunk_count=len(documents))
            return documents
        except Exception:
            logger.exception(
                "Retrieval failed for collection '%s'",
                collection_name
            )
            raise
