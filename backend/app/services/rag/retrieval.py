import logging
from .store import VectorStore
from langchain_core.documents import Document
logger = logging.getLogger(__name__)

class Retrieval:
    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore

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
        try:
            collection = self.vectorstore.client.get_collection(
                name=collection_name
            )
            query_embedding = await self.vectorstore.embedding_model.aembed_query(
                query
            )
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
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
            return documents
        except Exception:
            logger.exception(
                "Retrieval failed for collection '%s'",
                collection_name
            )
            raise