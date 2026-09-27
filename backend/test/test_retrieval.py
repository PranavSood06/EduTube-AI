import asyncio
from unittest.mock import patch

import pytest

from app.services.rag.retrieval import Retrieval


class FakeEmbeddingModel:
    async def aembed_query(self, query):
        assert query == "What is inertia?"
        return [0.1, 0.2]


class FakeCollection:
    def query(self, **kwargs):
        assert kwargs == {"query_embeddings": [[0.1, 0.2]], "n_results": 2}
        return {
            "documents": [["Inertia resists changes in motion."]],
            "metadatas": [[{"chunk_id": "video_0"}]],
        }


class FakeVectorStore:
    embedding_model = FakeEmbeddingModel()

    async def get_collection(self, name):
        assert name == "video"
        return FakeCollection()


def test_retrieve_returns_langchain_documents():
    async def run_synchronously(function, *args, **kwargs):
        return function(*args, **kwargs)

    with patch("app.services.rag.retrieval.asyncio.to_thread", new=run_synchronously):
        documents = asyncio.run(
            Retrieval(FakeVectorStore()).retrieve("What is inertia?", "video", k=2)
        )

    assert len(documents) == 1
    assert documents[0].page_content == "Inertia resists changes in motion."
    assert documents[0].metadata == {"chunk_id": "video_0"}


def test_retrieve_rejects_non_positive_k():
    with pytest.raises(ValueError, match="greater than 0"):
        asyncio.run(Retrieval(FakeVectorStore()).retrieve("question", "video", k=0))
