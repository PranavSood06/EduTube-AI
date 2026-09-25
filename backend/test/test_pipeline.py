import asyncio
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from app.services.rag.pipeline import Pipeline


class FakeCollection:
    def __init__(self, count=0):
        self._count = count
        self.upsert_calls = []

    def count(self):
        return self._count

    def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)


class FakeClient:
    def __init__(self, collection=None):
        self.collection = collection

    def get_collection(self, name):
        if self.collection is None:
            raise ValueError("collection not found")
        return self.collection


class FakeVectorStore:
    def __init__(self, collection=None):
        self.client = FakeClient(collection)
        self.collection = collection or FakeCollection()
        self.embedded_documents = None

    async def embed_docs(self, documents):
        self.embedded_documents = documents
        return [[0.5] for _ in documents]

    def create_new_collection(self, name):
        assert name == "video-123"
        return self.collection


def make_pipeline(vector_store):
    pipeline = Pipeline.__new__(Pipeline)
    pipeline.vector_store = vector_store
    return pipeline


def test_index_video_fetches_chunks_embeds_and_upserts():
    vector_store = FakeVectorStore()
    pipeline = make_pipeline(vector_store)
    transcript = [Document(page_content="Transcript text")]
    chunks = ["first chunk", "second chunk"]

    async def run_synchronously(function, *args):
        return function(*args)

    with (
        patch("app.services.rag.pipeline.asyncio.to_thread", new=run_synchronously),
        patch("app.services.rag.pipeline.YoutubeScrapper.transcript", return_value=transcript),
        patch("app.services.rag.pipeline.Splitters.RecursiveSplitter", return_value=chunks),
    ):
        indexed_count = asyncio.run(pipeline.index_video("video-123"))

    assert indexed_count == 2
    assert [document.page_content for document in vector_store.embedded_documents] == chunks
    assert vector_store.collection.upsert_calls[0]["ids"] == ["video-123_0", "video-123_1"]


def test_rag_pipeline_retrieves_context_and_generates_answer():
    vector_store = FakeVectorStore(FakeCollection(count=1))
    pipeline = make_pipeline(vector_store)
    pipeline.retrieval = type(
        "Retriever",
        (),
        {
            "retrieve": staticmethod(
                lambda query, collection_name, k: _documents(query, collection_name, k)
            )
        },
    )()

    async def answer(messages):
        prompt_text = messages[0].content
        assert "Relevant transcript passage" in prompt_text
        assert "What does the video explain?" in prompt_text
        return "Grounded answer"

    with patch.object(Pipeline, "generate_answer", new=staticmethod(answer)):
        result = asyncio.run(
            pipeline.rag_pipeline("video-123", "What does the video explain?", k=1)
        )

    assert result == "Grounded answer"


async def _documents(query, collection_name, k):
    assert (query, collection_name, k) == ("What does the video explain?", "video-123", 1)
    return [Document(page_content="Relevant transcript passage")]


def test_rag_pipeline_rejects_empty_question():
    pipeline = make_pipeline(FakeVectorStore())

    with pytest.raises(ValueError, match="question must not be empty"):
        asyncio.run(pipeline.rag_pipeline("video-123", ""))
