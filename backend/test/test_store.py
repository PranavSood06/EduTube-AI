import asyncio
from unittest.mock import patch

from langchain_core.documents import Document

from app.services.rag.store import VectorStore


class FakeClient:
    def __init__(self):
        self.collection = None

    def get_or_create_collection(self, name):
        self.collection = FakeCollection(name)
        return self.collection


class FakeCollection:
    def __init__(self, name):
        self.name = name
        self.upsert_calls = []

    def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)


class FakeEmbeddings:
    async def aembed_documents(self, texts):
        return [[float(len(text))] for text in texts]


async def run_synchronously(function, *args, **kwargs):
    return function(*args, **kwargs)


def test_create_new_collection_uses_configured_client():
    store = VectorStore.__new__(VectorStore)
    store.client = FakeClient()

    with patch("app.services.rag.store.asyncio.to_thread", new=run_synchronously):
        collection = asyncio.run(store.create_new_collection("video-123"))

    assert collection.name == "video-123"


def test_embed_docs_embeds_document_texts():
    store = VectorStore.__new__(VectorStore)
    store.embedding_model = FakeEmbeddings()

    embeddings = asyncio.run(
        store.embed_docs([Document(page_content="one"), Document(page_content="four")])
    )

    assert embeddings == [[3.0], [4.0]]


def test_store_in_collection_awaits_embeddings_and_upserts():
    store = VectorStore.__new__(VectorStore)
    store.client = FakeClient()
    store.embedding_model = FakeEmbeddings()

    with patch("app.services.rag.store.asyncio.to_thread", new=run_synchronously):
        stored_count = asyncio.run(
            store.store_in_collection(
                [Document(page_content="one", metadata={"chunk_id": "video-123_0"})],
                "video-123",
            )
        )

    assert stored_count == 1
    assert store.client.collection.upsert_calls[0]["embeddings"] == [[3.0]]
