import asyncio

from langchain_core.documents import Document

from app.services.rag.store import VectorStore


class FakeClient:
    def get_or_create_collection(self, name):
        return {"name": name}


class FakeEmbeddings:
    async def embed_documents(self, texts):
        return [[float(len(text))] for text in texts]


def test_create_new_collection_uses_configured_client():
    store = VectorStore.__new__(VectorStore)
    store.client = FakeClient()

    collection = store.create_new_collection("video-123")

    assert collection == {"name": "video-123"}


def test_embed_docs_embeds_document_texts():
    store = VectorStore.__new__(VectorStore)
    store.embedding_model = FakeEmbeddings()

    embeddings = asyncio.run(
        store.embed_docs([Document(page_content="one"), Document(page_content="four")])
    )

    assert embeddings == [[3.0], [4.0]]
