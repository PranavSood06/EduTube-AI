from langchain_core.documents import Document

from app.services.rag.chunking import Splitters


def test_recursive_splitter_splits_transcript_into_bounded_chunks():
    transcript = [Document(page_content="word " * 1_000)]

    chunks = Splitters.RecursiveSplitter(transcript)

    assert chunks
    assert len(chunks) < 100
    assert all(chunk.strip() for chunk in chunks)


def test_chunks_to_docs_adds_video_and_chunk_metadata():
    documents = Splitters.chunkstodocs(["first", "second"], "video-123")

    assert [document.page_content for document in documents] == ["first", "second"]
    assert [document.metadata for document in documents] == [
        {"chunk_id": "video-123_0", "video_id": "video-123", "chunk_index": 0},
        {"chunk_id": "video-123_1", "video_id": "video-123", "chunk_index": 1},
    ]
