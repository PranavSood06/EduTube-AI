import asyncio
import logging
from typing import Sequence
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langsmith import trace
from ..youtube.transcript import YoutubeScrapper
from .chunking import Splitters
from .observability import RAG_TAGS, add_run_metadata, index_inputs, trace_async
from .prompt import prompts
from .retrieval import Retrieval
from .store import VectorStore
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Pipeline:
    """Index video transcripts and answer questions grounded in their chunks."""

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self.vector_store = vector_store or VectorStore()
        self.retrieval = Retrieval(self.vector_store)

    @staticmethod
    @trace_async(
        name="Generate grounded answer",
        run_type="chain",
        tags=[*RAG_TAGS, "generation"],
    )
    async def generate_answer(chat: Sequence[BaseMessage]) -> str:
        """Generate an answer from already-rendered chat messages."""
        logger.info("Generating RAG response")
        try:
            model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
            answer = await model.ainvoke(list(chat))
            content = answer.content
            return content if isinstance(content, str) else str(content)
        except Exception:
            logger.exception("Error generating RAG response")
            raise

    @trace_async(
        name="Index YouTube transcript",
        run_type="chain",
        tags=[*RAG_TAGS, "indexing"],
        process_inputs=index_inputs,
    )
    async def index_video(self, video_id: str) -> int:
        """Fetch, chunk, embed, and persist a video's transcript.

        Chunks are upserted so re-indexing the same video does not create
        duplicate records.
        """
        if not video_id or not video_id.strip():
            raise ValueError("video_id must not be empty")

        video_id = video_id.strip()
        add_run_metadata(video_id=video_id, operation="index_video")
        logger.info("Indexing transcript for video '%s'", video_id)
        with trace(
            "Fetch YouTube transcript", run_type="tool",
            inputs={"video_id": video_id}, tags=[*RAG_TAGS, "transcript"],
        ) as run:
            transcript = await asyncio.to_thread(YoutubeScrapper.transcript, video_id)
            run.end(outputs={"transcript_document_count": len(transcript) if transcript else 0})
        if not transcript:
            raise ValueError(f"No transcript is available for video '{video_id}'")

        with trace(
            "Chunk transcript", run_type="chain",
            inputs={"video_id": video_id, "transcript_document_count": len(transcript)},
            tags=[*RAG_TAGS, "chunking"],
        ) as run:
            chunks = Splitters.RecursiveSplitter(transcript)
            run.end(outputs={"chunk_count": len(chunks)})
        if not chunks:
            raise ValueError(f"Transcript for video '{video_id}' contains no text")

        documents = Splitters.chunkstodocs(chunks, video_id)
        add_run_metadata(chunk_count=len(documents))
        indexed_count = await self.vector_store.store_in_collection(documents, video_id)
        logger.info("Indexed %d chunks for video '%s'", indexed_count, video_id)
        return indexed_count

    @trace_async(
        name="Answer video question",
        run_type="chain",
        tags=[*RAG_TAGS, "chat"],
    )
    async def rag_pipeline(self, video_id: str, question: str, *, k: int = 5) -> str:
        """Answer *question* using the transcript associated with *video_id*."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        if k <= 0:
            raise ValueError("k must be greater than 0")

        video_id = video_id.strip()
        question = question.strip()
        add_run_metadata(
            video_id=video_id,
            retrieval_k=k,
            operation="rag_pipeline",
        )
        with trace(
            "Check video index", run_type="tool",
            inputs={"video_id": video_id}, tags=[*RAG_TAGS, "indexing"],
        ) as run:
            try:
                collection = await self.vector_store.get_collection(video_id)
                indexed = await asyncio.to_thread(collection.count) > 0
            except Exception:
                indexed = False
            run.end(outputs={"indexed": indexed})

        if not indexed:
            await self.index_video(video_id)
        add_run_metadata(indexed_during_request=not indexed)

        documents = await self.retrieval.retrieve(question, video_id, k=k)
        add_run_metadata(retrieved_chunk_count=len(documents))
        if not documents:
            return "The available video content does not provide enough information to answer that question."

        with trace(
            "Build grounded prompt", run_type="prompt",
            inputs={
                "question": question,
                "chunk_ids": [document.metadata.get("chunk_id") for document in documents],
            },
            tags=[*RAG_TAGS, "prompt"],
        ) as run:
            context = "\n\n".join(document.page_content for document in documents)
            messages = prompts.YTchatPrompt().format_messages(
                context=context,
                question=question,
            )
            run.end(outputs={"message_count": len(messages), "context_length": len(context)})
        return await self.generate_answer(messages)
    
