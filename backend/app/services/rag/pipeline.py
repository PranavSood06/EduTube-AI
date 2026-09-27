import asyncio
import logging
from typing import Sequence
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
import logging
from ..youtube.transcript import YoutubeScrapper
from .chunking import Splitters
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

    async def index_video(self, video_id: str) -> int:
        """Fetch, chunk, embed, and persist a video's transcript.

        Chunks are upserted so re-indexing the same video does not create
        duplicate records.
        """
        if not video_id or not video_id.strip():
            raise ValueError("video_id must not be empty")

        video_id = video_id.strip()
        logger.info("Indexing transcript for video '%s'", video_id)
        transcript = await asyncio.to_thread(YoutubeScrapper.transcript, video_id)
        if not transcript:
            raise ValueError(f"No transcript is available for video '{video_id}'")

        chunks = Splitters.RecursiveSplitter(transcript)
        if not chunks:
            raise ValueError(f"Transcript for video '{video_id}' contains no text")

        documents = Splitters.chunkstodocs(chunks, video_id)
        indexed_count = await self.vector_store.store_in_collection(documents, video_id)
        logger.info("Indexed %d chunks for video '%s'", indexed_count, video_id)
        return indexed_count

    async def rag_pipeline(self, video_id: str, question: str, *, k: int = 5) -> str:
        """Answer *question* using the transcript associated with *video_id*."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        if k <= 0:
            raise ValueError("k must be greater than 0")

        video_id = video_id.strip()
        try:
            collection = await self.vector_store.get_collection(video_id)
            indexed = await asyncio.to_thread(collection.count) > 0
        except Exception:
            indexed = False

        if not indexed:
            await self.index_video(video_id)

        documents = await self.retrieval.retrieve(question, video_id, k=k)
        if not documents:
            return "The available video content does not provide enough information to answer that question."

        context = "\n\n".join(document.page_content for document in documents)
        messages = prompts.YTchatPrompt().format_messages(
            context=context,
            question=question.strip(),
        )
        return await self.generate_answer(messages)
    
