"""Shared LangSmith tracing helpers for the RAG service.

Tracing is controlled entirely by the standard ``LANGSMITH_*`` environment
variables.  The decorators remain safe when tracing is disabled, which keeps
local development and tests independent of a LangSmith API key.
"""

import functools
import inspect
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar

from langsmith import set_run_metadata, trace
from langchain_core.documents import Document


RAG_TAGS = ["edutube", "rag"]
P = ParamSpec("P")
T = TypeVar("T")


def trace_async(
    *,
    name: str,
    run_type: str,
    tags: list[str],
    process_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    process_outputs: Callable[[T], dict[str, Any]] | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Trace an async service method without moving it to a worker thread.

    The SDK's ``@traceable`` decorator prepares async runs in a worker thread.
    This wrapper uses the SDK's native ``trace`` context manager instead, keeping
    the service's event-loop and test monkeypatching semantics intact.
    """
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        signature = inspect.signature(func)

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            inputs = {
                key: value for key, value in bound.arguments.items() if key != "self"
            }
            if process_inputs:
                inputs = process_inputs(inputs)
            with trace(name=name, run_type=run_type, inputs=inputs, tags=tags) as run:
                result = await func(*args, **kwargs)
                run.end(outputs=process_outputs(result) if process_outputs else {"result": result})
                return result

        return wrapper

    return decorator


def add_run_metadata(**metadata: Any) -> None:
    """Attach searchable, request-specific fields to the active LangSmith run."""
    set_run_metadata(**{key: value for key, value in metadata.items() if value is not None})


def index_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep index traces compact; transcripts are represented by their chunk count."""
    return {"video_id": inputs.get("video_id")}


def retrieval_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": inputs.get("query"),
        "collection_name": inputs.get("collection_name"),
        "k": inputs.get("k"),
    }


def embedding_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Do not duplicate every transcript chunk in the embedding span input."""
    documents = inputs.get("documents") or inputs.get("chunk_docs") or []
    return {
        "document_count": len(documents),
        "chunk_ids": [document.metadata.get("chunk_id") for document in documents],
        "batch_size": inputs.get("batch_size"),
        "collection_name": inputs.get("name"),
    }


def embedding_outputs(vectors: list[list[float]]) -> dict[str, int]:
    return {
        "embedding_count": len(vectors),
        "dimensions": len(vectors[0]) if vectors else 0,
    }


def retrieval_outputs(documents: list[Document]) -> dict[str, Any]:
    return {
        "retrieved_chunk_count": len(documents),
        "documents": [
            {"page_content": document.page_content, "metadata": document.metadata}
            for document in documents
        ],
    }
