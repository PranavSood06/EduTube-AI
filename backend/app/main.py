from fastapi import FastAPI
from app.services.rag.pipeline import Pipeline
from app.schemas.chatRequest import ChatRequest
from app.schemas.videoRequest import VideoRequest
import logging
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.health import check_health
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost:5174/",
    "http://localhost:5173/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

process = Pipeline()

@app.get("/health")
def health():
    return check_health()

@app.post("/videos")
async def process_video(request: VideoRequest):
    logger.info("Storing transcript in vector DB")

    try:
        docs = await process.index_video(request.video_id)

        if docs == 0:
            raise ValueError(
                f"No documents were created for video {request.video_id}"
            )

        return {
            "status": "success",
            "video_id": request.video_id,
            "total_docs": docs
        }

    except Exception:
        logger.exception(
            "Failed to store transcript for video %s",
            request.video_id
        )
        raise


@app.post("/chat")
async def ask_question(request: ChatRequest):
    logger.info("Generating answer")

    try:
        answer = await process.rag_pipeline(
            request.video_id,
            request.query
        )

        return {
            "status": "success",
            "content": answer,
            "length": len(answer)
        }

    except Exception:
        logger.exception(
            "Error in RAG pipeline for video %s",
            request.video_id
        )
        raise