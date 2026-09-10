from app.rag.ingestion.loader import YTLoader

def test_ingestion():
    loader  = YTLoader(video_url="https://www.youtube.com/watch?v=gFx-NjTw3sM")
    transcript = loader.load_transcript()
    assert transcript != ""