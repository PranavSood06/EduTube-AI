from app.rag.embeddings.goggle_genai import Embeddings
from app.rag.ingestion.loader import YTLoader
from app.rag.chunking.splitting import Splitter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
loader = YTLoader("https://www.youtube.com/watch?v=AUQJ9eeP-Ls&t=1070s")
transcript = loader.load_transcript()

doc = Document(
    page_content=str(transcript.content),
    metadata={"source": "youtube"}
)
splitter = Splitter()

docs = splitter.split_([doc])
print(type(docs))
texts = [doc.page_content for doc in docs]

def test_embedding():
    embedding_model = Embeddings()
    embeddings = embedding_model.embed_documents(texts)
    assert embeddings is not None 
    assert len(embeddings) == len(texts)