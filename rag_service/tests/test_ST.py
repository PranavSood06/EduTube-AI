from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode([
    "FastAPI is a Python framework.",
    "RAG retrieves relevant documents."
])

# print(embeddings.shape)
