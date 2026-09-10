from app.rag.chunking.splitting import Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from app.rag.vector_store.chroma import Store


def test_vector_store():
    dir = Path(__file__).parent
    file_path = dir/"sample.txt"

    loader = TextLoader(file_path=str(file_path),autodetect_encoding=True)

    docs = loader.load()

    splitter = Splitter(RecursiveCharacterTextSplitter,500,25)
    chunks = splitter.split(docs)

    store = Store("test")
    result = store.add_documents(chunks)
    assert len(result) == len(chunks)




    
# print(chunks)
# https://www.youtube.com/watch?v=UwbASFlIS7A