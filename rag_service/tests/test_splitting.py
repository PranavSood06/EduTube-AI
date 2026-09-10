from app.rag.chunking.splitting import Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_community.document_loaders import TextLoader

dir = Path(__file__).parent
file_path = dir/"sample.txt"

loader = TextLoader(file_path=str(file_path),autodetect_encoding=True)

docs = loader.load()
# print(docs[0].page_content)

def test_splitter():
    splitter = Splitter(RecursiveCharacterTextSplitter,100,25)
    chunks = splitter.split(docs)
    assert len(chunks)!=0