from langchain_text_splitters import RecursiveCharacterTextSplitter

class Splitter:
    def __init__(self):
        self.chunk_size = 6000
        self.chunk_overlap = 2000
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def split_text(self,text):
        self.chunks = self.text_splitter.split_text(text)
        return self.chunks
    
    def split_documents(self,docs):
        self.chunks = self.text_splitter.split_documents(docs)
        return self.chunks

