class Splitter:
    def __init__(self,text_splitter,chunk_size:int,chunk_overlap:int):
        self.text_splitter = text_splitter
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self,docs):
        self.split = self.text_splitter(chunk_size = self.chunk_size , chunk_overlap = self.chunk_overlap)
        self.chunks = self.split.split_documents(docs)
        return self.chunks

