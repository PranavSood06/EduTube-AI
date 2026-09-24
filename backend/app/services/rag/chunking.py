from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
class Splitters:
    @staticmethod
    def RecursiveSplitter(transcript:List[Document]):
        Content = transcript[0].page_content
        x = 100
        while True:
            Splitter = RecursiveCharacterTextSplitter(
                chunk_size = x,
                chunk_overlap = x//4
            )
            chunks = Splitter.split_text(Content)
            if(len(chunks)<100): break 
            x += 100
        return chunks

    def chunkstodocs(chunks: List[str],video_id: str) -> List[Document]:
        documents = []
        for i, chunk_text in enumerate(chunks):
            document = Document(
                page_content=chunk_text,
                metadata={
                    "chunk_id": f"{video_id}_{i}",
                    "video_id": video_id,
                    "chunk_index": i
                }
            )
            documents.append(document)
        return documents
