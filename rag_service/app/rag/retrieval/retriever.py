from langchain_core.documents import Document


class Retriever:
    def __init__(self, vector_store, k: int = 5):
        self.vector_store = vector_store
        self.k = k

    def retrieve(self) -> list[Document]:
        return self.vector_store.as_retriever(
            k=self.k
        )