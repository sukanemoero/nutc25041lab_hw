from os import getenv

from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "embed"
if path := getenv("CACHE_DIR", None):
    _CACHE_DIR = Path(path).resolve()
    try:
        _CACHE_DIR.mkdir(parents=True)
        CACHE_DIR = _CACHE_DIR
    except Exception:
        pass
else:
    CACHE_DIR.mkdir(exist_ok=True, parents=True)


class LocalVector:
    def __init__(self, embeddings: Embeddings):
        """
        Initializes the local vector store using ChromaDB.

        Args:
            embeddings: The embedding model instance used to vectorize text.
        """
        self.embeddings = embeddings

        self.vector_db = Chroma(
            collection_name="local_cache_store",
            embedding_function=self.embeddings,
            persist_directory=CACHE_DIR,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )

    async def add_text(self, text: str, metadata: dict = {}) -> None:
        """
        Splits a raw string into chunks and adds them to the vector database.
        """
        chunks = self.text_splitter.split_text(text)

        docs = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]

        await self.vector_db.aadd_documents(docs)
        return docs

    async def search(self, query: str, k=3) -> list[tuple[Document, float]]:
        """
        Performs a similarity search and returns documents along with their
        distance scores (L2, Cosine, etc., depending on Chroma config).

        Returns:
            List of (Document, score) tuples. Lower scores usually mean higher similarity.
        """
        return await self.vector_db.asimilarity_search_with_score(query, k=k)
