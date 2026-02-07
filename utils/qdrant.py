from typing import overload
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import HttpUrl
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


class LocalQdrant:
    def qdrant(self):
        return QdrantVectorStore(
            self.client, self.collection_name, embedding=self.embeddings
        )

    @overload
    def __init__(
        self,
        *,
        embeddings: Embeddings,
        dims: int,
        location: str,
        collection_name: str = "NGGYU",
    ):
        pass

    @overload
    def __init__(
        self,
        *,
        embeddings: Embeddings,
        dims: int,
        path: str,
        collection_name: str = "NGGYU",
    ):
        pass

    @overload
    def __init__(
        self,
        *,
        embeddings: Embeddings,
        dims: int,
        url: HttpUrl,
        port: int,
        api_key: str,
        collection_name: str = "NGGYU",
    ):
        pass

    def __init__(
        self,
        *,
        embeddings: Embeddings,
        dims: int,
        location: str = "",
        path: str = "",
        collection_name: str = "NGGYU",
        url: HttpUrl = "http://localhost",
        port: int = 6333,
        api_key: str = None,
        **kwargs,
    ) -> None:
        if location:
            self.client = QdrantClient(location)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(url=url, port=port, api_key=api_key)
        self.collection_name = f"{collection_name}_{dims}"
        self.embeddings = embeddings
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
            )
