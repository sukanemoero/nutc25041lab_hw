from typing import Optional, overload, Any
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, SparseEmbeddings, RetrievalMode
from pydantic import HttpUrl
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams


class LocalQdrant:
    def qdrant(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID
            if self.sparse_embeddings
            else RetrievalMode.DENSE,
        )

    @overload
    def __init__(
        self,
        *,
        sparse_embeddings: Optional[SparseEmbeddings],
        embeddings: Embeddings,
        dims: int,
        location: str,
        collection_name: str = "NGGYU",
        force_recreate: bool = False,
    ):
        pass

    @overload
    def __init__(
        self,
        *,
        sparse_embeddings: Optional[SparseEmbeddings],
        embeddings: Embeddings,
        dims: int,
        path: str,
        collection_name: str = "NGGYU",
        force_recreate: bool = False,
    ):
        pass

    @overload
    def __init__(
        self,
        *,
        sparse_embeddings: Optional[SparseEmbeddings],
        embeddings: Embeddings,
        dims: int,
        url: HttpUrl,
        port: int,
        api_key: str,
        collection_name: str = "NGGYU",
        force_recreate: bool = False,
    ):
        pass

    def __init__(
        self,
        *,
        embeddings: Embeddings,
        dims: int,
        sparse_embeddings: Optional[SparseEmbeddings] = None,
        location: str = "",
        path: str = "",
        collection_name: str = "NGGYU",
        url: HttpUrl = "http://localhost",
        port: int = 6333,
        api_key: str = None,
        force_recreate: bool = False,
        **kwargs: Any,
    ) -> None:
        if location:
            self.client = QdrantClient(location=location, **kwargs)
        elif path:
            self.client = QdrantClient(path=path, **kwargs)
        else:
            self.client = QdrantClient(
                url=str(url), port=port, api_key=api_key, **kwargs
            )

        suffix = sparse_embeddings.__class__.__name__ if sparse_embeddings else "NORMAL"
        self.collection_name = f"{collection_name}_{dims}_{suffix}"

        self.embeddings = embeddings
        self.sparse_embeddings = sparse_embeddings

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and force_recreate:
            self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            sparse_vectors_config = {}
            if self.sparse_embeddings:
                sparse_vectors_config = {"langchain-sparse": SparseVectorParams()}

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
                sparse_vectors_config=sparse_vectors_config
                if self.sparse_embeddings
                else None,
            )
