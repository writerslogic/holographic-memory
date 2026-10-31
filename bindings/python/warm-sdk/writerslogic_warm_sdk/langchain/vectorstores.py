from typing import Any, Iterable, List, Optional
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from writerslogic_warm_sdk.client import WarmClient


class WarmVectorStore(VectorStore):
    def __init__(
        self,
        embedding: Embeddings,
        url: str = "http://localhost:8080",
        zero_trust_key: str = "",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: int = 30,
    ):
        self._embedding = embedding
        self._client = WarmClient(
            url=url,
            zero_trust_key=zero_trust_key,
            api_key=api_key,
            tenant_id=tenant_id,
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[Iterable[Optional[str]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts_list = list(texts)
        metadata_list = list(metadatas) if metadatas is not None else [{} for _ in texts_list]
        ids_list = list(ids) if ids is not None else [None for _ in texts_list]
        if len(metadata_list) != len(texts_list):
            raise ValueError("metadatas must have the same length as texts")
        if len(ids_list) != len(texts_list):
            raise ValueError("ids must have the same length as texts")
        if any(value is not None and (not isinstance(value, str) or not value) for value in ids_list):
            raise ValueError("each id must be a nonempty string or None")
        if not texts_list:
            return []
        embeddings = self._embedding.embed_documents(texts_list)
        if len(embeddings) != len(texts_list):
            raise ValueError("the embedder must return one embedding per text")
        resolved_ids = [value if value is not None else str(uuid4()) for value in ids_list]
        documents = [
            {"id": doc_id, "text": text, "vector": vector, "metadata": metadata}
            for doc_id, text, vector, metadata in zip(
                resolved_ids, texts_list, embeddings, metadata_list, strict=True
            )
        ]
        self._client.add_documents(documents)
        return resolved_ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        query_embedding = self._embedding.embed_query(query)
        response = self._client.query(query_vector=query_embedding, top_k=k, filter=filter)
        return [
            Document(
                id=match.get("id"),
                page_content=match.get("text", ""),
                metadata=match.get("metadata", {}),
            )
            for match in response.get("matches", [])
        ]

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[Iterable[Optional[str]]] = None,
        **kwargs: Any,
    ) -> "WarmVectorStore":
        store = cls(embedding=embedding, **kwargs)
        try:
            store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        except BaseException:
            store.close()
            raise
        return store
