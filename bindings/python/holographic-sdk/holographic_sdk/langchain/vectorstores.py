from typing import Any, List, Optional, Iterable
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from holographic_sdk.client import HolographicClient

class HolographicVectorStore(VectorStore):
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
        self._client = HolographicClient(
            url=url,
            zero_trust_key=zero_trust_key,
            api_key=api_key,
            tenant_id=tenant_id,
            timeout=timeout
        )

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any) -> List[str]:
        texts_list = list(texts)
        embeddings = self._embedding.embed_documents(texts_list)
        metadatas = metadatas or [{} for _ in texts_list]
        
        docs = []
        ids = []
        import uuid
        for text, emb, meta in zip(texts_list, embeddings, metadatas):
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            docs.append({
                "id": doc_id,
                "text": text,
                "vector": emb,
                "metadata": meta
            })
            
        self._client.add_documents(docs)
        return ids

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        query_embedding = self._embedding.embed_query(query)
        res = self._client.query(query_vector=query_embedding, top_k=k)
        
        docs = []
        for match in res.get("matches", []):
            docs.append(Document(
                page_content=match.get("text", ""),
                metadata=match.get("metadata", {})
            ))
        return docs

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]] = None, **kwargs: Any) -> "HolographicVectorStore":
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts=texts, metadatas=metadatas)
        return store
