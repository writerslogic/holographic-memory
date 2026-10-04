from typing import Any, Dict, List, Optional
from phi.vectordb.base import VectorDb
from phi.document import Document
from holographic_sdk.client import HolographicClient

class HolographicVectorDb(VectorDb):
    def __init__(
        self,
        url: str = "http://localhost:8080",
        zero_trust_key: str = "",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: int = 30,
        embedder: Any = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embedder = embedder
        self._client = HolographicClient(
            url=url,
            zero_trust_key=zero_trust_key,
            api_key=api_key,
            tenant_id=tenant_id,
            timeout=timeout
        )

    def insert(self, documents: List[Document], **kwargs) -> None:
        docs = []
        for doc in documents:
            embedding = doc.embedding
            if not embedding and self.embedder:
                embedding = self.embedder.get_embedding(doc.content)
            
            docs.append({
                "id": doc.name or str(hash(doc.content)),
                "text": doc.content,
                "vector": embedding if embedding else [],
                "metadata": doc.meta_data
            })
        self._client.add_documents(docs)

    def search(self, query: str, limit: int = 5, **kwargs) -> List[Document]:
        if not self.embedder:
            raise ValueError("Embedder must be provided for searching in HolographicVectorDb")
            
        embedding = self.embedder.get_embedding(query)
        res = self._client.query(query_vector=embedding, top_k=limit)
        
        results = []
        for match in res.get("matches", []):
            results.append(Document(
                content=match.get("text", ""),
                name=match.get("id", ""),
                meta_data=match.get("metadata", {})
            ))
        return results

    def drop(self) -> None:
        pass
