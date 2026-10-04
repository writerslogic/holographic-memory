from typing import Any, List, Optional
import requests
import hashlib
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
    MetadataFilters,
)
from llama_index.core.schema import TextNode, BaseNode

class HolographicVectorStore(BasePydanticVectorStore):
    stores_text: bool = True
    is_embedding_query: bool = True
    
    url: str
    zero_trust_key: str
    api_key: Optional[str] = None
    tenant_id: Optional[str] = None

    def __init__(
        self,
        url: str = "http://localhost:8080",
        zero_trust_key: str = "",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            url=url,
            zero_trust_key=zero_trust_key,
            api_key=api_key,
            tenant_id=tenant_id,
            **kwargs
        )

    def _encrypt_vector(self, embedding: List[float]) -> List[float]:
        if not self.zero_trust_key:
            return embedding
        seed = int(hashlib.sha256(self.zero_trust_key.encode()).hexdigest()[:8], 16)
        import random
        rng = random.Random(seed)
        return [val * (1.0 if rng.random() > 0.5 else -1.0) for val in embedding]

    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.tenant_id:
            headers["X-Tenant-ID"] = self.tenant_id
        return headers

    @property
    def client(self) -> Any:
        return self

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        headers = self._get_headers()
        ids = []
        for node in nodes:
            vector = self._encrypt_vector(node.embedding) if node.embedding else []
            payload = {
                "id": node.node_id,
                "text": node.get_content(metadata_mode="all"),
                "vector": vector,
                "metadata": node.metadata
            }
            res = requests.post(f"{self.url}/api/v1/documents", json=payload, headers=headers)
            res.raise_for_status()
            ids.append(node.node_id)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        headers = self._get_headers()
        res = requests.delete(f"{self.url}/api/v1/documents/{ref_doc_id}", headers=headers)
        res.raise_for_status()

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        headers = self._get_headers()
        encrypted_query = self._encrypt_vector(query.query_embedding) if query.query_embedding else []
        
        payload = {
            "query_vector": encrypted_query,
            "top_k": query.similarity_top_k
        }
        res = requests.post(f"{self.url}/api/v1/query", json=payload, headers=headers)
        res.raise_for_status()
        
        results = res.json()
        nodes = []
        similarities = []
        ids = []
        
        for doc in results.get("matches", []):
            node = TextNode(
                id_=doc.get("id"),
                text=doc.get("text", ""),
                metadata=doc.get("metadata", {})
            )
            nodes.append(node)
            similarities.append(doc.get("score", 0.0))
            ids.append(doc.get("id"))
            
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
