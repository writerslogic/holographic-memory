from typing import Any, Dict, List, Optional
from embedchain.vectordb.base import BaseVectorDB
from holographic_sdk.client import HolographicClient

class HolographicDB(BaseVectorDB):
    def __init__(
        self,
        url: str = "http://localhost:8080",
        zero_trust_key: str = "",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: int = 30,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._client = HolographicClient(
            url=url,
            zero_trust_key=zero_trust_key,
            api_key=api_key,
            tenant_id=tenant_id,
            timeout=timeout
        )

    def _get_or_create_db(self):
        return self._client

    def _get_or_create_collection(self, name):
        pass

    def add(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str], embeddings: List[List[float]], **kwargs) -> Any:
        docs = []
        for doc_id, text, meta, emb in zip(ids, documents, metadatas, embeddings):
            docs.append({
                "id": doc_id,
                "text": text,
                "vector": emb,
                "metadata": meta
            })
        self._client.add_documents(docs)

    def query(self, input_query: List[str], n_results: int, where: Dict[str, Any], citations: bool = False, **kwargs) -> Any:
        if not self.embedder:
            raise ValueError("Embedder must be set on VectorDB")
        
        query_embedding = self.embedder.embed(input_query)[0]
        res = self._client.query(query_vector=query_embedding, top_k=n_results)
        
        results = []
        for match in res.get("matches", []):
            results.append(match.get("text", ""))
        return results

    def count(self) -> int:
        return 0

    def reset(self):
        pass
