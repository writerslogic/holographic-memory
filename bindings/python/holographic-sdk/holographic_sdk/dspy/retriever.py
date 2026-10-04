from typing import Any, List, Optional, Union
import dspy
from holographic_sdk.client import HolographicClient

class HolographicRM(dspy.Retrieve):
    def __init__(
        self,
        url: str = "http://localhost:8080",
        zero_trust_key: str = "",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: int = 30,
        k: int = 3,
        embedder: Any = None
    ):
        super().__init__(k=k)
        self._client = HolographicClient(
            url=url,
            zero_trust_key=zero_trust_key,
            api_key=api_key,
            tenant_id=tenant_id,
            timeout=timeout
        )
        self.embedder = embedder

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction:
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        
        passages = []
        for query in queries:
            if not self.embedder:
                raise ValueError("An embedder must be provided to HolographicRM")
            
            embedding = self.embedder(query)
            res = self._client.query(query_vector=embedding, top_k=k)
            
            for doc in res.get("matches", []):
                passages.append(doc.get("text", ""))
                
        return dspy.Prediction(passages=passages)
