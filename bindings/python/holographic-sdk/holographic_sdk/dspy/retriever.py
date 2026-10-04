from typing import Any, List, Optional, Union
import requests
import hashlib
import dspy

class HolographicRM(dspy.Retrieve):
    def __init__(
        self,
        url: str = "http://localhost:8080",
        zero_trust_key: str = "",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        k: int = 3,
        embedder: Any = None
    ):
        super().__init__(k=k)
        self.url = url
        self.zero_trust_key = zero_trust_key
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.embedder = embedder

    def _encrypt_vector(self, embedding: List[float]) -> List[float]:
        if not self.zero_trust_key or not embedding:
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

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction:
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        
        passages = []
        for query in queries:
            if not self.embedder:
                raise ValueError("An embedder must be provided to HolographicRM")
            
            embedding = self.embedder(query)
            encrypted_query = self._encrypt_vector(embedding)
            
            headers = self._get_headers()
            payload = {
                "query_vector": encrypted_query,
                "top_k": k
            }
            res = requests.post(f"{self.url}/api/v1/query", json=payload, headers=headers)
            res.raise_for_status()
            
            for doc in res.json().get("matches", []):
                passages.append(doc.get("text", ""))
                
        return dspy.Prediction(passages=passages)
