import hashlib
import random
import httpx
from typing import List, Optional, Dict, Any

def _get_headers(api_key: Optional[str], tenant_id: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if tenant_id:
        headers["X-Tenant-ID"] = tenant_id
    return headers

def _encrypt_vector(embedding: List[float], zero_trust_key: str) -> List[float]:
    if not zero_trust_key or not embedding:
        return embedding
    try:
        import holographic_vsa
        return holographic_vsa.encrypt_vector(embedding, zero_trust_key)
    except ImportError:
        seed = int(hashlib.sha256(zero_trust_key.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        return [val * (1.0 if rng.random() > 0.5 else -1.0) for val in embedding]


class HolographicClient:
    """Core sync HTTP client for communicating with a Holographic Memory backend."""
    
    def __init__(
        self,
        url: str = "http://localhost:8080",
        zero_trust_key: str = "",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: int = 30,
    ):
        self.url = url.rstrip('/')
        self.zero_trust_key = zero_trust_key
        headers = _get_headers(api_key, tenant_id)
        self.client = httpx.Client(headers=headers, timeout=timeout)

    def encrypt_vector(self, embedding: List[float]) -> List[float]:
        return _encrypt_vector(embedding, self.zero_trust_key)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Batch upload documents to the holographic store."""
        if not documents:
            return
            
        docs_to_send = []
        for doc in documents:
            if "vector" in doc and doc["vector"]:
                doc["vector"] = self.encrypt_vector(doc["vector"])
            docs_to_send.append(doc)
            
        res = self.client.post(f"{self.url}/api/v1/documents/batch", json=docs_to_send)
        if res.status_code == 404: # Fallback for old API
            for doc in docs_to_send:
                self.client.post(f"{self.url}/api/v1/documents", json=doc).raise_for_status()
        else:
            res.raise_for_status()

    def delete_document(self, doc_id: str) -> None:
        res = self.client.delete(f"{self.url}/api/v1/documents/{doc_id}")
        res.raise_for_status()

    def query(self, query_vector: List[float], top_k: int = 3, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        encrypted_query = self.encrypt_vector(query_vector) if query_vector else []
        payload = {"query_vector": encrypted_query, "top_k": top_k}
        if filter:
            payload["filter"] = filter
            
        res = self.client.post(f"{self.url}/api/v1/query", json=payload)
        res.raise_for_status()
        return res.json()


class AsyncHolographicClient:
    """Core async HTTP client for communicating with a Holographic Memory backend."""
    
    def __init__(
        self,
        url: str = "http://localhost:8080",
        zero_trust_key: str = "",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: int = 30,
    ):
        self.url = url.rstrip('/')
        self.zero_trust_key = zero_trust_key
        headers = _get_headers(api_key, tenant_id)
        self.client = httpx.AsyncClient(headers=headers, timeout=timeout)

    def encrypt_vector(self, embedding: List[float]) -> List[float]:
        return _encrypt_vector(embedding, self.zero_trust_key)

    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return
            
        docs_to_send = []
        for doc in documents:
            if "vector" in doc and doc["vector"]:
                doc["vector"] = self.encrypt_vector(doc["vector"])
            docs_to_send.append(doc)
            
        res = await self.client.post(f"{self.url}/api/v1/documents/batch", json=docs_to_send)
        if res.status_code == 404: # Fallback for old API
            for doc in docs_to_send:
                res_single = await self.client.post(f"{self.url}/api/v1/documents", json=doc)
                res_single.raise_for_status()
        else:
            res.raise_for_status()

    async def delete_document(self, doc_id: str) -> None:
        res = await self.client.delete(f"{self.url}/api/v1/documents/{doc_id}")
        res.raise_for_status()

    async def query(self, query_vector: List[float], top_k: int = 3, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        encrypted_query = self.encrypt_vector(query_vector) if query_vector else []
        payload = {"query_vector": encrypted_query, "top_k": top_k}
        if filter:
            payload["filter"] = filter
            
        res = await self.client.post(f"{self.url}/api/v1/query", json=payload)
        res.raise_for_status()
        return res.json()
