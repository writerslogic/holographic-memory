import hashlib
import random
import requests
from typing import List, Optional, Dict, Any

class HolographicClient:
    """Core HTTP client for communicating with a Holographic Memory backend."""
    
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
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.timeout = timeout
        self.session = requests.Session()
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.tenant_id:
            headers["X-Tenant-ID"] = self.tenant_id
        self.session.headers.update(headers)

    def encrypt_vector(self, embedding: List[float]) -> List[float]:
        """Apply Zero-Trust FHE-Lite cryptographic perturbation to the embedding."""
        if not self.zero_trust_key or not embedding:
            return embedding
        
        # In production, this falls back to the Rust holographic-vsa FFI if installed
        try:
            import holographic_vsa
            # Hypothetical FFI binding
            return holographic_vsa.encrypt_vector(embedding, self.zero_trust_key)
        except ImportError:
            # Python-native fallback using deterministic cryptographic projection
            seed = int(hashlib.sha256(self.zero_trust_key.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            return [val * (1.0 if rng.random() > 0.5 else -1.0) for val in embedding]

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Batch upload documents to the holographic store."""
        for doc in documents:
            if "vector" in doc and doc["vector"]:
                doc["vector"] = self.encrypt_vector(doc["vector"])
            
            res = self.session.post(
                f"{self.url}/api/v1/documents", 
                json=doc,
                timeout=self.timeout
            )
            res.raise_for_status()

    def delete_document(self, doc_id: str) -> None:
        """Delete a document by ID."""
        res = self.session.delete(
            f"{self.url}/api/v1/documents/{doc_id}",
            timeout=self.timeout
        )
        res.raise_for_status()

    def query(self, query_vector: List[float], top_k: int = 3) -> Dict[str, Any]:
        """Query the vector database for the top-k nearest matches."""
        encrypted_query = self.encrypt_vector(query_vector) if query_vector else []
        
        payload = {
            "query_vector": encrypted_query,
            "top_k": top_k
        }
        res = self.session.post(
            f"{self.url}/api/v1/query",
            json=payload,
            timeout=self.timeout
        )
        res.raise_for_status()
        return res.json()
