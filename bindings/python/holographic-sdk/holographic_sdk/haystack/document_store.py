from typing import Any, Dict, List, Optional
import requests
import hashlib
from haystack import Document
from haystack.document_stores.types import DocumentStore

class HolographicDocumentStore:
    def __init__(
        self,
        url: str = "http://localhost:8080",
        zero_trust_key: str = "",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ):
        self.url = url
        self.zero_trust_key = zero_trust_key
        self.api_key = api_key
        self.tenant_id = tenant_id

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

    def count_documents(self) -> int:
        return 0  # To be implemented on the backend API

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        return []

    def write_documents(self, documents: List[Document], policy: Any = None) -> int:
        headers = self._get_headers()
        count = 0
        for doc in documents:
            vector = self._encrypt_vector(doc.embedding) if doc.embedding else []
            payload = {
                "id": doc.id,
                "text": doc.content,
                "vector": vector,
                "metadata": doc.meta
            }
            res = requests.post(f"{self.url}/api/v1/documents", json=payload, headers=headers)
            res.raise_for_status()
            count += 1
        return count

    def delete_documents(self, document_ids: List[str]) -> None:
        headers = self._get_headers()
        for doc_id in document_ids:
            res = requests.delete(f"{self.url}/api/v1/documents/{doc_id}", headers=headers)
            res.raise_for_status()
