from typing import Any, Dict, List, Optional
from haystack import Document
from holographic_sdk.client import HolographicClient

class HolographicDocumentStore:
    def __init__(
        self,
        url: str = "http://localhost:8080",
        zero_trust_key: str = "",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: int = 30,
    ):
        self._client = HolographicClient(
            url=url,
            zero_trust_key=zero_trust_key,
            api_key=api_key,
            tenant_id=tenant_id,
            timeout=timeout
        )

    def count_documents(self) -> int:
        return 0

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        return []

    def write_documents(self, documents: List[Document], policy: Any = None) -> int:
        docs = []
        for doc in documents:
            docs.append({
                "id": doc.id,
                "text": doc.content,
                "vector": doc.embedding if doc.embedding else [],
                "metadata": doc.meta
            })
        self._client.add_documents(docs)
        return len(docs)

    def delete_documents(self, document_ids: List[str]) -> None:
        for doc_id in document_ids:
            self._client.delete_document(doc_id)
