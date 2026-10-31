from typing import Any, Dict, List, Optional
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from writerslogic_warm_sdk.client import WarmClient

class WarmDocumentStore:
    def __init__(
        self,
        url: str = "http://localhost:8080",
        zero_trust_key: str = "",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: int = 30,
    ):
        self._client = WarmClient(
            url=url,
            zero_trust_key=zero_trust_key,
            api_key=api_key,
            tenant_id=tenant_id,
            timeout=timeout
        )

    def close(self) -> None:
        self._client.close()

    def count_documents(self) -> int:
        count = self._client.stats().get("documents")
        if type(count) is not int or count < 0:
            raise ValueError("WARM stats returned an invalid document count")
        return count

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        raise NotImplementedError(
            "Haystack document enumeration/filtering requires full document retrieval, "
            "which the WARM HTTP API does not expose"
        )

    def write_documents(
        self, documents: List[Document], policy: Optional[DuplicatePolicy] = DuplicatePolicy.NONE,
    ) -> int:
        if policy is not None and not isinstance(policy, DuplicatePolicy):
            raise TypeError("policy must be a Haystack DuplicatePolicy")
        if policy in (DuplicatePolicy.FAIL, DuplicatePolicy.SKIP):
            raise NotImplementedError(
                "FAIL and SKIP require atomic server-side duplicate handling; "
                "WARM currently supports OVERWRITE (also used for NONE)"
            )
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
