from typing import Any, List, Optional, Tuple
import requests
import hashlib
from semantic_kernel.memory.memory_store_base import MemoryStoreBase
from semantic_kernel.memory.memory_record import MemoryRecord

class HolographicMemoryStore(MemoryStoreBase):
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

    async def create_collection_async(self, collection_name: str) -> None:
        pass

    async def get_collections_async(self) -> List[str]:
        return ["default"]

    async def delete_collection_async(self, collection_name: str) -> None:
        pass

    async def does_collection_exist_async(self, collection_name: str) -> bool:
        return True

    async def upsert_async(self, collection_name: str, record: MemoryRecord) -> str:
        headers = self._get_headers()
        vector = self._encrypt_vector(record.embedding.tolist() if record.embedding is not None else [])
        payload = {
            "id": record._id,
            "text": record._text,
            "vector": vector,
            "metadata": {"collection": collection_name}
        }
        res = requests.post(f"{self.url}/api/v1/documents", json=payload, headers=headers)
        res.raise_for_status()
        return record._id

    async def upsert_batch_async(self, collection_name: str, records: List[MemoryRecord]) -> List[str]:
        return [await self.upsert_async(collection_name, r) for r in records]

    async def get_async(self, collection_name: str, key: str, with_embedding: bool = False) -> MemoryRecord:
        raise NotImplementedError

    async def get_batch_async(self, collection_name: str, keys: List[str], with_embeddings: bool = False) -> List[MemoryRecord]:
        raise NotImplementedError

    async def remove_async(self, collection_name: str, key: str) -> None:
        headers = self._get_headers()
        res = requests.delete(f"{self.url}/api/v1/documents/{key}", headers=headers)
        res.raise_for_status()

    async def remove_batch_async(self, collection_name: str, keys: List[str]) -> None:
        for k in keys:
            await self.remove_async(collection_name, k)

    async def get_nearest_matches_async(
        self,
        collection_name: str,
        embedding: Any,
        limit: int = 1,
        min_relevance_score: float = 0.0,
        with_embeddings: bool = False,
    ) -> List[Tuple[MemoryRecord, float]]:
        headers = self._get_headers()
        encrypted_query = self._encrypt_vector(embedding.tolist())
        payload = {
            "query_vector": encrypted_query,
            "top_k": limit
        }
        res = requests.post(f"{self.url}/api/v1/query", json=payload, headers=headers)
        res.raise_for_status()
        
        matches = []
        for doc in res.json().get("matches", []):
            if doc.get("score", 0.0) >= min_relevance_score:
                record = MemoryRecord(
                    id=doc.get("id"),
                    text=doc.get("text", ""),
                    is_reference=False,
                    embedding=None,
                    description=None,
                    additional_metadata=None
                )
                matches.append((record, doc.get("score", 0.0)))
        return matches

    async def get_nearest_match_async(
        self,
        collection_name: str,
        embedding: Any,
        min_relevance_score: float = 0.0,
        with_embedding: bool = False,
    ) -> Tuple[MemoryRecord, float]:
        matches = await self.get_nearest_matches_async(
            collection_name, embedding, limit=1, min_relevance_score=min_relevance_score
        )
        return matches[0] if matches else (None, 0.0)
