from typing import Any, List, Optional, Tuple
from semantic_kernel.memory.memory_store_base import MemoryStoreBase
from semantic_kernel.memory.memory_record import MemoryRecord
from holographic_sdk.client import HolographicClient

class HolographicMemoryStore(MemoryStoreBase):
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

    async def create_collection_async(self, collection_name: str) -> None:
        pass

    async def get_collections_async(self) -> List[str]:
        return ["default"]

    async def delete_collection_async(self, collection_name: str) -> None:
        pass

    async def does_collection_exist_async(self, collection_name: str) -> bool:
        return True

    async def upsert_async(self, collection_name: str, record: MemoryRecord) -> str:
        doc = {
            "id": record._id,
            "text": record._text,
            "vector": record.embedding.tolist() if record.embedding is not None else [],
            "metadata": {"collection": collection_name}
        }
        self._client.add_documents([doc])
        return record._id

    async def upsert_batch_async(self, collection_name: str, records: List[MemoryRecord]) -> List[str]:
        return [await self.upsert_async(collection_name, r) for r in records]

    async def get_async(self, collection_name: str, key: str, with_embedding: bool = False) -> MemoryRecord:
        raise NotImplementedError

    async def get_batch_async(self, collection_name: str, keys: List[str], with_embeddings: bool = False) -> List[MemoryRecord]:
        raise NotImplementedError

    async def remove_async(self, collection_name: str, key: str) -> None:
        self._client.delete_document(key)

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
        res = self._client.query(query_vector=embedding.tolist(), top_k=limit)
        
        matches = []
        for doc in res.get("matches", []):
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
