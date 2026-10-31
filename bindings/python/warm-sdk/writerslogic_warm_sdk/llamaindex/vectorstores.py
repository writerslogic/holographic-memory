from typing import Any, List, Optional
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilters,
    VectorStoreQueryMode,
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.schema import TextNode, BaseNode, MetadataMode
from writerslogic_warm_sdk.client import WarmClient, WarmError

class WarmVectorStore(BasePydanticVectorStore):
    stores_text: bool = True
    is_embedding_query: bool = True
    
    url: str
    zero_trust_key: str
    api_key: Optional[str] = None
    tenant_id: Optional[str] = None
    timeout: int = 30

    _client: WarmClient = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._client = WarmClient(
            url=self.url,
            zero_trust_key=self.zero_trust_key,
            api_key=self.api_key,
            tenant_id=self.tenant_id,
            timeout=self.timeout
        )

    @property
    def client(self) -> WarmClient:
        return self._client

    def close(self) -> None:
        self._client.close()

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        docs = []
        ids = []
        for node in nodes:
            ids.append(node.node_id)
            docs.append({
                "id": node.node_id,
                "text": node.get_content(metadata_mode=MetadataMode.NONE),
                "vector": node.embedding if node.embedding else [],
                "metadata": node.metadata
            })
        self.client.add_documents(docs)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError("reference-document deletion is unsupported; use delete_nodes with explicit node IDs")

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        if filters is not None or delete_kwargs:
            raise NotImplementedError("node deletion supports explicit IDs only")
        if not isinstance(node_ids, list) or any(not isinstance(key, str) or not key for key in node_ids):
            raise ValueError("node_ids must be a list of nonempty strings")
        for key in node_ids:
            try:
                self.client.delete_document(key)
            except WarmError as error:
                if error.status_code != 404 or error.code != "not_found":
                    raise

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.mode != VectorStoreQueryMode.DEFAULT:
            raise NotImplementedError("WARM supports dense similarity queries only")
        for field in ("doc_ids", "node_ids", "output_fields", "embedding_field", "alpha",
                      "mmr_threshold", "sparse_top_k", "hybrid_top_k"):
            if getattr(query, field, None) is not None:
                raise NotImplementedError(f"WARM does not support the query option {field}")
        filters = {}
        if query.filters is not None:
            if query.filters.condition not in (None, FilterCondition.AND):
                raise NotImplementedError("WARM supports AND metadata filters only")
            for item in query.filters.filters:
                if isinstance(item, MetadataFilters) or item.operator != FilterOperator.EQ:
                    raise NotImplementedError("WARM supports flat equality metadata filters only")
                if item.key in filters:
                    raise ValueError("duplicate equality filters for the same metadata key")
                filters[item.key] = item.value
        res = self.client.query(
            query_vector=query.query_embedding if query.query_embedding else [],
            top_k=query.similarity_top_k,
            filter=filters,
        )
        
        nodes, similarities, ids = [], [], []
        for doc in res.get("matches", []):
            nodes.append(TextNode(
                id_=doc.get("id"),
                text=doc.get("text", ""),
                metadata=doc.get("metadata", {})
            ))
            similarities.append(doc.get("score", 0.0))
            ids.append(doc.get("id"))
            
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
