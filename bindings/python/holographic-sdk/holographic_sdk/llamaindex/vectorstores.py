from typing import Any, List, Optional
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.core.schema import TextNode, BaseNode
from holographic_sdk.client import HolographicClient

class HolographicVectorStore(BasePydanticVectorStore):
    stores_text: bool = True
    is_embedding_query: bool = True
    
    url: str
    zero_trust_key: str
    api_key: Optional[str] = None
    tenant_id: Optional[str] = None
    timeout: int = 30

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def client(self) -> Any:
        return HolographicClient(
            url=self.url,
            zero_trust_key=self.zero_trust_key,
            api_key=self.api_key,
            tenant_id=self.tenant_id,
            timeout=self.timeout
        )

    def add(self, nodes: List[BaseNode], **add_kwargs: Any) -> List[str]:
        docs = []
        ids = []
        for node in nodes:
            ids.append(node.node_id)
            docs.append({
                "id": node.node_id,
                "text": node.get_content(metadata_mode="all"),
                "vector": node.embedding if node.embedding else [],
                "metadata": node.metadata
            })
        self.client.add_documents(docs)
        return ids

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self.client.delete_document(ref_doc_id)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        res = self.client.query(
            query_vector=query.query_embedding if query.query_embedding else [],
            top_k=query.similarity_top_k
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
