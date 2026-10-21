# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""LlamaIndex vector store backed by Holographic Memory."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import metadata_dict_to_node, node_to_metadata_dict

from ..core import AsyncHmsBackend, Cmp, HmsBackend, HmsConfig, Hit, Record
from ..core.filters import Filter, all_of
from ._filters import to_filter

NODE_ID_KEY = "_li_node_id"
_TEXT_MODES = (
    VectorStoreQueryMode.TEXT_SEARCH,
    VectorStoreQueryMode.SPARSE,
    VectorStoreQueryMode.HYBRID,
    VectorStoreQueryMode.SEMANTIC_HYBRID,
)


def _hit_to_node(hit: Hit) -> BaseNode:
    return metadata_dict_to_node(hit.metadata, text=hit.text)


class HolographicVectorStore(BasePydanticVectorStore):
    """`BasePydanticVectorStore` over `hms-server`.

    Configure through `config` (an `HmsConfig`) or the same keyword arguments
    (`url`, `api_key`, `tenant_id`, `collection`, `embedder`, `input_dim`, `text_queries`,
    `zero_trust_key`, `overfetch`, `max_fetch`, ...). `input_dim` must equal the server's
    `--input-dim`, which must equal your embedding model's dimension.

    Node text, metadata and the serialized node (minus text) are stored; `query` returns
    cosine similarity. Embeddings are not returned by `get_nodes`.
    """

    stores_text: bool = True
    flat_metadata: bool = False
    is_embedding_query: bool = True

    _config: HmsConfig = PrivateAttr()
    _sync: Optional[HmsBackend] = PrivateAttr(default=None)
    _async: Optional[AsyncHmsBackend] = PrivateAttr(default=None)

    def __init__(self, config: Optional[HmsConfig] = None, *, flat_metadata: bool = False, **hms_kwargs: Any) -> None:
        super().__init__(flat_metadata=flat_metadata)
        if config is not None and hms_kwargs:
            raise TypeError("pass either config or keyword arguments, not both")
        self._config = config or HmsConfig(**hms_kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "HolographicVectorStore"

    @property
    def client(self) -> HmsBackend:
        return self.backend

    @property
    def backend(self) -> HmsBackend:
        if self._sync is None:
            self._sync = HmsBackend(self._config)
        return self._sync

    @property
    def abackend(self) -> AsyncHmsBackend:
        """Created on first async use; it is bound to the event loop that first uses it."""
        if self._async is None:
            self._async = AsyncHmsBackend(self._config)
        return self._async

    def close(self) -> None:
        if self._sync is not None:
            self._sync.close()
            self._sync = None

    async def aclose(self) -> None:
        if self._async is not None:
            await self._async.aclose()
            self._async = None

    # --- writes ---------------------------------------------------------------------

    def _records(self, nodes: Sequence[BaseNode]) -> List[Record]:
        out = []
        for node in nodes:
            meta = node_to_metadata_dict(node, remove_text=True, flat_metadata=self.flat_metadata)
            meta[NODE_ID_KEY] = node.node_id
            embedding = node.get_embedding() if node.embedding is not None else None
            out.append(
                Record(
                    id=node.node_id,
                    text=node.get_content(metadata_mode=MetadataMode.NONE),
                    metadata=meta,
                    vector=embedding,
                )
            )
        return out

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        return self.backend.remember(self._records(nodes))

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        return await self.abackend.remember(self._records(nodes))

    # --- reads by id/filter ----------------------------------------------------------

    @staticmethod
    def _delete_filter(ref_doc_id: str) -> Dict[str, Any]:
        return {"ref_doc_id": ref_doc_id}

    def get_nodes(
        self, node_ids: Optional[List[str]] = None, filters: Optional[MetadataFilters] = None
    ) -> List[BaseNode]:
        flt = to_filter(filters)
        if node_ids is not None:
            recs = [r for r in self.backend.get(node_ids) if r is not None]
            recs = [r for r in recs if flt is None or flt.matches(r.metadata)]
        else:
            recs = self.backend.list(flt).records
        return [metadata_dict_to_node(r.metadata, text=r.text) for r in recs]

    async def aget_nodes(
        self, node_ids: Optional[List[str]] = None, filters: Optional[MetadataFilters] = None
    ) -> List[BaseNode]:
        flt = to_filter(filters)
        if node_ids is not None:
            recs = [r for r in await self.abackend.get(node_ids) if r is not None]
            recs = [r for r in recs if flt is None or flt.matches(r.metadata)]
        else:
            recs = (await self.abackend.list(flt)).records
        return [metadata_dict_to_node(r.metadata, text=r.text) for r in recs]

    # --- deletes ---------------------------------------------------------------------

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        self.backend.forget_where(self._delete_filter(ref_doc_id))

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        await self.abackend.forget_where(self._delete_filter(ref_doc_id))

    def delete_nodes(
        self, node_ids: Optional[List[str]] = None, filters: Optional[MetadataFilters] = None, **delete_kwargs: Any
    ) -> None:
        flt = to_filter(filters)
        if node_ids is None and flt is None:
            raise ValueError("delete_nodes needs node_ids or filters; use clear() to remove everything")
        if node_ids is not None and flt is None:
            self.backend.forget(node_ids)
        elif node_ids is not None:
            keep = [r.id for r in self.backend.get(node_ids) if r is not None and flt.matches(r.metadata)]
            self.backend.forget(keep)
        else:
            self.backend.forget_where(flt)

    async def adelete_nodes(
        self, node_ids: Optional[List[str]] = None, filters: Optional[MetadataFilters] = None, **delete_kwargs: Any
    ) -> None:
        flt = to_filter(filters)
        if node_ids is None and flt is None:
            raise ValueError("delete_nodes needs node_ids or filters; use clear() to remove everything")
        if node_ids is not None and flt is None:
            await self.abackend.forget(node_ids)
        elif node_ids is not None:
            found = await self.abackend.get(node_ids)
            await self.abackend.forget([r.id for r in found if r is not None and flt.matches(r.metadata)])
        else:
            await self.abackend.forget_where(flt)

    def clear(self) -> None:
        """Remove every record in this store's collection (other collections are untouched)."""
        self.backend.forget_where(None)

    async def aclear(self) -> None:
        await self.abackend.forget_where(None)

    # --- query -----------------------------------------------------------------------

    def _plan(self, query: VectorStoreQuery) -> Dict[str, Any]:
        mode = query.mode
        text_mode = mode in _TEXT_MODES
        if text_mode and not self._config.text_queries:
            raise ValueError(
                f"query mode {mode.value!r} needs the service's on-device model stages: "
                "construct the store with text_queries=True (hms-server built with local-models)"
            )
        if not text_mode and mode != VectorStoreQueryMode.DEFAULT:
            raise ValueError(f"query mode {mode.value!r} is not supported; use default or, with text_queries, a text/hybrid mode")
        terms: List[Filter] = []
        flt = to_filter(query.filters)
        if flt is not None:
            terms.append(flt)
        if query.node_ids:  # the index passes [] when it keeps no nodes of its own: no restriction
            terms.append(Cmp(NODE_ID_KEY, "in", list(query.node_ids)))
        if query.doc_ids:
            terms.append(Cmp("ref_doc_id", "in", list(query.doc_ids)))
        k = (query.hybrid_top_k or query.similarity_top_k) if text_mode else query.similarity_top_k
        if text_mode:
            if not query.query_str:
                raise ValueError(f"query mode {mode.value!r} needs query_str")
            return dict(query=query.query_str, vector=query.query_embedding, k=k, filter=all_of(terms))
        if query.query_embedding is not None:
            return dict(vector=list(query.query_embedding), k=k, filter=all_of(terms))
        if query.query_str:
            # No embedding supplied: the store's own embedder encodes the text. Only consistent
            # when nodes were added without embeddings too.
            return dict(query=query.query_str, k=k, filter=all_of(terms))
        raise ValueError("query needs query_embedding or query_str")

    @staticmethod
    def _result(hits: List[Hit]) -> VectorStoreQueryResult:
        nodes = [_hit_to_node(h) for h in hits]
        return VectorStoreQueryResult(
            nodes=nodes, similarities=[h.score for h in hits], ids=[n.node_id for n in nodes]
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        return self._result(self.backend.recall(**self._plan(query)))

    async def aquery(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        return self._result(await self.abackend.recall(**self._plan(query)))
