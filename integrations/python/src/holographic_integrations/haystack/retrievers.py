# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Haystack retriever components for `HolographicDocumentStore`."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import FilterPolicy, apply_filter_policy

from .document_store import HolographicDocumentStore


def _init(self, document_store, filters, top_k, filter_policy, return_embedding) -> None:
    # Plain helpers, not a base class: `@component` rebuilds the class, which breaks zero-argument `super()`.
    if not isinstance(document_store, HolographicDocumentStore):
        raise TypeError("document_store must be an instance of HolographicDocumentStore")
    if top_k <= 0:
        raise ValueError(f"top_k must be greater than 0, but got {top_k}")
    self.document_store = document_store
    self.filters = filters
    self.top_k = top_k
    self.filter_policy = filter_policy
    self.return_embedding = return_embedding


def _resolve(self, filters: Optional[Dict[str, Any]], top_k: Optional[int], return_embedding: Optional[bool]):
    merged = apply_filter_policy(self.filter_policy, self.filters, filters)
    k = self.top_k if top_k is None else top_k
    if k < 0:
        raise ValueError(f"top_k must be greater than or equal to 0, but got {k}")
    return merged, k, self.return_embedding if return_embedding is None else return_embedding


def _revive(cls, data: Dict[str, Any]):
    params = data.get("init_parameters", {})
    if "document_store" in params:
        params["document_store"] = HolographicDocumentStore.from_dict(params["document_store"])
    if "filter_policy" in params:
        params["filter_policy"] = FilterPolicy.from_str(params["filter_policy"])
    return default_from_dict(cls, data)


def _telemetry(self) -> Dict[str, Any]:
    return {"document_store": type(self.document_store).__name__}


@component
class HolographicEmbeddingRetriever:
    """Retrieve documents nearest to a query embedding. `score` is cosine similarity (or `(cos+1)/2` with `scale_score`).

    The embedding must have the store's `embedding_dim` entries. Documents written without an
    embedding were embedded by the store's own embedder, so use that embedder's space for queries.
    """

    def __init__(
        self,
        document_store: HolographicDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = False,
        return_embedding: bool = False,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
    ):
        _init(self, document_store, filters, top_k, filter_policy, return_embedding)
        self.scale_score = scale_score

    _get_telemetry_data = _telemetry

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HolographicEmbeddingRetriever":
        return _revive(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self, document_store=self.document_store.to_dict(), filters=self.filters, top_k=self.top_k,
            scale_score=self.scale_score, return_embedding=self.return_embedding, filter_policy=self.filter_policy.value,
        )

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
    ) -> Dict[str, List[Document]]:
        merged, k, emb = _resolve(self, filters, top_k, return_embedding)
        docs = self.document_store.embedding_retrieval(
            query_embedding, merged, k, self.scale_score if scale_score is None else scale_score, emb
        )
        return {"documents": docs}

    @component.output_types(documents=List[Document])
    async def run_async(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
    ) -> Dict[str, List[Document]]:
        merged, k, emb = _resolve(self, filters, top_k, return_embedding)
        docs = await self.document_store.embedding_retrieval_async(
            query_embedding, merged, k, self.scale_score if scale_score is None else scale_score, emb
        )
        return {"documents": docs}


@component
class HolographicTextRetriever:
    """Retrieve documents for a text query, embedding the query with the store's embedder or letting the service do it.

    With a service running on-device model stages and `text_queries=True` on the store, this is the
    service's hybrid BM25 + embedding search and `score` is the fused rank score, not a cosine.
    """

    def __init__(
        self,
        document_store: HolographicDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        return_embedding: bool = False,
        filter_policy: FilterPolicy = FilterPolicy.REPLACE,
    ):
        _init(self, document_store, filters, top_k, filter_policy, return_embedding)

    _get_telemetry_data = _telemetry

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HolographicTextRetriever":
        return _revive(cls, data)

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self, document_store=self.document_store.to_dict(), filters=self.filters, top_k=self.top_k,
            return_embedding=self.return_embedding, filter_policy=self.filter_policy.value,
        )

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        return_embedding: Optional[bool] = None,
    ) -> Dict[str, List[Document]]:
        merged, k, emb = _resolve(self, filters, top_k, return_embedding)
        return {"documents": self.document_store.text_retrieval(query, merged, k, emb)}

    @component.output_types(documents=List[Document])
    async def run_async(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        return_embedding: Optional[bool] = None,
    ) -> Dict[str, List[Document]]:
        merged, k, emb = _resolve(self, filters, top_k, return_embedding)
        return {"documents": await self.document_store.text_retrieval_async(query, merged, k, emb)}
