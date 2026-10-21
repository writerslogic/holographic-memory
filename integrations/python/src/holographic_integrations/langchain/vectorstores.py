# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""LangChain `VectorStore` over Holographic Memory."""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Type, TypeVar

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from ..core import AsyncHmsBackend, Hit, HmsBackend, HmsConfig, Record
from ..core.filters import FilterLike
from ..core.embedders import aembed_documents, aembed_query
from .embeddings import LangChainEmbedder

VST = TypeVar("VST", bound="HolographicVectorStore")


def _document(hit: Hit) -> Document:
    return Document(id=hit.id, page_content=hit.text, metadata=hit.metadata)


def _records(documents: Sequence[Document], ids: Optional[Sequence[str]]) -> List[Record]:
    if ids is not None and len(ids) != len(documents):
        raise ValueError("ids and documents must have the same length")
    out = []
    for i, doc in enumerate(documents):
        rid = ids[i] if ids is not None else (doc.id or str(uuid.uuid4()))
        out.append(Record(id=rid, text=doc.page_content, metadata=dict(doc.metadata)))
    return out


class HolographicVectorStore(VectorStore):
    """Vector store backed by `hms-server`.

    Args:
        embedding: LangChain embeddings used for documents and queries. Optional: without it the
            config's embedder is used (default: the lexical `HashingEmbedder`), or, with
            `text_queries=True`, the service embeds text itself.
        config: Connection and behavior knobs (`HmsConfig`). Individual fields may instead be
            passed as keyword arguments (`url`, `api_key`, `tenant_id`, `collection`,
            `zero_trust_key`, `overfetch`, `max_fetch`, `text_queries`, ...).
        relevance_score_fn: Maps the raw score to a `[0, 1]` relevance for
            `similarity_search_with_relevance_scores`. Default: `(cosine + 1) / 2`, which is only
            meaningful in vector mode; in text mode scores are fused ranks, so pass your own.

    Scores are cosine similarity (higher is better). Filters are LangChain-style dicts (see
    `holographic_integrations.core.filters.parse`); equality terms run in the service, anything
    else is applied to over-fetched candidates, so such a search can return fewer than `k`.
    """

    def __init__(
        self,
        embedding: Optional[Embeddings] = None,
        *,
        config: Optional[HmsConfig] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        **hms_kwargs: Any,
    ) -> None:
        self._embedding = embedding
        if config is not None and hms_kwargs:
            raise ValueError("pass either config or keyword options, not both")
        if embedding is not None:
            adapter = LangChainEmbedder(embedding, (config.input_dim if config else hms_kwargs.get("input_dim")))
            if config is not None:
                config = dataclasses.replace(config, embedder=adapter, input_dim=adapter.dim)
            else:
                config = HmsConfig(**{**hms_kwargs, "embedder": adapter, "input_dim": adapter.dim})
        elif config is None:
            config = HmsConfig(**hms_kwargs)
        self.config = config
        self._relevance_score_fn = relevance_score_fn
        self._sync: Optional[HmsBackend] = None
        self._async: Optional[AsyncHmsBackend] = None

    @property
    def backend(self) -> HmsBackend:
        if self._sync is None:
            self._sync = HmsBackend(self.config)
        return self._sync

    @property
    def abackend(self) -> AsyncHmsBackend:
        if self._async is None:
            self._async = AsyncHmsBackend(self.config)
        return self._async

    def close(self) -> None:
        if self._sync is not None:
            self._sync.close()
            self._sync = None

    async def aclose(self) -> None:
        if self._async is not None:
            await self._async.aclose()
            self._async = None

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    # ---- writes -----------------------------------------------------------------------

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        return self.backend.remember(_records(documents, kwargs.get("ids")))

    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        return await self.abackend.remember(_records(documents, kwargs.get("ids")))

    def add_texts(
        self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, *, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> List[str]:
        texts = list(texts)
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas and texts must have the same length")
        docs = [Document(page_content=t, metadata=(metadatas[i] if metadatas else {})) for i, t in enumerate(texts)]
        return self.add_documents(docs, ids=ids)

    async def aadd_texts(
        self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, *, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> List[str]:
        texts = list(texts)
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas and texts must have the same length")
        docs = [Document(page_content=t, metadata=(metadatas[i] if metadatas else {})) for i, t in enumerate(texts)]
        return await self.aadd_documents(docs, ids=ids)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by `ids`, or every record matching `filter=` (a filter is required without ids)."""
        if ids is not None:
            self.backend.forget(ids)
        elif kwargs.get("filter") is not None:
            self.backend.forget_where(kwargs["filter"])
        else:
            raise ValueError("delete needs ids or filter=; use delete_all() to clear the collection")
        return True

    async def adelete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids is not None:
            await self.abackend.forget(ids)
        elif kwargs.get("filter") is not None:
            await self.abackend.forget_where(kwargs["filter"])
        else:
            raise ValueError("adelete needs ids or filter=; use adelete_all() to clear the collection")
        return True

    def delete_all(self) -> int:
        """Remove every record in this store's collection (all records, up to paging)."""
        return self.backend.forget_where(None)

    async def adelete_all(self) -> int:
        return await self.abackend.forget_where(None)

    # ---- reads ------------------------------------------------------------------------

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        return [Document(id=r.id, page_content=r.text, metadata=r.metadata) for r in self.backend.get(ids) if r]

    async def aget_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        return [Document(id=r.id, page_content=r.text, metadata=r.metadata) for r in await self.abackend.get(ids) if r]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: FilterLike = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        hits = self.backend.recall(query, k=k, filter=filter)
        return [(_document(h), h.score) for h in hits]

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, filter: FilterLike = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        hits = await self.abackend.recall(query, k=k, filter=filter)
        return [(_document(h), h.score) for h in hits]

    def similarity_search(self, query: str, k: int = 4, filter: FilterLike = None, **kwargs: Any) -> List[Document]:
        return [d for d, _ in self.similarity_search_with_score(query, k, filter)]

    async def asimilarity_search(self, query: str, k: int = 4, filter: FilterLike = None, **kwargs: Any) -> List[Document]:
        return [d for d, _ in await self.asimilarity_search_with_score(query, k, filter)]

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, filter: FilterLike = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return [(_document(h), h.score) for h in self.backend.recall(vector=embedding, k=k, filter=filter)]

    async def asimilarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4, filter: FilterLike = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return [(_document(h), h.score) for h in await self.abackend.recall(vector=embedding, k=k, filter=filter)]

    def similarity_search_by_vector(self, embedding: List[float], k: int = 4, filter: FilterLike = None, **kwargs: Any) -> List[Document]:
        return [d for d, _ in self.similarity_search_with_score_by_vector(embedding, k, filter)]

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, filter: FilterLike = None, **kwargs: Any
    ) -> List[Document]:
        return [d for d, _ in await self.asimilarity_search_with_score_by_vector(embedding, k, filter)]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self._relevance_score_fn is not None:
            return self._relevance_score_fn
        return lambda cosine: min(1.0, max(0.0, (cosine + 1.0) / 2.0))

    # ---- maximal marginal relevance ---------------------------------------------------
    # The service does not return stored vectors, so candidates are re-embedded from their text
    # with the configured embedder. Needs a local embedder (not text_queries without one).

    def _mmr(self, query_vec: List[float], hits: List[Hit], cand_vecs: List[List[float]], k: int, lambda_mult: float) -> List[Document]:
        order = maximal_marginal_relevance(_np(query_vec), [_np(v) for v in cand_vecs], lambda_mult=lambda_mult, k=k)
        return [_document(hits[i]) for i in order]

    def _mmr_embedder(self):
        if self.config.embedder is None:
            raise NotImplementedError("MMR needs a local embedder to re-embed candidates; none is configured")
        return self.config.embedder

    def max_marginal_relevance_search_by_vector(
        self, embedding: List[float], k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, filter: FilterLike = None, **kwargs: Any
    ) -> List[Document]:
        embedder = self._mmr_embedder()
        hits = self.backend.recall(vector=embedding, k=min(max(fetch_k, k), self.config.max_fetch), filter=filter)
        if not hits:
            return []
        return self._mmr(embedding, hits, embedder.embed_documents([h.text for h in hits]), k, lambda_mult)

    async def amax_marginal_relevance_search_by_vector(
        self, embedding: List[float], k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, filter: FilterLike = None, **kwargs: Any
    ) -> List[Document]:
        embedder = self._mmr_embedder()
        hits = await self.abackend.recall(vector=embedding, k=min(max(fetch_k, k), self.config.max_fetch), filter=filter)
        if not hits:
            return []
        return self._mmr(embedding, hits, await aembed_documents(embedder, [h.text for h in hits]), k, lambda_mult)

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, filter: FilterLike = None, **kwargs: Any
    ) -> List[Document]:
        vec = self._mmr_embedder().embed_query(query)
        return self.max_marginal_relevance_search_by_vector(vec, k, fetch_k, lambda_mult, filter)

    async def amax_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, filter: FilterLike = None, **kwargs: Any
    ) -> List[Document]:
        vec = await aembed_query(self._mmr_embedder(), query)
        return await self.amax_marginal_relevance_search_by_vector(vec, k, fetch_k, lambda_mult, filter)

    # ---- constructors -----------------------------------------------------------------

    @classmethod
    def from_texts(
        cls: Type[VST], texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]] = None, *,
        ids: Optional[List[str]] = None, **kwargs: Any,
    ) -> VST:
        store = cls(embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    @classmethod
    async def afrom_texts(
        cls: Type[VST], texts: List[str], embedding: Embeddings, metadatas: Optional[List[dict]] = None, *,
        ids: Optional[List[str]] = None, **kwargs: Any,
    ) -> VST:
        store = cls(embedding, **kwargs)
        await store.aadd_texts(texts, metadatas, ids=ids)
        return store


def _np(vector: Sequence[float]):
    import numpy as np

    return np.asarray(vector, dtype=float)
