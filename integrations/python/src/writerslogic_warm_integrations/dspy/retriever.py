# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""DSPy retrievers backed by WARM.

Current DSPy has no retriever base class to subclass: a retriever is a callable returning
`dspy.Prediction(passages=[...])` (see `dspy.retrievers.Embeddings`). `WarmRetriever`
follows that shape, extends it with `scores`, `ids` and `metadatas`, and is a `dspy.Module` so it
gets callbacks, `acall`, and works inside any program. `WarmRM` is the legacy
`dspy.settings.configure(rm=...)` form that `dspy.Retrieve` still reads.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import dspy
from dspy.dsp.utils import dotdict

from writerslogic_warm_integrations.core import (
    AsyncWarmBackend, Embedder, FunctionEmbedder, Hit, WarmBackend, WarmConfig, Record,
)
from writerslogic_warm_integrations.core.filters import FilterLike

EmbedderLike = Union[Embedder, Callable[..., Any]]


def _coerce_embedder(embedder: Optional[EmbedderLike], dim: Optional[int]) -> Optional[Embedder]:
    """Accept a core `Embedder`, a `dspy.Embedder`, or any callable `list[str] -> array-like`."""
    if embedder is None or (hasattr(embedder, "embed_documents") and hasattr(embedder, "embed_query")):
        return embedder  # type: ignore[return-value]
    call = embedder

    def documents(texts: Sequence[str]) -> List[List[float]]:
        out = call(list(texts))
        return out.tolist() if hasattr(out, "tolist") else [list(r) for r in out]

    if dim is None:
        dim = len(documents(["dimension probe"])[0])
    return FunctionEmbedder(documents, dim)


class WarmRetriever(dspy.Module):
    """Retrieve passages from WARM.

    Args:
        k: default number of passages.
        config: an `WarmConfig`; or pass its fields as keyword arguments (`url`, `tenant_id`,
            `api_key`, `collection`, `text_queries`, `zero_trust_key`, `overfetch`, `max_fetch`, ...).
        embedder: encoder, either a core `Embedder` or a `dspy.Embedder` / callable
            (`list[str] -> array`). `input_dim` must then match the service (inferred when a
            callable is given). Default: the lexical `HashingEmbedder` (`input_dim` or 384).
        score_threshold: drop hits scoring below this. Scores are cosine similarity with a client
            embedder and the service's fused rank score under `text_queries`; choose accordingly.
        corpus: optional passages to index on construction.

    Call as `retriever("question")` or `retriever("question", k=5, filter={"topic": "x"})`; the
    result is a `dspy.Prediction` with `passages`, `scores`, `ids`, `metadatas`. Filters are
    equality pushed to the service, with operators (`$gt`, `$in`, ...) applied after an
    over-fetch, so they can return fewer than `k`.
    """

    def __init__(
        self,
        k: int = 5,
        *,
        config: Optional[WarmConfig] = None,
        embedder: Optional[EmbedderLike] = None,
        score_threshold: Optional[float] = None,
        corpus: Optional[Sequence[str]] = None,
        **warm: Any,
    ):
        super().__init__()
        if config is not None and warm:
            raise ValueError("pass either config or keyword arguments, not both")
        if config is None:
            warm["embedder"] = _coerce_embedder(embedder, warm.get("input_dim"))
            if warm["embedder"] is None:
                del warm["embedder"]
            config = WarmConfig(**warm)
        elif embedder is not None:
            raise ValueError("set the embedder on the WarmConfig when passing config")
        self.k = k
        self.score_threshold = score_threshold
        self.config = config
        self._sync: Optional[WarmBackend] = None
        self._async: Optional[AsyncWarmBackend] = None
        if corpus:
            self.add(corpus)

    @property
    def backend(self) -> WarmBackend:
        if self._sync is None:
            self._sync = WarmBackend(self.config)
        return self._sync

    @property
    def async_backend(self) -> AsyncWarmBackend:
        if self._async is None:
            self._async = AsyncWarmBackend(self.config)
        return self._async

    def close(self) -> None:
        if self._sync is not None:
            self._sync.close()
            self._sync = None

    async def aclose(self) -> None:
        if self._async is not None:
            await self._async.aclose()
            self._async = None

    def __deepcopy__(self, memo: Dict[int, Any]) -> "WarmRetriever":
        # dspy optimizers deep-copy programs; open HTTP clients cannot be copied, so the copy opens its own.
        clone = object.__new__(type(self))
        memo[id(self)] = clone
        state = {k: v for k, v in self.__dict__.items() if k not in ("_sync", "_async", "config")}
        import copy

        for key, value in state.items():
            clone.__dict__[key] = copy.deepcopy(value, memo)
        clone.__dict__["config"] = self.config
        clone.__dict__["_sync"] = None
        clone.__dict__["_async"] = None
        return clone

    # ---- writing ------------------------------------------------------------------------

    @staticmethod
    def _records(
        passages: Sequence[str], metadatas: Optional[Sequence[Dict[str, Any]]], ids: Optional[Sequence[str]]
    ) -> List[Record]:
        if metadatas is not None and len(metadatas) != len(passages):
            raise ValueError("metadatas must match passages in length")
        if ids is not None and len(ids) != len(passages):
            raise ValueError("ids must match passages in length")
        return [
            Record(ids[i] if ids else uuid.uuid4().hex, text, dict(metadatas[i]) if metadatas else {})
            for i, text in enumerate(passages)
        ]

    def add(
        self, passages: Sequence[str], metadatas: Optional[Sequence[Dict[str, Any]]] = None, ids: Optional[Sequence[str]] = None
    ) -> List[str]:
        """Index passages (each at most 64 KiB); returns their ids. Re-using an id replaces the passage."""
        return self.backend.remember(self._records(passages, metadatas, ids))

    async def aadd(
        self, passages: Sequence[str], metadatas: Optional[Sequence[Dict[str, Any]]] = None, ids: Optional[Sequence[str]] = None
    ) -> List[str]:
        return await self.async_backend.remember(self._records(passages, metadatas, ids))

    def delete(self, ids: Sequence[str]) -> int:
        return self.backend.forget(ids)

    async def adelete(self, ids: Sequence[str]) -> int:
        return await self.async_backend.forget(ids)

    # ---- retrieval ----------------------------------------------------------------------

    def _prediction(self, hits: List[Hit]) -> dspy.Prediction:
        if self.score_threshold is not None:
            hits = [h for h in hits if h.score >= self.score_threshold]
        return dspy.Prediction(
            passages=[h.text for h in hits],
            scores=[h.score for h in hits],
            ids=[h.id for h in hits],
            metadatas=[h.metadata for h in hits],
        )

    def forward(self, query: str, k: Optional[int] = None, filter: FilterLike = None) -> dspy.Prediction:
        return self._prediction(self.backend.recall(query, k=k or self.k, filter=filter))

    async def aforward(self, query: str, k: Optional[int] = None, filter: FilterLike = None) -> dspy.Prediction:
        return self._prediction(await self.async_backend.recall(query, k=k or self.k, filter=filter))

    def batch_search(self, queries: Sequence[str], k: Optional[int] = None, filter: FilterLike = None) -> List[dspy.Prediction]:
        return [self.forward(q, k=k, filter=filter) for q in queries]

    def as_tool(self, name: str = "search_memory", desc: str = "Search stored passages for text relevant to a query.") -> dspy.Tool:
        """A `dspy.Tool` returning passages as strings, for `dspy.ReAct`."""

        def search(query: str) -> List[str]:
            return self.forward(query).passages

        return dspy.Tool(search, name=name, desc=desc, args={"query": {"type": "string"}}, arg_types={"query": str})


class WarmRM:
    """Legacy retrieval model for `dspy.settings.configure(rm=WarmRM(...))` and `dspy.Retrieve`.

    Wraps a `WarmRetriever`. Returns `dotdict` rows with `long_text` (what `dspy.Retrieve`
    reads) plus `score`, `id` and `metadata`.
    """

    def __init__(self, retriever: Optional[WarmRetriever] = None, **kwargs: Any):
        self.retriever = retriever or WarmRetriever(**kwargs)

    def __call__(self, query_or_queries: Union[str, Sequence[str]], k: Optional[int] = None, **kwargs: Any) -> List[Any]:
        queries = [query_or_queries] if isinstance(query_or_queries, str) else list(query_or_queries)
        out: List[Any] = []
        seen = set()
        for q in queries:
            pred = self.retriever.forward(q, k=k, **kwargs)
            for text, score, rid, meta in zip(pred.passages, pred.scores, pred.ids, pred.metadatas):
                if rid not in seen:
                    seen.add(rid)
                    out.append(dotdict(long_text=text, score=score, id=rid, metadata=meta))
        return out

HolographicRetriever = WarmRetriever
HolographicRM = WarmRM
