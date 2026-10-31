# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""CrewAI knowledge storage (`crewai.knowledge.storage.base_knowledge_storage.BaseKnowledgeStorage`)."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from crewai.knowledge.storage.base_knowledge_storage import BaseKnowledgeStorage
from crewai.rag.types import SearchResult
from pydantic import ConfigDict, Field, PrivateAttr

from writerslogic_warm_integrations.core import AsyncWarmBackend, WarmBackend, WarmConfig, Record

from .storage import similarity

_USER = "k_"


class WarmKnowledgeStorage(BaseKnowledgeStorage):
    """Knowledge storage for `crewai.knowledge.Knowledge(storage=...)`.

    Documents are stored under a content-hash id, so saving the same text twice is a no-op
    replace. `search` joins multiple query strings with a space, like CrewAI's default
    storage, and applies `score_threshold` to the service's cosine mapped to CrewAI's
    1/(1+distance) scale (see `similarity`). `metadata_filter` is an equality filter on the
    keys the knowledge source attached; `Knowledge` itself attaches none, so use `save_with`
    to add some. `reset` deletes this storage's collection only.

    `config.collection` scopes the knowledge inside the tenant (default `"knowledge"`).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: WarmConfig = Field(default_factory=lambda: WarmConfig(collection="knowledge"))
    _sync: Optional[WarmBackend] = PrivateAttr(default=None)
    _async: Optional[AsyncWarmBackend] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self.config.collection is None:
            self.config.collection = "knowledge"
        self._sync = WarmBackend(self.config)
        self._async = AsyncWarmBackend(self.config)

    @staticmethod
    def _record(text: str, metadata: Optional[Dict[str, Any]]) -> Record:
        rid = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
        return Record(rid, text, {f"{_USER}{k}": v for k, v in (metadata or {}).items()})

    @staticmethod
    def _result(hit: Any) -> SearchResult:
        meta = {k[len(_USER):]: v for k, v in hit.metadata.items() if k.startswith(_USER)}
        return {"id": hit.id, "content": hit.text, "metadata": meta, "score": similarity(hit.score)}

    @staticmethod
    def _filter(metadata_filter: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return {f"{_USER}{k}": v for k, v in metadata_filter.items()} if metadata_filter else None

    def save_with(self, documents: List[str], metadata: Optional[Dict[str, Any]] = None) -> None:
        if documents:
            self._sync.remember([self._record(d, metadata) for d in documents])

    def save(self, documents: List[str]) -> None:
        self.save_with(documents)

    async def asave(self, documents: List[str]) -> None:
        if documents:
            await self._async.remember([self._record(d, None) for d in documents])

    def search(
        self,
        query: List[str],
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.6,
    ) -> List[SearchResult]:
        if not query:
            raise ValueError("Query cannot be empty")
        hits = self._sync.recall(" ".join(query), k=limit, filter=self._filter(metadata_filter))
        return [r for r in map(self._result, hits) if r["score"] >= score_threshold]

    async def asearch(
        self,
        query: List[str],
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.6,
    ) -> List[SearchResult]:
        if not query:
            raise ValueError("Query cannot be empty")
        hits = await self._async.recall(" ".join(query), k=limit, filter=self._filter(metadata_filter))
        return [r for r in map(self._result, hits) if r["score"] >= score_threshold]

    def reset(self) -> None:
        self._sync.forget_where(None)

    async def areset(self) -> None:
        await self._async.forget_where(None)
