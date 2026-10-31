# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""CrewAI unified-memory storage backend (`crewai.memory.storage.backend.StorageBackend`).

CrewAI's `Memory` embeds text itself and hands the vectors to the backend, so the vector
length must equal the `warm-server --input-dim`. Scope prefixes, categories and metadata
equality are all pushed to the service as equality terms (see `_scope_keys`), so they are
applied before ranking and never starve `limit`.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from crewai.memory.types import MemoryRecord, ScopeInfo

from writerslogic_warm_integrations.core import (
    AsyncWarmBackend,
    Eq,
    WarmBackend,
    WarmConfig,
    Or,
    Record,
)
from writerslogic_warm_integrations.core.filters import Filter, all_of

_M = "crew_m_"  # user metadata keys
_SCOPE = "crew_scope"
_CATS = "crew_categories"
_IMPORTANCE = "crew_importance"
_CREATED = "crew_created_at"
_ACCESSED = "crew_last_accessed"
_SOURCE = "crew_source"
_PRIVATE = "crew_private"
_ANCESTOR = "crew_s_"
_CATEGORY = "crew_c_"


def _digest(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:20]


def normalize_scope(scope: Optional[str]) -> str:
    if not scope or scope.strip("/") == "":
        return "/"
    parts = [p for p in scope.split("/") if p]
    return "/" + "/".join(parts)


def _ancestors(scope: str) -> List[str]:
    parts = [p for p in normalize_scope(scope).split("/") if p]
    return ["/" + "/".join(parts[: i + 1]) for i in range(len(parts))]


def _scope_keys(scope: str) -> Dict[str, bool]:
    return {f"{_ANCESTOR}{_digest(a)}": True for a in _ancestors(scope)}


def similarity(cosine: float) -> float:
    """Map the service's cosine to CrewAI's score scale.

    CrewAI's LanceDB backend reports 1 / (1 + L2 distance). For unit vectors the distance is
    sqrt(2 - 2 cos), so the same formula on the cosine gives comparable scores in (0, 1].
    """
    return 1.0 / (1.0 + math.sqrt(max(0.0, 2.0 - 2.0 * max(-1.0, min(1.0, cosine)))))


def to_record(record: MemoryRecord) -> Record:
    meta: Dict[str, Any] = {
        _SCOPE: normalize_scope(record.scope),
        _CATS: list(record.categories),
        _IMPORTANCE: record.importance,
        _CREATED: record.created_at.isoformat(),
        _ACCESSED: record.last_accessed.isoformat(),
        _SOURCE: record.source or "",
        _PRIVATE: bool(record.private),
    }
    meta.update(_scope_keys(record.scope))
    meta.update({f"{_CATEGORY}{_digest(c)}": True for c in record.categories})
    meta.update({f"{_M}{k}": v for k, v in record.metadata.items()})
    vector = [float(x) for x in record.embedding] if record.embedding else None
    return Record(id=record.id, text=record.content, metadata=meta, vector=vector)


def from_record(rec: Record) -> MemoryRecord:
    m = rec.metadata
    fields: Dict[str, Any] = {
        "id": rec.id,
        "content": rec.text,
        "scope": m.get(_SCOPE, "/"),
        "categories": list(m.get(_CATS, [])),
        "metadata": {k[len(_M):]: v for k, v in m.items() if k.startswith(_M)},
        "importance": float(m.get(_IMPORTANCE, 0.5)),
        "source": m.get(_SOURCE) or None,
        "private": bool(m.get(_PRIVATE, False)),
    }
    if _CREATED in m:
        fields["created_at"] = datetime.fromisoformat(m[_CREATED])
    if _ACCESSED in m:
        fields["last_accessed"] = datetime.fromisoformat(m[_ACCESSED])
    return MemoryRecord(**fields)


def build_filter(
    scope_prefix: Optional[str] = None,
    categories: Optional[Sequence[str]] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Optional[Filter]:
    terms: List[Filter] = []
    if scope_prefix is not None and normalize_scope(scope_prefix) != "/":
        terms.append(Eq(f"{_ANCESTOR}{_digest(normalize_scope(scope_prefix))}", True))
    if categories:
        cats = [Eq(f"{_CATEGORY}{_digest(c)}", True) for c in categories]
        terms.append(cats[0] if len(cats) == 1 else Or(tuple(cats)))
    for key, value in (metadata_filter or {}).items():
        terms.append(Eq(f"{_M}{key}", value))
    return all_of(terms)


class WarmMemoryStorage:
    """`StorageBackend` for `crewai.Memory(storage=...)`.

    Args:
        config: connection and behavior knobs (`WarmConfig`); or pass its fields as keywords.
            `config.embedder` is only used for records that arrive without an embedding.

    Limits (the service has no list or offset endpoint): `list_records`, `get_scope_info`,
    `list_scopes`, `list_categories` and `count` read at most `config.max_fetch` records per
    call (default 100, the server's `--max-top-k`) and emit a `UserWarning` when that
    truncates. `search` and `delete` are exact. `touch_records` is not implemented because
    the service does not return stored vectors, so `last_accessed` is set at save time only.
    """

    def __init__(self, config: Optional[WarmConfig] = None, **kwargs: Any):
        self.config = config or WarmConfig(**kwargs)
        self._sync = WarmBackend(self.config)
        self._async = AsyncWarmBackend(self.config)

    def close(self) -> None:
        self._sync.close()

    # -- writes ---------------------------------------------------------------------------
    def save(self, records: List[MemoryRecord]) -> None:
        if records:
            self._sync.remember([to_record(r) for r in records])

    async def asave(self, records: List[MemoryRecord]) -> None:
        if records:
            await self._async.remember([to_record(r) for r in records])

    def update(self, record: MemoryRecord) -> None:
        self.save([record])

    # -- search ---------------------------------------------------------------------------
    def _hits(self, hits: Sequence[Any], min_score: float) -> List[Tuple[MemoryRecord, float]]:
        out = []
        for h in hits:
            score = similarity(h.score)
            if score >= min_score:
                out.append((from_record(Record(h.id, h.text, h.metadata)), score))
        return out

    def search(
        self,
        query_embedding: List[float],
        scope_prefix: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[MemoryRecord, float]]:
        hits = self._sync.recall(
            vector=[float(x) for x in query_embedding],
            k=min(limit, self.config.max_fetch),
            filter=build_filter(scope_prefix, categories, metadata_filter),
        )
        return self._hits(hits, min_score)

    async def asearch(
        self,
        query_embedding: List[float],
        scope_prefix: Optional[str] = None,
        categories: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[MemoryRecord, float]]:
        hits = await self._async.recall(
            vector=[float(x) for x in query_embedding],
            k=min(limit, self.config.max_fetch),
            filter=build_filter(scope_prefix, categories, metadata_filter),
        )
        return self._hits(hits, min_score)

    # -- reads ----------------------------------------------------------------------------
    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        (rec,) = self._sync.get([record_id])
        return from_record(rec) if rec else None

    def _scan(self, scope_prefix: Optional[str]) -> List[MemoryRecord]:
        page = self._sync.list(build_filter(scope_prefix))
        if page.truncated:
            warnings.warn(
                f"scope listing truncated at max_fetch={self.config.max_fetch} records; "
                "raise WarmConfig.max_fetch together with the server's --max-top-k",
                stacklevel=3,
            )
        return [from_record(r) for r in page.records]

    def list_records(
        self, scope_prefix: Optional[str] = None, limit: int = 200, offset: int = 0
    ) -> List[MemoryRecord]:
        records = sorted(self._scan(scope_prefix), key=lambda r: r.created_at, reverse=True)
        return records[offset : offset + limit]

    def get_scope_info(self, scope: str) -> ScopeInfo:
        scope = normalize_scope(scope)
        records = self._scan(scope)
        child_prefix = "/" if scope == "/" else scope + "/"
        children, categories = set(), set()
        for r in records:
            categories.update(r.categories)
            if r.scope.startswith(child_prefix):
                first = r.scope[len(child_prefix):].split("/", 1)[0]
                if first:
                    children.add(child_prefix + first)
        created = [r.created_at for r in records]
        return ScopeInfo(
            path=scope,
            record_count=len(records),
            categories=sorted(categories),
            oldest_record=min(created) if created else None,
            newest_record=max(created) if created else None,
            child_scopes=sorted(children),
        )

    def list_scopes(self, parent: str = "/") -> List[str]:
        return self.get_scope_info(parent).child_scopes

    def list_categories(self, scope_prefix: Optional[str] = None) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self._scan(scope_prefix):
            for c in r.categories:
                counts[c] = counts.get(c, 0) + 1
        return counts

    def count(self, scope_prefix: Optional[str] = None) -> int:
        return len(self._scan(scope_prefix))

    # -- deletes --------------------------------------------------------------------------
    @staticmethod
    def _eligible(rec: Record, flt: Optional[Filter], older_than: Optional[datetime]) -> bool:
        if flt is not None and not flt.matches(rec.metadata):
            return False
        return older_than is None or datetime.fromisoformat(rec.metadata[_CREATED]) < older_than

    def delete(
        self,
        scope_prefix: Optional[str] = None,
        categories: Optional[List[str]] = None,
        record_ids: Optional[List[str]] = None,
        older_than: Optional[datetime] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> int:
        flt = build_filter(scope_prefix, categories, metadata_filter)
        if record_ids:
            ids = list(dict.fromkeys(record_ids))
            if flt is None and older_than is None:
                return self._sync.forget(ids)
            found = [r for r in self._sync.get(ids) if r is not None]
            return self._sync.forget([r.id for r in found if self._eligible(r, flt, older_than)])
        if older_than is None:
            return self._sync.forget_where(flt)
        removed = 0
        while True:
            page = self._sync.list(flt)
            victims = [r.id for r in page.records if self._eligible(r, None, older_than)]
            if not victims:
                return removed
            removed += self._sync.forget(victims)

    async def adelete(
        self,
        scope_prefix: Optional[str] = None,
        categories: Optional[List[str]] = None,
        record_ids: Optional[List[str]] = None,
        older_than: Optional[datetime] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> int:
        flt = build_filter(scope_prefix, categories, metadata_filter)
        if record_ids:
            ids = list(dict.fromkeys(record_ids))
            if flt is None and older_than is None:
                return await self._async.forget(ids)
            found = [r for r in await self._async.get(ids) if r is not None]
            return await self._async.forget([r.id for r in found if self._eligible(r, flt, older_than)])
        if older_than is None:
            return await self._async.forget_where(flt)
        removed = 0
        while True:
            page = await self._async.list(flt)
            victims = [r.id for r in page.records if self._eligible(r, None, older_than)]
            if not victims:
                return removed
            removed += await self._async.forget(victims)

    def reset(self, scope_prefix: Optional[str] = None) -> None:
        self._sync.forget_where(build_filter(scope_prefix))
