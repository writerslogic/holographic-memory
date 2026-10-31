# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Letta archival memory on WARM.

`ArchivalMemory` / `AsyncArchivalMemory` expose the same four methods as letta-client's
`client.agents.passages` (create, list, delete, search; letta-client 1.12) and return letta-client's own
`Passage` and `PassageSearchResponse` types, so they can stand in for that resource. They also implement the
semantics of Letta's built-in tools: `archival_memory_insert` (returns None) and `archival_memory_search`
(returns a list of `{"id", "timestamp", "content", "tags"}`; default 5 results; empty query returns an empty list),
as defined in letta 0.16.8 (`function_sets/base.py`, `core_tool_executor.py`, `agent_manager.py`).

Each Letta agent id is its own collection inside the configured tenant.
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
import zlib
from datetime import datetime, time as dtime, timezone, tzinfo
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
from zoneinfo import ZoneInfo

from letta_client.types import Passage
from letta_client.types.agents.passage_search_response import PassageSearchResponse, Result

from writerslogic_warm_integrations.core import AsyncWarmBackend, Cmp, Eq, Filter, WarmBackend, WarmConfig, Hit, Or, Record
from writerslogic_warm_integrations.core.filters import all_of

TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
BUCKET_KEY = "letta_bucket"
DEFAULT_TOP_K = 5  # RETRIEVAL_QUERY_DEFAULT_PAGE_SIZE in letta
DEFAULT_LIST_LIMIT = 100
DEFAULT_BUCKETS = 32

DateLike = Union[str, datetime, None]


def _stamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(TS_FORMAT)


def _parse_dt(value: DateLike, tz: tzinfo, *, end: bool = False) -> Optional[datetime]:
    """ISO 8601 (`2024-01-15` or `2024-01-15T14:30`). Naive values are read in `tz`; a bare end date means end of day."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Invalid datetime format: {value}. Use ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM)") from exc
        if end and len(value) == 10:
            dt = datetime.combine(dt.date(), dtime(23, 59, 59, 999999))
    return dt.replace(tzinfo=tz) if dt.tzinfo is None else dt


class _Options:
    def __init__(self, config: Optional[WarmConfig], timezone_name: str, buckets: int, default_top_k: int):
        if buckets < 1:
            raise ValueError("buckets must be >= 1")
        self.config = config or WarmConfig()
        self.tz = ZoneInfo(timezone_name)
        self.buckets = buckets
        self.default_top_k = default_top_k

    def agent_config(self, agent_id: str) -> WarmConfig:
        if not agent_id:
            raise ValueError("Expected a non-empty value for `agent_id`")
        return dataclasses.replace(self.config, collection=f"letta:{agent_id}" if "/" not in agent_id else agent_id.replace("/", "_"))

    # ---- record <-> Passage

    def new_record(self, agent_id: str, text: str, tags: Optional[Sequence[str]], created_at: DateLike) -> Record:
        pid = f"passage-{uuid.uuid4()}"
        when = _parse_dt(created_at, timezone.utc) or datetime.now(timezone.utc)
        meta: Dict[str, Any] = {"tags": list(tags or []), "created_at": _stamp(when), BUCKET_KEY: zlib.crc32(pid.encode()) % self.buckets}
        return Record(pid, text, meta)

    @staticmethod
    def passage(agent_id: str, rec: Record) -> Passage:
        created = datetime.strptime(rec.metadata["created_at"], TS_FORMAT).replace(tzinfo=timezone.utc)
        return Passage(
            id=rec.id, text=rec.text, tags=list(rec.metadata.get("tags") or []), archive_id=agent_id,
            created_at=created, updated_at=created, is_deleted=False,
        )

    def timestamp(self, rec_meta: Dict[str, Any]) -> str:
        created = datetime.strptime(rec_meta["created_at"], TS_FORMAT).replace(tzinfo=timezone.utc)
        return created.astimezone(self.tz).isoformat()

    def result(self, hit: Hit) -> Result:
        return Result(id=hit.id, content=hit.text, timestamp=self.timestamp(hit.metadata), tags=list(hit.metadata.get("tags") or []))

    def search_filter(
        self, tags: Optional[Sequence[str]], tag_match_mode: str, start: DateLike, end: DateLike
    ) -> Optional[Filter]:
        if tag_match_mode not in ("any", "all"):
            raise ValueError("tag_match_mode must be 'any' or 'all'")
        terms: List[Filter] = []
        if tags:
            parts = tuple(Cmp("tags", "contains", t) for t in tags)
            terms.append(Or(parts) if tag_match_mode == "any" else all_of(list(parts)))  # type: ignore[arg-type]
        s, e = _parse_dt(start, self.tz), _parse_dt(end, self.tz, end=True)
        if s is not None:
            terms.append(Cmp("created_at", "gte", _stamp(s)))
        if e is not None:
            terms.append(Cmp("created_at", "lte", _stamp(e)))
        return all_of(terms)

    def page(self, records: List[Record], agent_id: str, after: Optional[str], before: Optional[str], ascending: Optional[bool],
             limit: Optional[int], search: Optional[str]) -> List[Passage]:
        ordered = sorted(records, key=lambda r: (r.metadata["created_at"], r.id), reverse=ascending is False)
        ids = [r.id for r in ordered]
        if after is not None:
            ordered = ordered[ids.index(after) + 1 :] if after in ids else ordered
            ids = [r.id for r in ordered]
        if before is not None and before in ids:
            ordered = ordered[: ids.index(before)]
        if search:
            needle = search.lower()
            ordered = [r for r in ordered if needle in r.text.lower()]
        return [self.passage(agent_id, r) for r in ordered[: limit if limit is not None else DEFAULT_LIST_LIMIT]]


class ArchivalMemory:
    """Synchronous archival memory, one collection per Letta agent id.

    Knobs: `config` (an `WarmConfig`: URL, key, tenant, embedder, masking, text_queries, ...), `timezone` for naive
    datetimes and result timestamps, `buckets` for `list` enumeration, `default_top_k` for searches without `top_k`.
    """

    def __init__(self, config: Optional[WarmConfig] = None, *, timezone: str = "UTC", buckets: int = DEFAULT_BUCKETS, default_top_k: int = DEFAULT_TOP_K):
        self._o = _Options(config, timezone, buckets, default_top_k)
        self._backends: Dict[str, WarmBackend] = {}

    def _b(self, agent_id: str) -> WarmBackend:
        b = self._backends.get(agent_id)
        if b is None:
            b = self._backends[agent_id] = WarmBackend(self._o.agent_config(agent_id))
        return b

    def close(self) -> None:
        for b in self._backends.values():
            b.close()
        self._backends.clear()

    def __enter__(self) -> "ArchivalMemory":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- client.agents.passages surface

    def create(self, agent_id: str, *, text: str, created_at: DateLike = None, tags: Optional[Sequence[str]] = None, **_ignored: Any) -> List[Passage]:
        rec = self._o.new_record(agent_id, text, tags, created_at)
        self._b(agent_id).remember([rec])
        return [self._o.passage(agent_id, rec)]

    def list(self, agent_id: str, *, after: Optional[str] = None, ascending: Optional[bool] = None, before: Optional[str] = None,
             limit: Optional[int] = None, search: Optional[str] = None, **_ignored: Any) -> List[Passage]:
        return self._o.page(self._all(agent_id), agent_id, after, before, ascending, limit, search)

    def _all(self, agent_id: str) -> List[Record]:
        b = self._b(agent_id)
        page = b.list()
        if not page.truncated:
            return page.records
        out: List[Record] = []
        for bucket in range(self._o.buckets):
            out.extend(b.list(Eq(BUCKET_KEY, bucket)).records)
        return out

    def delete(self, memory_id: str, *, agent_id: str, **_ignored: Any) -> object:
        if not memory_id:
            raise ValueError("Expected a non-empty value for `memory_id`")
        if not self._b(agent_id).forget([memory_id]):
            raise KeyError(f"passage {memory_id} not found for agent {agent_id}")
        return {}

    def search(self, agent_id: str, *, query: str, end_datetime: DateLike = None, start_datetime: DateLike = None,
               tag_match_mode: Literal["any", "all"] = "any", tags: Optional[Sequence[str]] = None, top_k: Optional[int] = None,
               **_ignored: Any) -> PassageSearchResponse:
        if not query or not query.strip():
            return PassageSearchResponse(count=0, results=[])
        flt = self._o.search_filter(tags, tag_match_mode, start_datetime, end_datetime)
        hits = self._b(agent_id).recall(query, k=top_k or self._o.default_top_k, filter=flt)
        return PassageSearchResponse(count=len(hits), results=[self._o.result(h) for h in hits])

    # ---- Letta built-in tool semantics

    def archival_memory_insert(self, agent_id: str, content: str, tags: Optional[List[str]] = None) -> Optional[str]:
        self.create(agent_id, text=content, tags=tags)
        return None

    def archival_memory_search(self, agent_id: str, query: str, tags: Optional[List[str]] = None,
                               tag_match_mode: Literal["any", "all"] = "any", top_k: Optional[int] = None,
                               start_datetime: Optional[str] = None, end_datetime: Optional[str] = None) -> List[Dict[str, Any]]:
        res = self.search(agent_id, query=query, tags=tags, tag_match_mode=tag_match_mode, top_k=top_k,
                          start_datetime=start_datetime, end_datetime=end_datetime)
        return [{"id": r.id, "timestamp": r.timestamp, "content": r.content, "tags": r.tags or []} for r in res.results]


class AsyncArchivalMemory:
    """Asynchronous counterpart of `ArchivalMemory`; same options."""

    def __init__(self, config: Optional[WarmConfig] = None, *, timezone: str = "UTC", buckets: int = DEFAULT_BUCKETS, default_top_k: int = DEFAULT_TOP_K):
        self._o = _Options(config, timezone, buckets, default_top_k)
        self._backends: Dict[str, AsyncWarmBackend] = {}

    def _b(self, agent_id: str) -> AsyncWarmBackend:
        b = self._backends.get(agent_id)
        if b is None:
            b = self._backends[agent_id] = AsyncWarmBackend(self._o.agent_config(agent_id))
        return b

    async def aclose(self) -> None:
        await asyncio.gather(*(b.aclose() for b in self._backends.values()))
        self._backends.clear()

    async def __aenter__(self) -> "AsyncArchivalMemory":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def create(self, agent_id: str, *, text: str, created_at: DateLike = None, tags: Optional[Sequence[str]] = None, **_ignored: Any) -> List[Passage]:
        rec = self._o.new_record(agent_id, text, tags, created_at)
        await self._b(agent_id).remember([rec])
        return [self._o.passage(agent_id, rec)]

    async def list(self, agent_id: str, *, after: Optional[str] = None, ascending: Optional[bool] = None, before: Optional[str] = None,
                   limit: Optional[int] = None, search: Optional[str] = None, **_ignored: Any) -> List[Passage]:
        return self._o.page(await self._all(agent_id), agent_id, after, before, ascending, limit, search)

    async def _all(self, agent_id: str) -> List[Record]:
        b = self._b(agent_id)
        page = await b.list()
        if not page.truncated:
            return page.records
        parts = await asyncio.gather(*(b.list(Eq(BUCKET_KEY, bucket)) for bucket in range(self._o.buckets)))
        return [r for p in parts for r in p.records]

    async def delete(self, memory_id: str, *, agent_id: str, **_ignored: Any) -> object:
        if not memory_id:
            raise ValueError("Expected a non-empty value for `memory_id`")
        if not await self._b(agent_id).forget([memory_id]):
            raise KeyError(f"passage {memory_id} not found for agent {agent_id}")
        return {}

    async def search(self, agent_id: str, *, query: str, end_datetime: DateLike = None, start_datetime: DateLike = None,
                     tag_match_mode: Literal["any", "all"] = "any", tags: Optional[Sequence[str]] = None, top_k: Optional[int] = None,
                     **_ignored: Any) -> PassageSearchResponse:
        if not query or not query.strip():
            return PassageSearchResponse(count=0, results=[])
        flt = self._o.search_filter(tags, tag_match_mode, start_datetime, end_datetime)
        hits = await self._b(agent_id).recall(query, k=top_k or self._o.default_top_k, filter=flt)
        return PassageSearchResponse(count=len(hits), results=[self._o.result(h) for h in hits])

    async def archival_memory_insert(self, agent_id: str, content: str, tags: Optional[List[str]] = None) -> Optional[str]:
        await self.create(agent_id, text=content, tags=tags)
        return None

    async def archival_memory_search(self, agent_id: str, query: str, tags: Optional[List[str]] = None,
                                     tag_match_mode: Literal["any", "all"] = "any", top_k: Optional[int] = None,
                                     start_datetime: Optional[str] = None, end_datetime: Optional[str] = None) -> List[Dict[str, Any]]:
        res = await self.search(agent_id, query=query, tags=tags, tag_match_mode=tag_match_mode, top_k=top_k,
                                start_datetime=start_datetime, end_datetime=end_datetime)
        return [{"id": r.id, "timestamp": r.timestamp, "content": r.content, "tags": r.tags or []} for r in res.results]
