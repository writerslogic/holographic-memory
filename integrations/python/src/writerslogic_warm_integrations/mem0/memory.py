# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Drop-in `Memory` and `AsyncMemory` with mem0ai 2.x signatures and return shapes, backed by WARM.

    from writerslogic_warm_integrations.mem0 import Memory   # was: from mem0 import Memory

Differences from mem0 are listed in docs/integrations/frameworks/mem0.md.
"""

from __future__ import annotations

import asyncio
import inspect
import warnings
from typing import Any, Dict, List, Mapping, Optional

from writerslogic_warm_integrations.core import And, AsyncWarmBackend, Eq, Filter, WarmBackend, Record
from writerslogic_warm_integrations.core.filters import all_of

from . import _support as s
from ._support import BUCKET_KEY, History, Settings


def _scope_eq(filters: Mapping[str, Any]) -> Filter:
    flt = all_of([Eq(k, v) for k, v in filters.items() if k in s.ENTITY_PARAMS and v])
    return flt if flt is not None else And(())


def _entity_filters(user_id: Optional[str], agent_id: Optional[str], run_id: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)):
        value = s.trim_entity_id(value, key)
        if value:
            out[key] = value
    if not out:
        raise ValueError(
            "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
        )
    return out


class _Base:
    """State and pure logic shared by both flavours."""

    def __init__(self, config: Any = None, **options: Any):
        self.settings = Settings(config, **options)
        self.history_log = History(self.settings.history_db_path)

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any], **options: Any):
        return cls(config_dict, **options)

    # ---- request building

    def _search_filter(self, filters: Optional[Mapping[str, Any]], show_expired: bool) -> Filter:
        eff = s.effective_scope_filters(filters)
        flt = s.translate_filters(eff)
        return all_of([flt] + ([] if show_expired else [s.NotExpired()]))  # type: ignore[list-item]

    def _finish_search(self, hits, top_k: int, threshold: float, explain: bool) -> List[Dict[str, Any]]:
        out = []
        for hit in hits:
            if hit.score < threshold or not hit.text:
                continue
            item = s.to_item(hit.id, hit.text, hit.metadata, hit.score)
            if explain:
                item["score_details"] = {"semantic": hit.score}
            out.append(item)
        return out[:top_k]

    def _rerank(self, query: str, items: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        return self.settings.reranker(query, items, top_k)  # type: ignore[misc]

    def _add_prelude(self, messages, user_id, agent_id, run_id, metadata, timestamp, expiration_date, memory_type):
        s.reject_unsupported_add(timestamp, memory_type, agent_id)
        base = s.entity_metadata(user_id, agent_id, run_id, metadata)
        exp = s.normalize_expiration(expiration_date)
        if exp is not None:
            base["expiration_date"] = exp
        return s.normalize_messages(messages), base

    @staticmethod
    def _scope_of(base: Mapping[str, Any]) -> Dict[str, Any]:
        return {k: base[k] for k in s.ENTITY_PARAMS if base.get(k)}

    def _updated_record(self, memory_id: str, existing: Record, text: Optional[str], metadata, expiration_date) -> Record:
        update_meta = dict(metadata) if metadata is not None else None
        if expiration_date is not s._UNSET:
            update_meta = update_meta or {}
            update_meta["expiration_date"] = s.normalize_expiration(expiration_date)
        meta = dict(existing.metadata)
        prev = existing.text
        if update_meta is not None:
            meta.update(s.strip_identity(update_meta, existing.metadata, "update()"))
        new_text = prev if text is None else text
        meta["hash"] = s._md5(new_text)
        meta["updated_at"] = s.now_iso()
        return Record(memory_id, new_text, meta)

    @staticmethod
    def _check_update_args(text, metadata, expiration_date, data):
        if data is not None:
            warnings.warn("The `data` argument to update() is deprecated; use `text` instead.", DeprecationWarning, stacklevel=3)
            if text is None:
                text = data
        if text is None and metadata is None and expiration_date is s._UNSET:
            raise ValueError("At least one of text, metadata, or expiration_date must be provided.")
        return text

    def _log_add(self, rec: Record, event: Dict[str, Any]) -> None:
        self.history_log.add(
            rec.id, None, rec.text, "ADD", created_at=rec.metadata.get("created_at"), updated_at=rec.metadata.get("updated_at"),
            actor_id=rec.metadata.get("actor_id"), role=rec.metadata.get("role"),
        )

    def _log_update(self, prev: str, rec: Record) -> None:
        self.history_log.add(
            rec.id, prev, rec.text, "UPDATE", created_at=rec.metadata.get("created_at"), updated_at=rec.metadata.get("updated_at"),
            actor_id=rec.metadata.get("actor_id"), role=rec.metadata.get("role"),
        )

    def _log_delete(self, rec: Record) -> None:
        self.history_log.add(
            rec.id, rec.text, None, "DELETE", created_at=rec.metadata.get("created_at"), updated_at=s.now_iso(), is_deleted=1,
            actor_id=rec.metadata.get("actor_id"), role=rec.metadata.get("role"),
        )


class Memory(_Base):
    """Synchronous mem0-compatible memory. `Memory()` talks to `http://127.0.0.1:8080` with a hashing embedder.

    Pass `warm=WarmConfig(...)` (URL, API key, tenant, collection, embedder, text_queries, zero_trust_key, ...) or a
    mem0-style dict through `Memory.from_config`. See the module docs for `extractor`, `reranker`,
    `history_db_path`, `buckets` and `strict_infer`.
    """

    def __init__(self, config: Any = None, **options: Any):
        super().__init__(config, **options)
        self.backend = WarmBackend(self.settings.warm)

    def close(self) -> None:
        self.backend.close()
        self.history_log.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- add

    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[Any] = None,
        expiration_date: Optional[Any] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        msgs, base = self._add_prelude(messages, user_id, agent_id, run_id, metadata, timestamp, expiration_date, memory_type)
        if self.settings.check_infer(infer):
            pairs = self._extract(msgs, base, prompt)
        else:
            pairs = s.verbatim_records(msgs, base, self.settings.buckets)
        if pairs:
            self.backend.remember([r for r, _ in pairs])
            for rec, event in pairs:
                self._log_add(rec, event)
        return {"results": [e for _, e in pairs]}

    def _extract(self, msgs, base, prompt):
        scope = self._scope_of(base)
        query = s.parse_messages(msgs)
        existing = self.backend.recall(query, k=10, filter=scope) if query else []
        facts = self.settings.extractor(msgs, [h.text for h in existing], prompt or self.settings.custom_instructions)  # type: ignore[misc]
        if inspect.isawaitable(facts):
            facts.close() if inspect.iscoroutine(facts) else None
            raise TypeError("the extractor is a coroutine function; use AsyncMemory or a synchronous extractor")
        return s.facts_to_records(facts, {h.metadata.get("hash") for h in existing}, base, self.settings.buckets)

    # ---- read

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        rec = self.backend.get([memory_id])[0]
        return None if rec is None else s.to_item(rec.id, rec.text, rec.metadata)

    def get_all(self, *, filters: Optional[Dict[str, Any]] = None, top_k: int = 20, show_expired: bool = False, **kwargs: Any) -> Dict[str, Any]:
        s.reject_entity_kwargs(kwargs, "get_all")
        s.validate_search_params(top_k=top_k)
        flt = self._search_filter(filters, show_expired)
        records = self._enumerate(flt, top_k)
        return {"results": [s.to_item(r.id, r.text, r.metadata, with_score=False) for r in records]}

    def _enumerate(self, flt: Filter, limit: int) -> List[Record]:
        if limit == 0:
            return []
        page = self.backend.list(flt, limit=limit)
        if not page.truncated:
            return page.records[:limit]
        out: List[Record] = []
        incomplete = False
        for bucket in range(self.settings.buckets):
            part = self.backend.list(all_of([flt, Eq(BUCKET_KEY, bucket)]), limit=limit - len(out))
            incomplete |= part.truncated
            out.extend(part.records)
            if len(out) >= limit:
                return out[:limit]
        if incomplete:
            warnings.warn("get_all: a bucket exceeded the service's per-query cap; results may be incomplete", RuntimeWarning, stacklevel=3)
        return out

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.1,
        rerank: bool = False,
        explain: bool = False,
        reference_date: Optional[Any] = None,
        show_expired: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if reference_date is not None:
            raise ValueError("The 'reference_date' parameter is not supported in the open-source memory SDK.")
        s.reject_entity_kwargs(kwargs, "search")
        s.validate_search_params(threshold=threshold, top_k=top_k)
        query = s.trim_query(query)
        flt = self._search_filter(filters, show_expired)
        if top_k == 0:
            return {"results": []}
        hits = self.backend.recall(query, k=top_k, filter=flt)
        items = self._finish_search(hits, top_k, threshold if threshold is not None else 0.1, explain)
        if rerank and self.settings.reranker and items:
            items = self._rerank(query, items, top_k)
        return {"results": items}

    # ---- write

    def update(self, memory_id, text: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, expiration_date: Any = s._UNSET, data: Optional[str] = None) -> Dict[str, str]:
        text = self._check_update_args(text, metadata, expiration_date, data)
        existing = self.backend.get([memory_id])[0]
        if existing is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        rec = self._updated_record(memory_id, existing, text, metadata, expiration_date)
        self.backend.remember([rec])
        self._log_update(existing.text, rec)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id: str) -> Dict[str, str]:
        existing = self.backend.get([memory_id])[0]
        if existing is None:
            raise ValueError(f"Memory with id {memory_id} not found")
        self.backend.forget([memory_id])
        self._log_delete(existing)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None) -> Dict[str, str]:
        flt = _scope_eq(_entity_filters(user_id, agent_id, run_id))
        seen = set()
        while True:
            page = self.backend.list(flt)
            if not page.records:
                break
            batch = tuple(sorted(r.id for r in page.records))
            if batch in seen:
                break
            seen.add(batch)
            self.backend.forget([r.id for r in page.records])
            for rec in page.records:
                self._log_delete(rec)
        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id: str) -> List[Dict[str, Any]]:
        return self.history_log.get(memory_id)

    def reset(self) -> None:
        """Delete every memory in this backend's collection and clear the history."""
        self.backend.forget_where(None)
        self.history_log.reset()


class AsyncMemory(_Base):
    """Asynchronous counterpart of `Memory`. Same options; coroutine methods."""

    def __init__(self, config: Any = None, **options: Any):
        super().__init__(config, **options)
        self.backend = AsyncWarmBackend(self.settings.warm)

    def close(self) -> None:
        """Close the history database. Use `aclose()` to close the HTTP client too."""
        self.history_log.close()

    async def aclose(self) -> None:
        await self.backend.aclose()
        self.close()

    async def __aenter__(self) -> "AsyncMemory":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[Any] = None,
        expiration_date: Optional[Any] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        msgs, base = self._add_prelude(messages, user_id, agent_id, run_id, metadata, timestamp, expiration_date, memory_type)
        if self.settings.check_infer(infer):
            pairs = await self._extract(msgs, base, prompt)
        else:
            pairs = s.verbatim_records(msgs, base, self.settings.buckets)
        if pairs:
            await self.backend.remember([r for r, _ in pairs])
            for rec, event in pairs:
                await asyncio.to_thread(self._log_add, rec, event)
        return {"results": [e for _, e in pairs]}

    async def _extract(self, msgs, base, prompt):
        scope = self._scope_of(base)
        query = s.parse_messages(msgs)
        existing = await self.backend.recall(query, k=10, filter=scope) if query else []
        facts = self.settings.extractor(msgs, [h.text for h in existing], prompt or self.settings.custom_instructions)  # type: ignore[misc]
        if inspect.isawaitable(facts):
            facts = await facts
        return s.facts_to_records(facts, {h.metadata.get("hash") for h in existing}, base, self.settings.buckets)

    async def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        rec = (await self.backend.get([memory_id]))[0]
        return None if rec is None else s.to_item(rec.id, rec.text, rec.metadata)

    async def get_all(self, *, filters: Optional[Dict[str, Any]] = None, top_k: int = 20, show_expired: bool = False, **kwargs: Any) -> Dict[str, Any]:
        s.reject_entity_kwargs(kwargs, "get_all")
        s.validate_search_params(top_k=top_k)
        flt = self._search_filter(filters, show_expired)
        records = await self._enumerate(flt, top_k)
        return {"results": [s.to_item(r.id, r.text, r.metadata, with_score=False) for r in records]}

    async def _enumerate(self, flt: Filter, limit: int) -> List[Record]:
        if limit == 0:
            return []
        page = await self.backend.list(flt, limit=limit)
        if not page.truncated:
            return page.records[:limit]
        parts = await asyncio.gather(
            *(self.backend.list(all_of([flt, Eq(BUCKET_KEY, b)]), limit=limit) for b in range(self.settings.buckets))
        )
        out = [r for p in parts for r in p.records]
        if any(p.truncated for p in parts):
            warnings.warn("get_all: a bucket exceeded the service's per-query cap; results may be incomplete", RuntimeWarning, stacklevel=3)
        return out[:limit]

    async def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        threshold: float = 0.1,
        rerank: bool = False,
        explain: bool = False,
        reference_date: Optional[Any] = None,
        show_expired: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if reference_date is not None:
            raise ValueError("The 'reference_date' parameter is not supported in the open-source memory SDK.")
        s.reject_entity_kwargs(kwargs, "search")
        s.validate_search_params(threshold=threshold, top_k=top_k)
        query = s.trim_query(query)
        flt = self._search_filter(filters, show_expired)
        if top_k == 0:
            return {"results": []}
        hits = await self.backend.recall(query, k=top_k, filter=flt)
        items = self._finish_search(hits, top_k, threshold if threshold is not None else 0.1, explain)
        if rerank and self.settings.reranker and items:
            items = self._rerank(query, items, top_k)
            if inspect.isawaitable(items):
                items = await items
        return {"results": items}

    async def update(self, memory_id, text: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, expiration_date: Any = s._UNSET, data: Optional[str] = None) -> Dict[str, str]:
        text = self._check_update_args(text, metadata, expiration_date, data)
        existing = (await self.backend.get([memory_id]))[0]
        if existing is None:
            raise ValueError(f"Memory with id {memory_id} not found. Please provide a valid 'memory_id'")
        rec = self._updated_record(memory_id, existing, text, metadata, expiration_date)
        await self.backend.remember([rec])
        await asyncio.to_thread(self._log_update, existing.text, rec)
        return {"message": "Memory updated successfully!"}

    async def delete(self, memory_id: str) -> Dict[str, str]:
        existing = (await self.backend.get([memory_id]))[0]
        if existing is None:
            raise ValueError(f"Memory with id {memory_id} not found")
        await self.backend.forget([memory_id])
        await asyncio.to_thread(self._log_delete, existing)
        return {"message": "Memory deleted successfully!"}

    async def delete_all(self, user_id: Optional[str] = None, agent_id: Optional[str] = None, run_id: Optional[str] = None) -> Dict[str, str]:
        flt = _scope_eq(_entity_filters(user_id, agent_id, run_id))
        seen = set()
        while True:
            page = await self.backend.list(flt)
            if not page.records:
                break
            batch = tuple(sorted(r.id for r in page.records))
            if batch in seen:
                break
            seen.add(batch)
            await self.backend.forget([r.id for r in page.records])
            for rec in page.records:
                await asyncio.to_thread(self._log_delete, rec)
        return {"message": "Memories deleted successfully!"}

    async def history(self, memory_id: str) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.history_log.get, memory_id)

    async def reset(self) -> None:
        await self.backend.forget_where(None)
        await asyncio.to_thread(self.history_log.reset)
