# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""LangGraph `BaseStore` over Holographic Memory.

Layout. Every item is one record whose text is the JSON value; its metadata carries the
namespace as per-level keys (`lg_ns0`, `lg_ns1`, ...), `lg_depth`, `lg_key`, timestamps, and
every short scalar field of the value as `v_<field>`. Namespace prefixes and scalar equality
filters therefore run in the service. A second record per namespace (`lg_marker`) makes
`list_namespaces` possible without scanning items.

Without a query, results are sorted by (namespace, key) after fetching at most `max_fetch`
matches, so a search over more matches than that is incomplete. Limits: a value must fit the service's 64 KiB text limit; `offset + limit` of a search and the
number of namespaces returned are capped by the service's `top_k` ceiling (`max_fetch`);
no TTL; with several index fields the texts are joined into one vector (LangGraph's
in-memory store keeps one vector per field).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langgraph.store.base import (
    BaseStore, GetOp, Item, ListNamespacesOp, MatchCondition, Op, PutOp, Result, SearchItem, SearchOp,
)
from langgraph.store.base.embed import ensure_embeddings, get_text_at_path

from ..core import AsyncHmsBackend, HmsBackend, HmsConfig, Record
from ..core.embedders import Embedder, aembed_documents
from ..core.filters import And, Eq, Filter
from ..langchain.embeddings import LangChainEmbedder

_MAX_DEPTH = 16
_SCALAR_LIMIT = 256


def _item_id(ns: Tuple[str, ...], key: str) -> str:
    return "lgi" + hashlib.sha256(json.dumps([list(ns), key]).encode()).hexdigest()


def _marker_id(ns: Tuple[str, ...]) -> str:
    return "lgn" + hashlib.sha256(json.dumps(list(ns)).encode()).hexdigest()


def _ns_eq(ns: Sequence[str]) -> List[Filter]:
    return [Eq(f"lg_ns{i}", part) for i, part in enumerate(ns) if part != "*"]


def _namespace(meta: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(meta[f"lg_ns{i}"] for i in range(meta["lg_depth"]))


def _pushable(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int)) or (isinstance(value, str) and len(value) <= _SCALAR_LIMIT)


def _compare(item_value: Any, flt: Any) -> bool:
    if isinstance(flt, dict):
        if any(k.startswith("$") for k in flt):
            return all(_operator(item_value, k, v) for k, v in flt.items())
        return isinstance(item_value, dict) and all(_compare(item_value.get(k), v) for k, v in flt.items())
    if isinstance(flt, (list, tuple)):
        return (
            isinstance(item_value, (list, tuple))
            and len(item_value) == len(flt)
            and all(_compare(a, b) for a, b in zip(item_value, flt))
        )
    return item_value == flt


def _operator(value: Any, op: str, operand: Any) -> bool:
    if op == "$eq":
        return value == operand
    if op == "$ne":
        return value != operand
    try:
        a, b = float(value), float(operand)
    except (TypeError, ValueError):
        return False
    if op == "$gt":
        return a > b
    if op == "$gte":
        return a >= b
    if op == "$lt":
        return a < b
    if op == "$lte":
        return a <= b
    raise ValueError(f"Unsupported operator: {op}")


class _Namespace(Filter):
    """Client-side namespace condition (suffix matches and wildcards beyond the pushed levels)."""

    def __init__(self, conditions: Sequence[MatchCondition]):
        self.conditions = conditions

    def matches(self, metadata) -> bool:
        ns = _namespace(metadata)
        return all(_matches_ns(c, ns) for c in self.conditions)


def _matches_ns(cond: MatchCondition, ns: Tuple[str, ...]) -> bool:
    path = cond.path
    if len(ns) < len(path):
        return False
    pairs = zip(ns, path) if cond.match_type == "prefix" else zip(reversed(ns), reversed(path))
    return all(p == "*" or n == p for n, p in pairs)


def _iso(ts: datetime) -> str:
    return ts.isoformat()


class HolographicStore(BaseStore):
    """Persistent key-value store with optional semantic search.

    Args:
        index: LangGraph `IndexConfig` (`embed`, `dims`, `fields`). Without it semantic search is
            off, as in every `BaseStore`: `query` is ignored and `index` arguments to `put` have no
            effect. `embed` may be a LangChain `Embeddings`, a function, or a provider string.
        config / **hms_kwargs: `HmsConfig` knobs (`url`, `api_key`, `tenant_id`, `collection`, ...).
            The default collection is `langgraph-store`.
    """

    supports_ttl = False

    def __init__(self, index: Optional[Dict[str, Any]] = None, *, config: Optional[HmsConfig] = None, **hms_kwargs: Any) -> None:
        if config is not None and hms_kwargs:
            raise ValueError("pass either config or keyword options, not both")
        self.index_config = dict(index) if index else None
        self._fields: List[str] = list((index or {}).get("fields") or ["$"])
        self._embedder: Optional[Embedder] = None
        if index and index.get("embed") is not None:
            self._embedder = LangChainEmbedder(ensure_embeddings(index["embed"]), index.get("dims"))
        if config is None:
            hms_kwargs.setdefault("collection", "langgraph-store")
            if self._embedder is not None:
                hms_kwargs.update(embedder=self._embedder, input_dim=self._embedder.dim)
            config = HmsConfig(**hms_kwargs)
        elif self._embedder is not None:
            import dataclasses

            config = dataclasses.replace(config, embedder=self._embedder, input_dim=self._embedder.dim)
        self.config = config
        self._const = [1.0] + [0.0] * (config.input_dim - 1)
        self._sync: Optional[HmsBackend] = None
        self._async: Optional[AsyncHmsBackend] = None

    # ---- lifecycle --------------------------------------------------------------------

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

    # ---- encoding ---------------------------------------------------------------------

    def _index_text(self, value: Dict[str, Any], index: Any) -> Optional[str]:
        """The text to embed, or None when the item is not indexed."""
        if self._embedder is None or index is False:
            return None
        fields = index if isinstance(index, list) else self._fields
        parts: List[str] = []
        for path in fields:
            parts.extend(get_text_at_path(value, path))
        return "\n".join(parts) if parts else None

    def _item_record(self, op: PutOp, created: datetime, now: datetime, vector: Optional[List[float]], indexed: bool) -> Record:
        ns = tuple(op.namespace)
        if len(ns) > _MAX_DEPTH:
            raise ValueError(f"namespace depth is limited to {_MAX_DEPTH}")
        meta: Dict[str, Any] = {f"lg_ns{i}": part for i, part in enumerate(ns)}
        meta.update(lg_depth=len(ns), lg_key=op.key, lg_marker=False, lg_indexed=indexed, lg_created=_iso(created), lg_updated=_iso(now))
        for field, value in op.value.items():
            if _pushable(value):
                meta[f"v_{field}"] = value
        return Record(_item_id(ns, op.key), json.dumps(op.value, ensure_ascii=False), meta, vector)

    def _marker_record(self, ns: Tuple[str, ...]) -> Record:
        meta: Dict[str, Any] = {f"lg_ns{i}": part for i, part in enumerate(ns)}
        meta.update(lg_depth=len(ns), lg_marker=True)
        return Record(_marker_id(ns), "/".join(ns), meta, self._const)

    @staticmethod
    def _item(meta: Dict[str, Any], text: str) -> Item:
        return Item(
            value=json.loads(text), key=meta["lg_key"], namespace=_namespace(meta),
            created_at=datetime.fromisoformat(meta["lg_created"]), updated_at=datetime.fromisoformat(meta["lg_updated"]),
        )

    @staticmethod
    def _search_item(meta: Dict[str, Any], text: str, score: Optional[float]) -> SearchItem:
        return SearchItem(
            _namespace(meta), meta["lg_key"], json.loads(text),
            datetime.fromisoformat(meta["lg_created"]), datetime.fromisoformat(meta["lg_updated"]), score,
        )

    def _search_plan(self, op: SearchOp) -> Tuple[Filter, Any, int]:
        terms: List[Filter] = [Eq("lg_marker", False), *_ns_eq(op.namespace_prefix)]
        flt = op.filter or {}
        for field, wanted in flt.items():
            if _pushable(wanted) and not isinstance(wanted, bool):
                terms.append(Eq(f"v_{field}", wanted))
        semantic = bool(op.query) and self._embedder is not None
        if semantic:
            terms.append(Eq("lg_indexed", True))
        window = op.offset + op.limit
        if window > self.config.max_fetch:
            raise ValueError(f"offset + limit = {window} exceeds max_fetch={self.config.max_fetch}")
        return And(tuple(terms)), flt, window

    def _finish_search(self, op: SearchOp, flt: Any, window: int, rows: List[Tuple[Dict[str, Any], str, Optional[float]]]) -> List[SearchItem]:
        kept = [
            self._search_item(meta, text, score)
            for meta, text, score in rows
            if not flt or _compare(json.loads(text), flt)
        ]
        if not (op.query and self._embedder is not None):
            kept.sort(key=lambda it: (it.namespace, it.key))
        return kept[op.offset : window]

    def _ns_conditions(self, op: ListNamespacesOp) -> Tuple[Filter, List[Tuple[str, ...]]]:
        terms: List[Filter] = [Eq("lg_marker", True)]
        conds = list(op.match_conditions or ())
        for c in conds:
            if c.match_type == "prefix":
                terms.extend(_ns_eq(c.path))
        if conds:
            terms.append(_Namespace(conds))
        return And(tuple(terms)), conds

    def _finish_namespaces(self, op: ListNamespacesOp, namespaces: Iterable[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
        out = {ns[: op.max_depth] if op.max_depth is not None else ns for ns in namespaces}
        return sorted(out)[op.offset : op.offset + op.limit]

    def _empty_filter(self, ns: Tuple[str, ...]) -> Filter:
        return And((Eq("lg_marker", False), Eq("lg_depth", len(ns)), *_ns_eq(ns)))

    # ---- sync -------------------------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> List[Result]:
        return [self._run(op) for op in ops]

    def _run(self, op: Op) -> Result:
        b = self.backend
        if isinstance(op, GetOp):
            rec = b.get([_item_id(op.namespace, op.key)])[0]
            return self._item(rec.metadata, rec.text) if rec else None
        if isinstance(op, SearchOp):
            plan, flt, window = self._search_plan(op)
            if op.query and self._embedder is not None:
                hits = b.recall(op.query, k=window if not flt else min(self.config.max_fetch, max(window, 1) * self.config.overfetch), filter=plan)
                rows = [(h.metadata, h.text, h.score) for h in hits]
            else:
                rows = [(r.metadata, r.text, None) for r in b.list(plan).records]
            return self._finish_search(op, flt, window, rows)
        if isinstance(op, ListNamespacesOp):
            plan, _ = self._ns_conditions(op)
            return self._finish_namespaces(op, (_namespace(r.metadata) for r in b.list(plan).records))
        if isinstance(op, PutOp):
            self._put(op)
            return None
        raise ValueError(f"Unknown operation type: {type(op)}")

    def _put(self, op: PutOp) -> None:
        if op.ttl is not None:
            raise NotImplementedError("HolographicStore does not support TTL")
        b = self.backend
        ns = tuple(op.namespace)
        item_id = _item_id(ns, op.key)
        if op.value is None:
            if b.forget([item_id]) and not b.list(self._empty_filter(ns), limit=1).records:
                b.forget([_marker_id(ns)])
            return
        prev = b.get([item_id])[0]
        now = datetime.now(timezone.utc)
        created = datetime.fromisoformat(prev.metadata["lg_created"]) if prev else now
        text = self._index_text(op.value, op.index)
        vector = self._embedder.embed_documents([text])[0] if text is not None else self._const
        b.remember([self._item_record(op, created, now, vector, text is not None), self._marker_record(ns)])

    # ---- async ------------------------------------------------------------------------

    async def abatch(self, ops: Iterable[Op]) -> List[Result]:
        return [await self._arun(op) for op in ops]

    async def _arun(self, op: Op) -> Result:
        b = self.abackend
        if isinstance(op, GetOp):
            rec = (await b.get([_item_id(op.namespace, op.key)]))[0]
            return self._item(rec.metadata, rec.text) if rec else None
        if isinstance(op, SearchOp):
            plan, flt, window = self._search_plan(op)
            if op.query and self._embedder is not None:
                hits = await b.recall(op.query, k=window if not flt else min(self.config.max_fetch, max(window, 1) * self.config.overfetch), filter=plan)
                rows = [(h.metadata, h.text, h.score) for h in hits]
            else:
                page = await b.list(plan)
                rows = [(r.metadata, r.text, None) for r in page.records]
            return self._finish_search(op, flt, window, rows)
        if isinstance(op, ListNamespacesOp):
            plan, _ = self._ns_conditions(op)
            page = await b.list(plan)
            return self._finish_namespaces(op, (_namespace(r.metadata) for r in page.records))
        if isinstance(op, PutOp):
            await self._aput(op)
            return None
        raise ValueError(f"Unknown operation type: {type(op)}")

    async def _aput(self, op: PutOp) -> None:
        if op.ttl is not None:
            raise NotImplementedError("HolographicStore does not support TTL")
        b = self.abackend
        ns = tuple(op.namespace)
        item_id = _item_id(ns, op.key)
        if op.value is None:
            if await b.forget([item_id]) and not (await b.list(self._empty_filter(ns), limit=1)).records:
                await b.forget([_marker_id(ns)])
            return
        prev = (await b.get([item_id]))[0]
        now = datetime.now(timezone.utc)
        created = datetime.fromisoformat(prev.metadata["lg_created"]) if prev else now
        text = self._index_text(op.value, op.index)
        vector = (await aembed_documents(self._embedder, [text]))[0] if text is not None else self._const
        await b.remember([self._item_record(op, created, now, vector, text is not None), self._marker_record(ns)])
