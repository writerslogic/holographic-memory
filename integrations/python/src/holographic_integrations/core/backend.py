# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""The one place that talks to Holographic Memory.

Every adapter goes through `HmsBackend` (sync) or `AsyncHmsBackend` (async). The operation
set is the memory API: `remember`, `recall`, `get`, `list`, `forget`. It is implemented here
on top of `hms-server`'s documents endpoints; when the SDK grows native remember/recall/
forget/explain, only this module changes.

Enumeration uses the service's metadata cursor. Residual filters select document ids before
ranking; vector scores merge across bounded id groups. Fused text queries require their
eligible set to fit one group because independently normalized scores cannot be merged.

Limits inherited from `hms-server` (docs/SERVER.md):

- `get` fetches text through an exact id-filtered vector query.
- One `list` returns at most `max_fetch` records and reports whether more matched.
- Ids are 1..256 bytes, may not start with `hms:`; text is at most 64 KiB per record.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from urllib.parse import quote

import httpx
from holographic_sdk import AsyncHolographicClient, HolographicClient

from .embedders import Embedder, HashingEmbedder, aembed_documents, aembed_query
from .filters import Filter, FilterLike, parse, split
from .models import COLLECTION_KEY, ID_KEY, RESERVED_PREFIX, Hit, HmsError, Page, Record

_RETRY_STATUSES = (503,)
_MAX_RETRIES = 3
_MAX_ID_BYTES = 256


@dataclass
class HmsConfig:
    """Connection and behavior knobs shared by every adapter.

    `collection` scopes records inside a tenant: ids are namespaced by it and every query is
    restricted to it. Without one, a backend sees only records written without one (documents
    written by other clients, which lack the reserved keys, are not visible). Use `tenant_id` for hard isolation (separate stores).

    `embedder` is the encoder. Default: `HashingEmbedder(input_dim or 384)`, a lexical
    embedder that needs no model. With `text_queries=True` the service embeds text itself
    (it must run with on-device model stages) and `embedder` is optional; `input_dim` must
    then equal the service's `--input-dim`.

    `zero_trust_key` masks vectors client-side (see `holographic_sdk.VectorMask`): obfuscation,
    not encryption.
    """

    url: str = "http://127.0.0.1:8080"
    api_key: Optional[str] = None
    tenant_id: Optional[str] = None
    timeout: float = 30.0
    zero_trust_key: str = ""
    mask_salt: Optional[bytes] = None
    collection: Optional[str] = None
    embedder: Optional[Embedder] = None
    input_dim: Optional[int] = None
    text_queries: bool = False
    max_fetch: int = 100
    overfetch: int = 5
    batch_size: int = 256
    transport: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not 1 <= self.max_fetch:
            raise ValueError("max_fetch must be >= 1")
        if self.overfetch < 1:
            raise ValueError("overfetch must be >= 1")
        if not 1 <= self.batch_size <= 1000:
            raise ValueError("batch_size must be 1..1000")
        if self.collection is not None and (not self.collection or "/" in self.collection):
            raise ValueError("collection must be non-empty and contain no '/'")
        if self.embedder is None and not (self.text_queries and self.input_dim):
            self.embedder = HashingEmbedder(self.input_dim or 384)
        dim = self.embedder.dim if self.embedder is not None else self.input_dim
        if self.input_dim is not None and self.embedder is not None and self.embedder.dim != self.input_dim:
            raise ValueError(f"embedder.dim {self.embedder.dim} != input_dim {self.input_dim}")
        self.input_dim = dim


def _check_metadata(metadata: Dict[str, Any]) -> None:
    for key in metadata:
        if not isinstance(key, str):
            raise ValueError("metadata keys must be strings")
        if key.startswith(RESERVED_PREFIX):
            raise ValueError(f"metadata key {key!r} uses the reserved prefix {RESERVED_PREFIX!r}")


def _error(res: httpx.Response) -> HmsError:
    code = None
    message = res.text[:300]
    try:
        body = res.json()
        err = body.get("error", {})
        code, message = err.get("code"), err.get("message", message)
    except (ValueError, AttributeError):
        # Not a JSON problem document: keep the raw status text as the message.
        pass
    return HmsError(f"hms-server returned {res.status_code}: {message}", res.status_code, code)


class _Shared:
    """Request construction and response parsing, independent of sync or async I/O."""

    def __init__(self, config: HmsConfig):
        self.config = config
        self.base = config.url.rstrip("/")
        if config.input_dim is None:
            raise ValueError("input_dim or an embedder is required")
        self._probe = [1.0] + [0.0] * (config.input_dim - 1)
        self._document_ids_capability: Optional[bool] = None

    def server_id(self, record_id: str) -> str:
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record id must be a non-empty string")
        sid = record_id if self.config.collection is None else f"{self.config.collection}/{record_id}"
        if sid.startswith("hms:"):
            raise ValueError("ids may not start with 'hms:'")
        if len(sid.encode("utf-8")) > _MAX_ID_BYTES:
            raise ValueError(f"id exceeds {_MAX_ID_BYTES} bytes")
        return sid

    def wire_doc(self, record: Record, vector: Optional[Sequence[float]]) -> Dict[str, Any]:
        _check_metadata(record.metadata)
        meta = dict(record.metadata)
        meta[ID_KEY] = record.id
        meta[COLLECTION_KEY] = self.config.collection or ""
        doc: Dict[str, Any] = {"id": self.server_id(record.id), "text": record.text, "metadata": meta}
        if vector is not None:
            doc["vector"] = list(vector)
        return doc

    def scope(self, pushed: Dict[str, Any]) -> Dict[str, Any]:
        return {**pushed, COLLECTION_KEY: self.config.collection or ""}

    def plan(self, flt: FilterLike, k: int) -> Tuple[Dict[str, Any], Optional[Filter], int]:
        parsed = parse(flt)
        pushed, residual = split(parsed)
        if k < 1:
            raise ValueError("k must be >= 1")
        if k > self.config.max_fetch:
            raise ValueError(
                f"k={k} exceeds max_fetch={self.config.max_fetch}; the service caps top_k (--max-top-k)"
            )
        fetch = k if residual is None else min(self.config.max_fetch, k * self.config.overfetch)
        return self.scope(pushed), residual, fetch

    def query_body(self, vector: Optional[List[float]], text: Optional[str], top_k: int, pushed: Dict[str, Any]) -> Dict[str, Any]:
        body: Dict[str, Any] = {"top_k": top_k}
        if vector is not None:
            body["query_vector"] = vector
        if text is not None:
            body["query"] = text
        if pushed:
            body["filter"] = pushed
        return body

    @staticmethod
    def to_hit(match: Dict[str, Any]) -> Hit:
        meta = dict(match.get("metadata") or {})
        rid = meta.pop(ID_KEY, match.get("id", ""))
        meta.pop(COLLECTION_KEY, None)
        return Hit(id=rid, text=match.get("text") or "", metadata=meta, score=float(match.get("score", 0.0)))

    @staticmethod
    def to_record(hit: Hit) -> Record:
        return Record(id=hit.id, text=hit.text, metadata=hit.metadata)

    def finish(self, matches: List[Dict[str, Any]], residual: Optional[Filter], k: int) -> List[Hit]:
        hits = [self.to_hit(m) for m in matches]
        if residual is not None:
            hits = [h for h in hits if residual.matches(h.metadata)]
        return hits[:k]

    def metadata_match(self, document: Dict[str, Any], filter: Optional[Filter]) -> Optional[Tuple[str, str]]:
        metadata = document.get("metadata") or {}
        if not isinstance(metadata, dict):
            return None
        if metadata.get(COLLECTION_KEY) != (self.config.collection or ""):
            return None
        rid = metadata.get(ID_KEY)
        if not isinstance(rid, str) or document.get("id") != self.server_id(rid):
            return None
        clean = {key: value for key, value in metadata.items() if key not in (ID_KEY, COLLECTION_KEY)}
        if filter is not None and not filter.matches(clean):
            return None
        return document["id"], rid

    def list_limit(self, limit: Optional[int]) -> int:
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("limit must be a positive integer")
        return min(self.config.max_fetch if limit is None else limit, self.config.max_fetch)

    @staticmethod
    def check_cursor(previous: Optional[str], following: Any) -> None:
        if not isinstance(following, str) or (previous is not None and following <= previous):
            raise HmsError("service returned a non-advancing metadata cursor", code="invalid_response")

    def best_matches(self, matches: List[Dict[str, Any]], filter: FilterLike, k: int) -> List[Dict[str, Any]]:
        parsed = parse(filter)
        eligible = [match for match in matches if parsed is None or parsed.matches(self.to_hit(match).metadata)]
        eligible.sort(key=lambda match: (-float(match["score"]), match["id"]))
        return eligible[:k]

    def cache_document_ids_capability(self, response: httpx.Response) -> None:
        if response.status_code not in (200, 404, 405):
            raise _error(response)
        supported = False
        if response.status_code == 200:
            try:
                properties = response.json()["components"]["schemas"]["QueryIn"]["properties"]
                supported = isinstance(properties, dict) and "document_ids" in properties
            except (ValueError, KeyError, TypeError):
                pass
        self._document_ids_capability = supported

    def require_document_ids_capability(self) -> None:
        if not self._document_ids_capability:
            raise HmsError("residual filtering requires a server with document_ids support; upgrade hms-server", code="unsupported_capability")

    def checked_matches(self, response: httpx.Response, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            payload = response.json()
        except ValueError:
            raise HmsError("service returned an invalid query response", code="invalid_response") from None
        if not isinstance(payload, dict) or not isinstance(payload.get("matches"), list):
            raise HmsError("service returned an invalid query response", code="invalid_response")
        matches = payload["matches"]
        if "document_ids" in request:
            allowed = set(request["document_ids"])
            for match in matches:
                try:
                    selected = self.metadata_match(match, None) if isinstance(match, dict) else None
                except (TypeError, ValueError):
                    selected = None
                if selected is None or selected[0] not in allowed:
                    raise HmsError("service returned a document outside the selected ids or collection", code="invalid_response")
        return matches

    def mask(self, vector: List[float]) -> List[float]:
        return self._client_mask(vector)

    def chunks(self, items: Sequence[Any]) -> List[Sequence[Any]]:
        n = self.config.batch_size
        return [items[i : i + n] for i in range(0, len(items), n)]

    def embeds_locally(self, record: Record) -> bool:
        if record.vector is not None:
            return False
        return not self.config.text_queries


class HmsBackend(_Shared):
    """Synchronous backend."""

    def __init__(self, config: Optional[HmsConfig] = None, **kwargs: Any):
        super().__init__(config or HmsConfig(**kwargs))
        c = self.config
        self._sdk = HolographicClient(
            url=c.url, zero_trust_key=c.zero_trust_key, api_key=c.api_key, tenant_id=c.tenant_id,
            timeout=c.timeout, mask_salt=c.mask_salt, transport=c.transport,
        )
        self._http = self._sdk.client
        self._client_mask = self._sdk.encrypt_vector

    def close(self) -> None:
        self._sdk.close()

    def __enter__(self) -> "HmsBackend":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _send(self, method: str, path: str, body: Any = None, *, params: Any = None) -> httpx.Response:
        for attempt in range(_MAX_RETRIES + 1):
            res = self._http.request(method, f"{self.base}{path}", json=body, params=params)
            if res.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES:
                time.sleep(min(float(res.headers.get("retry-after", "1")), 5.0))
                continue
            return res
        raise AssertionError("unreachable")

    def remember(self, records: Sequence[Record]) -> List[str]:
        """Store records, replacing any with the same id. Returns the ids in order."""
        records = list(records)
        for start in range(0, len(records), self.config.batch_size):
            batch = records[start : start + self.config.batch_size]
            need = [r for r in batch if self.embeds_locally(r)]
            vectors = iter(self.config.embedder.embed_documents([r.text for r in need]) if need else [])
            docs = [
                self.wire_doc(r, next(vectors) if self.embeds_locally(r) else r.vector)
                for r in batch
            ]
            wire = [{**d, "vector": self.mask(d["vector"])} if "vector" in d else d for d in docs]
            res = self._send("POST", "/api/v1/documents/batch", wire)
            if res.status_code != 200:
                raise _error(res)
        return [r.id for r in records]

    def recall(
        self, query: Optional[str] = None, *, vector: Optional[List[float]] = None, k: int = 4, filter: FilterLike = None
    ) -> List[Hit]:
        """Top-`k` hits. Pass `query` text (embedded here, or by the service in text mode) and/or `vector`."""
        if query is None and vector is None:
            raise ValueError("recall needs query text or a vector")
        pushed, residual, fetch = self.plan(filter, k)
        if self.config.text_queries and query is not None:
            body = self.query_body(None if vector is None else self.mask(vector), query, fetch, pushed)
        else:
            vec = vector if vector is not None else self.config.embedder.embed_query(query)
            body = self.query_body(self.mask(vec), None, fetch, pushed)
        if residual is not None:
            return self.finish(self._filtered_matches(body, parse(filter), k), residual, k)
        return self.finish(self._matches(body), None, k)

    def _metadata(self, filter: FilterLike = None):
        parsed = parse(filter)
        cursor = None
        while True:
            params = {"limit": 1000}
            if cursor is not None:
                params["after"] = cursor
            res = self._send("GET", "/api/v1/documents", params=params)
            if res.status_code != 200:
                raise _error(res)
            page = res.json()
            for document in page["documents"]:
                selected = self.metadata_match(document, parsed)
                if selected is not None:
                    yield selected
            following = page.get("next_cursor")
            if following is None:
                return
            self.check_cursor(cursor, following)
            cursor = following

    def _filtered_matches(self, body: Dict[str, Any], filter: FilterLike, k: int) -> List[Dict[str, Any]]:
        if self._document_ids_capability is None:
            self.cache_document_ids_capability(self._send("GET", "/openapi.json"))
        self.require_document_ids_capability()
        ids: List[str] = []
        best: List[Dict[str, Any]] = []
        for sid, _ in self._metadata(filter):
            if body.get("query") is not None and len(ids) == 1000:
                raise HmsError("text-mode residual filtering supports at most 1000 matching records; narrow the filter", code="filter_limit")
            ids.append(sid)
            if len(ids) == 1000 and body.get("query") is None:
                best = self.best_matches(best + self._matches({**body, "top_k": k, "document_ids": ids}), filter, k)
                ids = []
        if ids:
            best = self.best_matches(best + self._matches({**body, "top_k": k, "document_ids": ids}), filter, k)
        return best

    def _matches(self, body: Dict[str, Any]) -> List[Dict[str, Any]]:
        res = self._send("POST", "/api/v1/query", body)
        if res.status_code != 200:
            raise _error(res)
        return self.checked_matches(res, body)

    def get(self, ids: Sequence[str]) -> List[Optional[Record]]:
        """Records for `ids` in order; `None` where an id is absent."""
        out: List[Optional[Record]] = []
        for rid in ids:
            body = self.query_body(self.mask(self._probe), None, 1, self.scope({ID_KEY: rid}))
            matches = self._matches(body)
            out.append(self.to_record(self.to_hit(matches[0])) if matches else None)
        return out

    def list(self, filter: FilterLike = None, limit: Optional[int] = None) -> Page:
        """Enumerate matching metadata in id order, then fetch the bounded result texts."""
        want = self.list_limit(limit)
        ids = []
        for _, rid in self._metadata(filter):
            ids.append(rid)
            if len(ids) > want:
                break
        return Page([record for record in self.get(ids[:want]) if record is not None], len(ids) > want)

    def forget(self, ids: Sequence[str]) -> int:
        """Delete by id. Returns how many existed."""
        removed = 0
        for rid in ids:
            res = self._send("DELETE", f"/api/v1/documents/{quote(self.server_id(rid), safe='')}")
            if res.status_code == 404:
                continue
            if res.status_code != 200:
                raise _error(res)
            removed += 1
        return removed

    def forget_where(self, filter: FilterLike = None) -> int:
        """Delete every matching record by scanning paginated metadata without retaining text."""
        total = 0
        for _, rid in self._metadata(filter):
            total += self.forget([rid])
        return total

    def maintain(self) -> bool:
        """Train the service's document index; true when an index is now trained."""
        res = self._send("POST", "/api/v1/index/maintain", {})
        if res.status_code != 200:
            raise _error(res)
        return bool(res.json().get("documentIndexTrained"))


class AsyncHmsBackend(_Shared):
    """Asynchronous backend with the same operations as `HmsBackend`."""

    def __init__(self, config: Optional[HmsConfig] = None, **kwargs: Any):
        super().__init__(config or HmsConfig(**kwargs))
        c = self.config
        self._sdk = AsyncHolographicClient(
            url=c.url, zero_trust_key=c.zero_trust_key, api_key=c.api_key, tenant_id=c.tenant_id,
            timeout=c.timeout, mask_salt=c.mask_salt, transport=c.transport,
        )
        self._http = self._sdk.client
        self._client_mask = self._sdk.encrypt_vector

    async def aclose(self) -> None:
        await self._sdk.aclose()

    async def __aenter__(self) -> "AsyncHmsBackend":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def _send(self, method: str, path: str, body: Any = None, *, params: Any = None) -> httpx.Response:
        for attempt in range(_MAX_RETRIES + 1):
            res = await self._http.request(method, f"{self.base}{path}", json=body, params=params)
            if res.status_code in _RETRY_STATUSES and attempt < _MAX_RETRIES:
                await asyncio.sleep(min(float(res.headers.get("retry-after", "1")), 5.0))
                continue
            return res
        raise AssertionError("unreachable")

    async def remember(self, records: Sequence[Record]) -> List[str]:
        records = list(records)
        for start in range(0, len(records), self.config.batch_size):
            batch = records[start : start + self.config.batch_size]
            need = [r for r in batch if self.embeds_locally(r)]
            vectors = iter(await aembed_documents(self.config.embedder, [r.text for r in need]) if need else [])
            docs = [
                self.wire_doc(r, next(vectors) if self.embeds_locally(r) else r.vector)
                for r in batch
            ]
            wire = [{**d, "vector": self.mask(d["vector"])} if "vector" in d else d for d in docs]
            res = await self._send("POST", "/api/v1/documents/batch", wire)
            if res.status_code != 200:
                raise _error(res)
        return [r.id for r in records]

    async def recall(
        self, query: Optional[str] = None, *, vector: Optional[List[float]] = None, k: int = 4, filter: FilterLike = None
    ) -> List[Hit]:
        if query is None and vector is None:
            raise ValueError("recall needs query text or a vector")
        pushed, residual, fetch = self.plan(filter, k)
        if self.config.text_queries and query is not None:
            body = self.query_body(None if vector is None else self.mask(vector), query, fetch, pushed)
        else:
            vec = vector if vector is not None else await aembed_query(self.config.embedder, query)
            body = self.query_body(self.mask(vec), None, fetch, pushed)
        if residual is not None:
            return self.finish(await self._filtered_matches(body, parse(filter), k), residual, k)
        return self.finish(await self._matches(body), None, k)

    async def _metadata(self, filter: FilterLike = None):
        parsed = parse(filter)
        cursor = None
        while True:
            params = {"limit": 1000}
            if cursor is not None:
                params["after"] = cursor
            res = await self._send("GET", "/api/v1/documents", params=params)
            if res.status_code != 200:
                raise _error(res)
            page = res.json()
            for document in page["documents"]:
                selected = self.metadata_match(document, parsed)
                if selected is not None:
                    yield selected
            following = page.get("next_cursor")
            if following is None:
                return
            self.check_cursor(cursor, following)
            cursor = following

    async def _filtered_matches(self, body: Dict[str, Any], filter: FilterLike, k: int) -> List[Dict[str, Any]]:
        if self._document_ids_capability is None:
            self.cache_document_ids_capability(await self._send("GET", "/openapi.json"))
        self.require_document_ids_capability()
        ids: List[str] = []
        best: List[Dict[str, Any]] = []
        async for sid, _ in self._metadata(filter):
            if body.get("query") is not None and len(ids) == 1000:
                raise HmsError("text-mode residual filtering supports at most 1000 matching records; narrow the filter", code="filter_limit")
            ids.append(sid)
            if len(ids) == 1000 and body.get("query") is None:
                best = self.best_matches(best + await self._matches({**body, "top_k": k, "document_ids": ids}), filter, k)
                ids = []
        if ids:
            best = self.best_matches(best + await self._matches({**body, "top_k": k, "document_ids": ids}), filter, k)
        return best

    async def _matches(self, body: Dict[str, Any]) -> List[Dict[str, Any]]:
        res = await self._send("POST", "/api/v1/query", body)
        if res.status_code != 200:
            raise _error(res)
        return self.checked_matches(res, body)

    async def get(self, ids: Sequence[str]) -> List[Optional[Record]]:
        gate = asyncio.Semaphore(8)

        async def one(rid: str) -> Optional[Record]:
            async with gate:
                body = self.query_body(self.mask(self._probe), None, 1, self.scope({ID_KEY: rid}))
                matches = await self._matches(body)
            return self.to_record(self.to_hit(matches[0])) if matches else None

        return list(await asyncio.gather(*(one(r) for r in ids)))

    async def list(self, filter: FilterLike = None, limit: Optional[int] = None) -> Page:
        want = self.list_limit(limit)
        ids = []
        async for _, rid in self._metadata(filter):
            ids.append(rid)
            if len(ids) > want:
                break
        return Page([record for record in await self.get(ids[:want]) if record is not None], len(ids) > want)

    async def forget(self, ids: Sequence[str]) -> int:
        removed = 0
        for rid in ids:
            res = await self._send("DELETE", f"/api/v1/documents/{quote(self.server_id(rid), safe='')}")
            if res.status_code == 404:
                continue
            if res.status_code != 200:
                raise _error(res)
            removed += 1
        return removed

    async def forget_where(self, filter: FilterLike = None) -> int:
        total = 0
        async for _, rid in self._metadata(filter):
            total += await self.forget([rid])
        return total

    async def maintain(self) -> bool:
        res = await self._send("POST", "/api/v1/index/maintain", {})
        if res.status_code != 200:
            raise _error(res)
        return bool(res.json().get("documentIndexTrained"))
