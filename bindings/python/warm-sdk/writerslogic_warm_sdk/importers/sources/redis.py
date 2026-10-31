# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Redis Stack / RediSearch indexes: hashes and JSON documents read by key scan."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ..common import redact_uri, secret_from, uri_secret
from ..model import ImporterError, MissingDependency, Record
from .base import Source, parse_vector

_PAGE = 500


def _dec(value: Any) -> Any:
    """Hash values arrive as bytes: UTF-8 text when valid, else left as bytes (base64 later)."""
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return bytes(value)
    return value


def parse_ft_info(raw: Any) -> Dict[str, Any]:
    """Pull prefixes, key type and the vector attribute out of an FT.INFO reply (RESP2 list or RESP3 dict)."""
    info = _pairs(raw)
    defn = _pairs(info.get("index_definition", []))
    prefixes = [_s(p) for p in (defn.get("prefixes") or [])]
    key_type = _s(defn.get("key_type", "HASH")).upper()
    vector_attr = None
    vector_dim = None
    for attr in info.get("attributes", []) or []:
        a = _pairs(attr) if isinstance(attr, (list, tuple, dict)) else {}
        flat = [_s(x) for x in (attr if isinstance(attr, (list, tuple)) else [])]
        if "VECTOR" in flat or _s(a.get("type", "")).upper() == "VECTOR":
            vector_attr = _s(a.get("identifier") or a.get("attribute") or (flat[1] if len(flat) > 1 else ""))
            if "DIM" in flat:
                vector_dim = int(flat[flat.index("DIM") + 1])
    return {"prefixes": prefixes, "key_type": key_type, "vector_attr": vector_attr, "vector_dim": vector_dim,
            "num_docs": int(float(_s(info.get("num_docs", 0)) or 0))}


def _s(v: Any) -> str:
    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)


def _pairs(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return {_s(k): v for k, v in raw.items()}
    out: Dict[str, Any] = {}
    if isinstance(raw, (list, tuple)):
        for i in range(0, len(raw) - 1, 2):
            out[_s(raw[i])] = raw[i + 1]
    return out


class RedisSource(Source):
    name = "redis"
    summary = "Redis Stack / RediSearch hashes and JSON documents (key scan; --index discovers prefix and vector field)"

    def __init__(self, url: str, prefix: str, index: Optional[str], key_type: str, id_field: Optional[str],
                 text_field: Optional[str], vector_field: Optional[str], namespace_field: Optional[str],
                 created_field: Optional[str], updated_field: Optional[str], metadata_fields: Optional[List[str]],
                 page_size: int = _PAGE, client: Any = None):
        self.url, self.prefix, self.index, self.key_type = url, prefix, index, key_type
        self.id_field, self.text_field, self.vector_field = id_field, text_field, vector_field
        self.namespace_field, self.created_field, self.updated_field = namespace_field, created_field, updated_field
        self.metadata_fields = metadata_fields
        self.page = page_size
        self._client = client
        self._discovered = False

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--url", help="redis:// or rediss:// URL (default $REDIS_URL, else redis://127.0.0.1:6379); "
                                          "the password is read from the URL or $REDIS_PASSWORD and never logged")
        parser.add_argument("--prefix", default="", help="key prefix to scan (default: from --index, else every key)")
        parser.add_argument("--index", help="RediSearch index name: FT.INFO supplies prefix, key type and vector field")
        parser.add_argument("--key-type", choices=["auto", "hash", "json"], default="auto")
        parser.add_argument("--id-field", help="field holding the id (default: the key without its prefix)")
        parser.add_argument("--text-field", default="content", help="field holding the text (default content)")
        parser.add_argument("--vector-field", help="field holding the vector (default: from --index, else 'embedding')")
        parser.add_argument("--namespace-field")
        parser.add_argument("--created-field")
        parser.add_argument("--updated-field")
        parser.add_argument("--metadata-fields", help="comma-separated fields; default every other field")
        parser.add_argument("--page-size", type=int, default=_PAGE)

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        url = args.url or env.get("REDIS_URL") or "redis://127.0.0.1:6379"
        uri_secret(url)
        secret_from(None, dict(env), "REDIS_PASSWORD")
        meta = [m.strip() for m in args.metadata_fields.split(",")] if args.metadata_fields else None
        return cls(url, args.prefix, args.index, args.key_type, args.id_field, args.text_field, args.vector_field,
                   args.namespace_field, args.created_field, args.updated_field, meta, args.page_size)

    def describe(self) -> Dict[str, Any]:
        return {"url": redact_uri(self.url), "prefix": self.prefix, "index": self.index, "key_type": self.key_type,
                "fields": [self.id_field, self.text_field, self.vector_field, self.namespace_field,
                           self.created_field, self.updated_field, self.metadata_fields]}

    def _conn(self):
        if self._client is None:
            try:
                import redis
            except ImportError:
                raise MissingDependency("redis", "redis", "import-redis") from None
            import os

            self._client = redis.Redis.from_url(self.url, password=os.environ.get("REDIS_PASSWORD") or None)
        return self._client

    def _discover(self) -> None:
        if self._discovered:
            return
        self._discovered = True
        r = self._conn()
        if self.index:
            try:
                info = parse_ft_info(r.execute_command("FT.INFO", self.index))
            except Exception as exc:  # redis.ResponseError: no such index / no RediSearch module
                raise ImporterError(f"FT.INFO {self.index} failed ({type(exc).__name__}); is RediSearch loaded and the index name right?") from None
            if not self.prefix and info["prefixes"]:
                self.prefix = info["prefixes"][0]
            if self.key_type == "auto":
                self.key_type = info["key_type"].lower()
            if not self.vector_field and info["vector_attr"]:
                self.vector_field = info["vector_attr"].lstrip("$.")
            self._num_docs = info["num_docs"]
        if self.vector_field is None:
            self.vector_field = "embedding"

    def count(self) -> Optional[int]:
        self._discover()
        if self.index:
            return getattr(self, "_num_docs", None)
        if not self.prefix:
            return int(self._conn().dbsize())
        return None

    def _type_of(self, key: bytes) -> str:
        if self.key_type != "auto":
            return self.key_type
        t = self._conn().type(key)
        t = t.decode() if isinstance(t, bytes) else t
        return "json" if t == "ReJSON-RL" else "hash"

    def records(self, cursor: Any = None) -> Iterator[Record]:
        self._discover()
        r = self._conn()
        scan_cursor = 0
        seen: set = set()  # SCAN may return a key more than once; bounded by the page window only
        while True:
            scan_cursor, keys = r.scan(cursor=scan_cursor, match=(self.prefix + "*") if self.prefix else None,
                                       count=self.page)
            keys = sorted(k for k in keys if k not in seen)
            seen = set(keys)
            if keys:
                yield from self._page(keys)
            if int(scan_cursor) == 0:
                return

    def _page(self, keys: List[bytes]) -> Iterator[Record]:
        r = self._conn()
        kinds = {k: self._type_of(k) for k in keys}
        pipe = r.pipeline(transaction=False)
        for k in keys:
            if kinds[k] == "json":
                pipe.execute_command("JSON.GET", k, "$")
            else:
                pipe.hgetall(k)
        for k, raw in zip(keys, pipe.execute()):
            kid = k.decode("utf-8", "replace")
            ident = kid[len(self.prefix):] if self.prefix and kid.startswith(self.prefix) else kid
            if kinds[k] == "json":
                try:
                    doc = json.loads(raw)
                    doc = doc[0] if isinstance(doc, list) and doc else doc
                    if not isinstance(doc, dict):
                        raise ValueError("not an object")
                except (TypeError, ValueError) as exc:
                    yield Record(id=ident, error=f"key {kid}: unreadable JSON document ({exc})")
                    continue
            else:
                if not raw:
                    continue  # expired or deleted between SCAN and HGETALL
                doc = {}
                for f, v in raw.items():
                    f = _dec(f)
                    doc[f] = v if f == self.vector_field else _dec(v)  # raw float32 bytes must stay bytes
            yield self._record(ident, doc)

    def _record(self, ident: str, doc: Dict[str, Any]) -> Record:
        used = set()

        def take(name: Optional[str]) -> Any:
            if name and name in doc:
                used.add(name)
                return doc[name]
            return None

        if self.id_field and self.id_field in doc:
            ident = str(take(self.id_field))
        text = take(self.text_field)
        vec = take(self.vector_field)
        ns = take(self.namespace_field)
        created, updated = take(self.created_field), take(self.updated_field)
        if vec is not None and not isinstance(vec, bytes):  # bytes: raw little-endian float32
            vec = parse_vector(vec)
        if self.metadata_fields is not None:
            meta = {f: doc[f] for f in self.metadata_fields if f in doc}
        else:
            meta = {f: v for f, v in doc.items() if f not in used}
        return Record(id=ident, text=None if text is None else str(text), vector=vec, metadata=meta,
                      namespace="" if ns is None else str(ns), created_at=created, updated_at=updated)


__all__ = ["RedisSource", "parse_ft_info"]
