# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pinecone source: list ids page by page, fetch them in batches, per namespace (serverless indexes)."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ..common import redact_uri, secret_from
from ..model import ImporterError, MissingDependency, Record
from ._fields import get_path, pick_text_field, text_value, without_paths
from .base import Source

_DEFAULT_NS_KEYS = ("", "__default__")


def _attr(obj: Any, *names: str) -> Any:
    """Attribute or key of a client response object, whichever the client version exposes."""
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


class PineconeSource(Source):
    name = "pinecone"
    summary = "Pinecone serverless index: list/fetch per namespace (seekable by page token; API key from env)"
    seekable = True

    def __init__(
        self,
        host: Optional[str] = None,
        *,
        index_name: Optional[str] = None,
        api_key: Optional[str] = None,
        namespaces: Optional[List[str]] = None,
        prefix: Optional[str] = None,
        text_field: Optional[str] = None,
        timestamp_field: Optional[str] = None,
        updated_field: Optional[str] = None,
        page_size: int = 100,
        index: Any = None,
    ):
        if not api_key and index is None:
            raise ImporterError("set PINECONE_API_KEY")
        if not (host or index_name) and index is None:
            raise ImporterError("give --host (the index data-plane URL) or --index-name")
        if not 1 <= page_size <= 100:
            raise ImporterError("--page-size must be 1..100 (the list limit)")
        self.host, self.index_name, self.api_key = host, index_name, api_key
        self.namespaces = sorted(namespaces or [])
        self.prefix, self.text_field = prefix, text_field
        self.timestamp_field, self.updated_field, self.page_size = timestamp_field, updated_field, page_size
        self._index = index

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--host", help="index data-plane URL (from the console or describe_index); "
                                           "an http:// host works for Pinecone Local")
        parser.add_argument("--index-name", help="resolve the host through the control plane instead of --host")
        parser.add_argument("--namespace", dest="namespaces", action="append", default=[],
                            help="namespace to import, repeatable (default: every namespace; '' is the default one)")
        parser.add_argument("--prefix", help="only ids with this prefix")
        parser.add_argument("--text-field", help="metadata key holding the text (default: text, content, chunk_text, ...)")
        parser.add_argument("--timestamp-field", help="metadata key holding the creation time")
        parser.add_argument("--updated-field", help="metadata key holding the last-update time")
        parser.add_argument("--page-size", type=int, default=100, help="ids per list page and per fetch (max 100)")

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(args.host, index_name=args.index_name, api_key=secret_from(None, env, "PINECONE_API_KEY"),
                   namespaces=args.namespaces, prefix=args.prefix, text_field=args.text_field,
                   timestamp_field=args.timestamp_field, updated_field=args.updated_field, page_size=args.page_size)

    def describe(self) -> Dict[str, Any]:
        return {"host": redact_uri(self.host) if self.host else None, "index_name": self.index_name,
                "namespaces": self.namespaces, "prefix": self.prefix, "text_field": self.text_field}

    # -- client ---------------------------------------------------------------------

    def _idx(self) -> Any:
        if self._index is None:
            try:
                from pinecone import Pinecone
            except ImportError:
                raise MissingDependency("pinecone", "pinecone", "import-pinecone") from None
            pc = Pinecone(api_key=self.api_key)
            host = self.host
            if not host:
                host = _attr(pc.describe_index(self.index_name), "host")
                if not host:
                    raise ImporterError(f"the control plane returned no host for index '{self.index_name}'")
            self._index = pc.Index(host=host)
        return self._index

    def close(self) -> None:
        closer = getattr(self._index, "close", None)
        if callable(closer):
            closer()

    def _namespaces(self) -> List[str]:
        stats = self._idx().describe_index_stats()
        found = _attr(stats, "namespaces") or {}
        names = sorted(found.keys()) if found else [""]
        if self.namespaces:
            missing = [n for n in self.namespaces if n not in names and not (n == "" and "__default__" in names)]
            if missing:
                raise ImporterError(f"namespace(s) {missing} not in the index; it has {names}")
            wanted = set(self.namespaces)
            names = [n for n in names if n in wanted or (n == "__default__" and "" in wanted)]
        return names

    def count(self) -> Optional[int]:
        try:
            stats = self._idx().describe_index_stats()
            namespaces = _attr(stats, "namespaces") or {}
            if self.namespaces or self.prefix:
                return None
            total = _attr(stats, "total_vector_count", "totalVectorCount")
            if total is None:
                total = sum(int(_attr(v, "vector_count", "vectorCount") or 0) for v in namespaces.values())
            return int(total)
        except ImporterError:
            return None

    # -- reading --------------------------------------------------------------------

    def records(self, cursor: Any = None) -> Iterator[Record]:
        idx = self._idx()
        resume_ns, resume_token, resume_offset = (cursor if cursor else (None, None, -1))
        for ns in self._namespaces():
            if resume_ns is not None and ns < resume_ns:
                continue
            resuming = resume_ns is not None and ns == resume_ns
            token = resume_token if resuming else None
            skip = (resume_offset + 1) if resuming else 0
            record_ns = "" if ns == "__default__" else ns
            while True:
                kwargs: Dict[str, Any] = {"namespace": ns, "limit": self.page_size}
                if self.prefix:
                    kwargs["prefix"] = self.prefix
                if token:
                    kwargs["pagination_token"] = token
                page = idx.list_paginated(**kwargs)
                ids = [str(_attr(v, "id")) for v in (_attr(page, "vectors") or [])]
                if ids:
                    fetched = _attr(idx.fetch(ids=ids, namespace=ns), "vectors") or {}
                    for offset, vid in enumerate(ids):
                        if offset < skip:
                            continue
                        yield self._record(vid, fetched.get(vid), record_ns, [ns, token, offset])
                skip = 0
                nxt = _attr(_attr(page, "pagination"), "next")
                if not nxt:
                    break
                token = nxt

    def _record(self, vid: str, vec: Any, namespace: str, cursor: Any) -> Record:
        if vec is None:
            return Record(id=vid, namespace=namespace, cursor=cursor,
                          error="listed but not returned by fetch (deleted during the import?)")
        meta = dict(_attr(vec, "metadata") or {})
        text_key = pick_text_field(meta, self.text_field)
        text = text_value(get_path(meta, text_key)) if text_key else None
        created = get_path(meta, self.timestamp_field) if self.timestamp_field else None
        updated = get_path(meta, self.updated_field) if self.updated_field else None
        meta = without_paths(meta, [p for p in (text_key, self.timestamp_field, self.updated_field) if p])
        sparse = _attr(vec, "sparse_values", "sparseValues")
        if sparse:
            meta["pinecone_sparse"] = {"indices": list(_attr(sparse, "indices") or []),
                                       "values": list(_attr(sparse, "values") or [])}
        values = _attr(vec, "values")
        return Record(id=vid, text=text, vector=list(values) if values else None, metadata=meta,
                      namespace=namespace, created_at=created, updated_at=updated, cursor=cursor)


__all__ = ["PineconeSource"]
