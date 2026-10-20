# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Qdrant: embedded local storage (--path) or a server / Qdrant Cloud (--url)."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ..common import redact_uri, secret_from
from ..model import ImporterError, MissingDependency, Record
from .base import Source

PAGE = 256
_TEXT_KEYS = ("document", "text", "content", "page_content", "data")


class QdrantSource(Source):
    name = "qdrant"
    summary = "Qdrant: local path or server/cloud URL, all or selected collections (scroll by point id)"
    seekable = True  # scroll is ordered by point id; the cursor is the last imported point id

    def __init__(
        self,
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collections: Optional[List[str]] = None,
        vector_name: Optional[str] = None,
        text_field: Optional[str] = None,
        metadata_field: Optional[str] = None,
        namespace_field: Optional[str] = None,
        created_at_field: Optional[str] = None,
        updated_at_field: Optional[str] = None,
        page_size: int = PAGE,
        timeout: int = 60,
    ):
        if bool(path) == bool(url):
            raise ImporterError("give exactly one of --path (local) or --url (server)")
        self.path, self.url, self.api_key = path, url, api_key
        self.collections = list(collections or [])
        self.vector_name, self.text_field, self.metadata_field = vector_name, text_field, metadata_field
        self.namespace_field = namespace_field
        self.created_at_field, self.updated_at_field = created_at_field, updated_at_field
        self.page_size, self.timeout = page_size, timeout
        self._client: Any = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--path", help="local Qdrant storage directory")
        parser.add_argument("--url", help="Qdrant server or cloud URL")
        parser.add_argument("--collection", action="append", default=[], help="collection, repeatable (default: all)")
        parser.add_argument("--vector-name", help="named vector to carry (required when a collection has several)")
        parser.add_argument("--text-field", help=f"payload key holding the text (default: first of {', '.join(_TEXT_KEYS)})")
        parser.add_argument("--metadata-field", help="payload key holding a nested metadata object to flatten "
                            "(auto 'metadata' for LangChain-style page_content payloads)")
        parser.add_argument("--namespace-field", help="payload key whose value is the namespace (default: the collection name)")
        parser.add_argument("--created-at-field", help="payload key with the creation time (copied, also kept in metadata)")
        parser.add_argument("--updated-at-field", help="payload key with the update time (copied, also kept in metadata)")
        parser.add_argument("--page-size", type=int, default=PAGE)
        parser.epilog = "API key: $QDRANT_API_KEY."

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(args.path, args.url, secret_from(None, env, "QDRANT_API_KEY"), args.collection, args.vector_name,
                   args.text_field, args.metadata_field, args.namespace_field, args.created_at_field,
                   args.updated_at_field, args.page_size)

    def describe(self) -> Dict[str, Any]:
        return {"path": self.path, "url": redact_uri(self.url) if self.url else None,
                "collections": sorted(self.collections), "vector_name": self.vector_name,
                "text_field": self.text_field, "metadata_field": self.metadata_field,
                "namespace_field": self.namespace_field}

    def _connect(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError:
                raise MissingDependency("qdrant", "qdrant-client", "import-qdrant") from None
            if self.path:
                self._client = QdrantClient(path=self.path)
            else:
                self._client = QdrantClient(url=self.url, api_key=self.api_key, timeout=self.timeout,
                                            check_compatibility=False)
        return self._client

    def _names(self) -> List[str]:
        if self.collections:
            return sorted(self.collections)
        return sorted(c.name for c in self._connect().get_collections().collections)

    def count(self) -> Optional[int]:
        client = self._connect()
        return sum(client.count(n, exact=True).count for n in self._names())

    def _vector(self, raw: Any) -> Any:
        if raw is None or isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            if self.vector_name:
                return self._vec_value(raw.get(self.vector_name))
            dense = {k: v for k, v in raw.items() if isinstance(v, list) and (not v or not isinstance(v[0], list))}
            if len(dense) == 1:
                return next(iter(dense.values()))
            if not dense:
                return None
            raise ImporterError(f"collection has several named vectors {sorted(dense)}; choose one with --vector-name")
        return None

    @staticmethod
    def _vec_value(v: Any) -> Any:
        return v if isinstance(v, list) and (not v or not isinstance(v[0], list)) else None

    def records(self, cursor: Any = None) -> Iterator[Record]:
        client = self._connect()
        names = self._names()
        # cursor = [collection, last point id]
        start_name, last = (cursor if cursor else (None, None))
        for name in names:
            if start_name is not None and name < start_name:
                continue
            resume = last if name == start_name else None
            offset: Any = resume
            while True:
                points, nxt = client.scroll(name, limit=self.page_size, offset=offset, with_payload=True,
                                            with_vectors=True)
                for p in points:
                    if resume is not None and p.id == resume:
                        continue  # scroll's offset is inclusive
                    yield self._record(name, p)
                if nxt is None or not points:
                    break
                offset = nxt
                resume = None

    def _record(self, collection: str, p: Any) -> Record:
        payload = dict(p.payload or {})
        tkey = self.text_field or next((k for k in _TEXT_KEYS if k in payload), None)
        text = payload.pop(tkey, None) if tkey else None
        if text is not None and not isinstance(text, str):
            text = str(text)
        nested_key = self.metadata_field or ("metadata" if tkey == "page_content" else None)
        nested = payload.pop(nested_key, None) if nested_key else None
        meta = dict(payload)
        if isinstance(nested, dict):
            meta.update(nested)
        elif nested is not None:
            meta[nested_key] = nested
        ns = meta.get(self.namespace_field) if self.namespace_field else None
        return Record(
            id=str(p.id),
            text=text,
            vector=self._vector(p.vector),
            metadata=meta,
            namespace=str(ns) if ns not in (None, "") else collection,
            created_at=meta.get(self.created_at_field) if self.created_at_field else None,
            updated_at=meta.get(self.updated_at_field) if self.updated_at_field else None,
            cursor=[collection, p.id],
        )

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
