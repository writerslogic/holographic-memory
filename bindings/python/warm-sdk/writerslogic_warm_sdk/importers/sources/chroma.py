# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Chroma: a persistent directory or an HTTP server, every collection."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ..common import redact_uri, secret_from
from ..model import ImporterError, MissingDependency, Record
from .base import Source

PAGE = 500


class ChromaSource(Source):
    name = "chroma"
    summary = "Chroma: persistent directory (--path) or server (--host), all or selected collections"
    seekable = False  # Chroma offsets follow insertion order but are not a keyset

    def __init__(
        self,
        path: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 8000,
        ssl: bool = False,
        api_key: Optional[str] = None,
        tenant: Optional[str] = None,
        database: Optional[str] = None,
        collections: Optional[List[str]] = None,
        created_at_key: Optional[str] = None,
        updated_at_key: Optional[str] = None,
        page_size: int = PAGE,
    ):
        if bool(path) == bool(host):
            raise ImporterError("give exactly one of --path (local) or --host (server)")
        self.path, self.host, self.port, self.ssl = path, host, port, ssl
        self.api_key, self.tenant, self.database = api_key, tenant, database
        self.collections = list(collections or [])
        self.created_at_key, self.updated_at_key = created_at_key, updated_at_key
        self.page_size = page_size
        self._client: Any = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--path", help="Chroma persistent directory")
        parser.add_argument("--host", help="Chroma server host")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--ssl", action="store_true")
        parser.add_argument("--chroma-tenant", dest="tenant")
        parser.add_argument("--chroma-database", dest="database")
        parser.add_argument("--collection", action="append", default=[], help="collection name, repeatable (default: all)")
        parser.add_argument("--created-at-key", help="metadata key holding the creation time (copied to warm_created_at)")
        parser.add_argument("--updated-at-key", help="metadata key holding the update time (copied to warm_updated_at)")
        parser.add_argument("--page-size", type=int, default=PAGE)
        parser.epilog = "Server token: $CHROMA_API_KEY (sent as X-Chroma-Token)."

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(args.path, args.host, args.port, args.ssl, secret_from(None, env, "CHROMA_API_KEY"),
                   args.tenant, args.database, args.collection, args.created_at_key, args.updated_at_key,
                   args.page_size)

    def describe(self) -> Dict[str, Any]:
        return {"path": self.path, "host": redact_uri(f"http://{self.host}:{self.port}") if self.host else None,
                "tenant": self.tenant, "database": self.database, "collections": sorted(self.collections)}

    def _connect(self):
        if self._client is not None:
            return self._client
        try:
            import chromadb
        except ImportError:
            raise MissingDependency("chroma", "chromadb", "import-chroma") from None
        kw: Dict[str, Any] = {}
        if self.tenant:
            kw["tenant"] = self.tenant
        if self.database:
            kw["database"] = self.database
        if self.path:
            self._client = chromadb.PersistentClient(path=self.path, **kw)
        else:
            headers = {"X-Chroma-Token": self.api_key} if self.api_key else None
            self._client = chromadb.HttpClient(host=self.host, port=self.port, ssl=self.ssl, headers=headers, **kw)
        return self._client

    def _collections(self) -> List[Any]:
        client = self._connect()
        names = self.collections or sorted(getattr(c, "name", c) for c in client.list_collections())
        return [client.get_collection(n) for n in names]

    def count(self) -> Optional[int]:
        return sum(c.count() for c in self._collections())

    def records(self, cursor: Any = None) -> Iterator[Record]:
        for col in self._collections():
            offset = 0
            while True:
                page = col.get(limit=self.page_size, offset=offset,
                               include=["documents", "metadatas", "embeddings", "uris"])
                ids = page["ids"]
                if not len(ids):
                    break
                docs, metas, embs, uris = page.get("documents"), page.get("metadatas"), page.get("embeddings"), page.get("uris")
                for i, ident in enumerate(ids):
                    meta = dict(metas[i]) if metas is not None and metas[i] else {}
                    if uris is not None and uris[i]:
                        meta["chroma_uri"] = uris[i]
                    yield Record(
                        id=str(ident),
                        text=docs[i] if docs is not None else None,
                        vector=embs[i] if embs is not None else None,
                        metadata=meta,
                        namespace=col.name,
                        created_at=meta.get(self.created_at_key) if self.created_at_key else None,
                        updated_at=meta.get(self.updated_at_key) if self.updated_at_key else None,
                    )
                offset += len(ids)

    def close(self) -> None:
        self._client = None
