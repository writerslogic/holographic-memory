# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""OpenAI vector stores (the Assistants/Responses file_search stores).

OpenAI does not expose embeddings, so this source is text-only: each parsed chunk of
each file becomes a record and WARM encodes it. Endpoints used (all documented):
GET /v1/vector_stores, /v1/vector_stores/{id}/files, /v1/vector_stores/{id}/files/{file_id}/content,
GET /v1/files/{file_id} (file name)."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ..common import register_secret, secret_from
from ..model import ImporterError, MissingDependency, Record
from .base import Source


class OpenAiVectorStoreSource(Source):
    name = "openai-vector-store"
    summary = "OpenAI vector stores: every file's parsed text chunks (text only; OpenAI exposes no embeddings)"

    def __init__(self, api_key: str, base_url: Optional[str], stores: List[str], namespace_by: str,
                 organization: Optional[str] = None, page_size: int = 100):
        if namespace_by not in ("id", "name"):
            raise ImporterError("--namespace-by must be id or name")
        try:
            import openai  # noqa: F401
        except ImportError:
            raise MissingDependency("openai-vector-store", "openai", "import-openai") from None
        self.base_url, self.stores, self.namespace_by = base_url, stores, namespace_by
        self.organization, self.page_size = organization, page_size
        register_secret(api_key)
        self._api_key = api_key
        self._client = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--vector-store", action="append", default=[], dest="stores",
                            help="vector store id, repeatable (default: every store the key can see)")
        parser.add_argument("--base-url", default=None, help="API base URL (default: OpenAI, or $OPENAI_BASE_URL)")
        parser.add_argument("--organization", default=None)
        parser.add_argument("--namespace-by", default="id", choices=["id", "name"],
                            help="namespace = vector store id (default) or its name")
        parser.add_argument("--page-size", type=int, default=100)
        parser.epilog = "Credential: $OPENAI_API_KEY."

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        key = secret_from(None, dict(env), "OPENAI_API_KEY")
        if not key:
            raise ImporterError("set OPENAI_API_KEY")
        return cls(key, args.base_url or env.get("OPENAI_BASE_URL"), args.stores, args.namespace_by,
                   args.organization, args.page_size)

    def describe(self) -> Dict[str, Any]:
        return {"base_url": self.base_url, "stores": sorted(self.stores), "namespace_by": self.namespace_by}

    def _c(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(api_key=self._api_key, base_url=self.base_url,
                                         organization=self.organization, max_retries=5)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def _stores(self) -> List[Any]:
        c = self._c()
        if self.stores:
            return [c.vector_stores.retrieve(s) for s in sorted(self.stores)]
        return sorted(c.vector_stores.list(limit=self.page_size, order="asc"), key=lambda s: s.id)

    def records(self, cursor: Any = None) -> Iterator[Record]:
        c = self._c()
        for store in self._stores():
            ns = store.id if self.namespace_by == "id" else (store.name or store.id)
            for vf in c.vector_stores.files.list(store.id, limit=self.page_size, order="asc"):
                try:
                    filename = c.files.retrieve(vf.id).filename
                except Exception:  # the file object may be gone while its chunks remain
                    filename = None
                if vf.status != "completed":
                    yield Record(id=f"{vf.id}", namespace=ns,
                                 error=f"file {vf.id} in {store.id} has status {vf.status}; it has no parsed chunks")
                    continue
                page = c.vector_stores.files.content(vf.id, vector_store_id=store.id)
                if getattr(page, "has_more", False):
                    yield Record(id=f"{vf.id}#truncated", namespace=ns,
                                 error=f"file {vf.id} has more content pages than the API lets a client follow")
                n = 0
                for part in page:
                    n += 1
                    meta: Dict[str, Any] = {
                        "file_id": vf.id, "vector_store_id": store.id, "vector_store_name": store.name,
                        "chunk_index": n - 1, "content_type": getattr(part, "type", "text"),
                    }
                    if filename:
                        meta["filename"] = filename
                    attrs = dict(vf.attributes or {})
                    if attrs:
                        meta.update({k: v for k, v in attrs.items() if k not in meta})
                    if store.metadata:
                        meta["vector_store_metadata"] = dict(store.metadata)
                    yield Record(id=f"{vf.id}#{n - 1}", text=part.text, metadata=meta, namespace=ns,
                                 created_at=vf.created_at)
