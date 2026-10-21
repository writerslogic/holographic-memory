# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Key-value store, docstore and index store over Holographic Memory.

Each value is one record whose metadata carries the JSON value, so a value must fit the
service's 64 KiB metadata limit (a larger one is rejected with `HmsError`, status 422).
`get_all` enumerates a collection through `buckets` hash partitions of at most `max_fetch`
records each; a bucket that is full raises `HmsError` rather than return a partial
collection. Capacity per collection is therefore about `buckets * max_fetch` entries when
keys hash evenly (default 8 * 100). Raise `buckets` for larger docstores.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import zlib
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.index_store.keyval_index_store import KVIndexStore
from llama_index.core.storage.kvstore.types import DEFAULT_COLLECTION, BaseKVStore

from ..core import AsyncHmsBackend, HashingEmbedder, HmsBackend, HmsConfig, HmsError, Record

_COLLECTION = "kv_collection"
_KEY = "kv_key"
_BUCKET = "kv_bucket"
_VALUE = "kv_value"


def _bucket(key: str, buckets: int) -> int:
    return zlib.crc32(key.encode("utf-8")) % buckets


def _record_id(collection: str, key: str) -> str:
    raw = f"kv/{collection}/{key}"
    if len(raw.encode("utf-8")) <= 200:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"kv/#{digest}"


class HolographicKVStore(BaseKVStore):
    """`BaseKVStore` whose entries are Holographic Memory records.

    Entries carry an explicit hashed-key vector, so no embedding model is involved. Put it
    under `KVDocumentStore`/`KVIndexStore` (see `HolographicDocumentStore`) or any other
    consumer of `BaseKVStore`.
    """

    def __init__(self, config: Optional[HmsConfig] = None, *, buckets: int = 8, **hms_kwargs: Any) -> None:
        if buckets < 1:
            raise ValueError("buckets must be >= 1")
        base = config or HmsConfig(**hms_kwargs)
        if config is not None and hms_kwargs:
            raise TypeError("pass either config or keyword arguments, not both")
        self._config = dataclasses.replace(base, embedder=HashingEmbedder(base.input_dim), text_queries=False)
        self._hash = HashingEmbedder(self._config.input_dim)
        self.buckets = buckets
        self._sync: Optional[HmsBackend] = None
        self._async: Optional[AsyncHmsBackend] = None

    @property
    def backend(self) -> HmsBackend:
        if self._sync is None:
            self._sync = HmsBackend(self._config)
        return self._sync

    @property
    def abackend(self) -> AsyncHmsBackend:
        if self._async is None:
            self._async = AsyncHmsBackend(self._config)
        return self._async

    def close(self) -> None:
        if self._sync is not None:
            self._sync.close()
            self._sync = None

    async def aclose(self) -> None:
        if self._async is not None:
            await self._async.aclose()
            self._async = None

    def _record(self, key: str, val: dict, collection: str) -> Record:
        return Record(
            id=_record_id(collection, key),
            text=key,
            metadata={_COLLECTION: collection, _KEY: key, _BUCKET: _bucket(key, self.buckets), _VALUE: val},
            vector=self._hash.embed_query(key),
        )

    def _page_filter(self, collection: str, bucket: int) -> Dict[str, Any]:
        return {_COLLECTION: collection, _BUCKET: bucket}

    def _collect(self, collection: str, pages: List[Any]) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        for page in pages:
            if page.truncated:
                raise HmsError(
                    f"collection {collection!r} has a bucket with more than max_fetch="
                    f"{self._config.max_fetch} entries; raise buckets (now {self.buckets})"
                )
            for rec in page.records:
                out[rec.metadata[_KEY]] = rec.metadata[_VALUE]
        return out

    # --- sync ---

    def put(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        self.backend.remember([self._record(key, val, collection)])

    def put_all(self, kv_pairs: List[Tuple[str, dict]], collection: str = DEFAULT_COLLECTION, batch_size: int = 1) -> None:
        self.backend.remember([self._record(k, v, collection) for k, v in kv_pairs])

    def get(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        rec = self.backend.get([_record_id(collection, key)])[0]
        return None if rec is None else rec.metadata[_VALUE]

    def get_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        pages = [self.backend.list(self._page_filter(collection, b)) for b in range(self.buckets)]
        return self._collect(collection, pages)

    def delete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        return self.backend.forget([_record_id(collection, key)]) == 1

    # --- async ---

    async def aput(self, key: str, val: dict, collection: str = DEFAULT_COLLECTION) -> None:
        await self.abackend.remember([self._record(key, val, collection)])

    async def aput_all(self, kv_pairs: List[Tuple[str, dict]], collection: str = DEFAULT_COLLECTION, batch_size: int = 1) -> None:
        await self.abackend.remember([self._record(k, v, collection) for k, v in kv_pairs])

    async def aget(self, key: str, collection: str = DEFAULT_COLLECTION) -> Optional[dict]:
        rec = (await self.abackend.get([_record_id(collection, key)]))[0]
        return None if rec is None else rec.metadata[_VALUE]

    async def aget_all(self, collection: str = DEFAULT_COLLECTION) -> Dict[str, dict]:
        pages = await asyncio.gather(*(self.abackend.list(self._page_filter(collection, b)) for b in range(self.buckets)))
        return self._collect(collection, list(pages))

    async def adelete(self, key: str, collection: str = DEFAULT_COLLECTION) -> bool:
        return await self.abackend.forget([_record_id(collection, key)]) == 1


class HolographicDocumentStore(KVDocumentStore):
    """`KVDocumentStore` persisted in Holographic Memory. See the module note for limits."""

    def __init__(self, config: Optional[HmsConfig] = None, *, namespace: Optional[str] = None, buckets: int = 8, **hms_kwargs: Any):
        self.kvstore = HolographicKVStore(config, buckets=buckets, **hms_kwargs)
        super().__init__(self.kvstore, namespace=namespace)


class HolographicIndexStore(KVIndexStore):
    """`KVIndexStore` persisted in Holographic Memory."""

    def __init__(self, config: Optional[HmsConfig] = None, *, namespace: Optional[str] = None, buckets: int = 8, **hms_kwargs: Any):
        self.kvstore = HolographicKVStore(config, buckets=buckets, **hms_kwargs)
        super().__init__(self.kvstore, namespace=namespace)
