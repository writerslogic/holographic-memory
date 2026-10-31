# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Haystack `DocumentStore` backed by WARM."""

from __future__ import annotations

import asyncio
import json
import zlib
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from haystack import default_from_dict, default_to_dict
from haystack.dataclasses import ByteStream, Document, SparseEmbedding
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.filters import document_matches_filter

from writerslogic_warm_integrations.core import (
    AsyncWarmBackend, Embedder, HashingEmbedder, Hit, WarmBackend, WarmConfig, WarmError, Record,
)

from . import _filters

BUCKET_KEY = "hs_bucket"
PAYLOAD_KEY = "hs_payload"
_RESERVED = (BUCKET_KEY, PAYLOAD_KEY)
_BUCKET_SLOTS = 4096


def _hashable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _strip(key: str) -> str:
    return key[5:] if key.startswith("meta.") else key


class WarmDocumentStore:
    """Haystack document store over `warm-server`.

    Args:
        url, tenant_id, collection, timeout: connection and scoping (see `WarmConfig`).
        api_key: `Secret`; default reads `WARM_API_KEY` if set.
        zero_trust_key: optional `Secret` enabling client-side vector masking (obfuscation, not encryption).
        embedding_dim: the service's `--input-dim` (default 384). Used for the default embedder.
        embedder: encoder for documents that carry no embedding and for text queries when
            `text_queries` is false. Default `HashingEmbedder(embedding_dim)` (lexical).
            Not serializable unless it is a `HashingEmbedder`.
        text_queries: send text to the service, which embeds it (needs on-device model stages).
        return_embedding: whether `filter_documents` returns stored embeddings (default True, like the in-memory store).
        list_buckets: enumeration fan-out. The service cannot list, so enumeration queries one
            probe per bucket of at most `max_fetch` records: `count_documents`, `filter_documents`
            and the other whole-store methods work up to roughly `list_buckets * max_fetch`
            documents (fewer if the hash spreads unevenly) and raise `WarmError` past that.
            Fixed for the life of the data: do not change it for an existing tenant.
        overfetch, max_fetch: retrieval over-fetch factor and the service's `--max-top-k`.
    """

    def __init__(
        self,
        *,
        url: str = "http://127.0.0.1:8080",
        api_key: Optional[Secret] = None,
        tenant_id: Optional[str] = None,
        collection: Optional[str] = None,
        zero_trust_key: Optional[Secret] = None,
        embedding_dim: int = 384,
        embedder: Optional[Embedder] = None,
        text_queries: bool = False,
        return_embedding: bool = True,
        list_buckets: int = 16,
        overfetch: int = 5,
        max_fetch: int = 100,
        timeout: float = 30.0,
        transport: Any = None,
    ):
        if not 1 <= list_buckets <= _BUCKET_SLOTS:
            raise ValueError(f"list_buckets must be 1..{_BUCKET_SLOTS}")
        self.url, self.tenant_id, self.collection = url, tenant_id, collection
        self.api_key = api_key if api_key is not None else Secret.from_env_var("WARM_API_KEY", strict=False)
        self.zero_trust_key = zero_trust_key
        self.embedding_dim, self.embedder, self.text_queries = embedding_dim, embedder, text_queries
        self.return_embedding, self.list_buckets = return_embedding, list_buckets
        self.overfetch, self.max_fetch, self.timeout, self.transport = overfetch, max_fetch, timeout, transport
        self._config = WarmConfig(
            url=url,
            api_key=self.api_key.resolve_value() if self.api_key else None,
            tenant_id=tenant_id,
            collection=collection,
            zero_trust_key=(zero_trust_key.resolve_value() if zero_trust_key else None) or "",
            embedder=embedder,
            input_dim=embedding_dim,
            text_queries=text_queries,
            overfetch=overfetch,
            max_fetch=max_fetch,
            timeout=timeout,
            transport=transport,
        )
        self._sync: Optional[WarmBackend] = None
        self._async: Optional[AsyncWarmBackend] = None

    # ---- backends -------------------------------------------------------------------------

    @property
    def backend(self) -> WarmBackend:
        if self._sync is None:
            self._sync = WarmBackend(self._config)
        return self._sync

    @property
    def async_backend(self) -> AsyncWarmBackend:
        if self._async is None:
            self._async = AsyncWarmBackend(replace(self._config, transport=self._async_transport()))
        return self._async

    def _async_transport(self) -> Any:
        import httpx

        return self.transport if isinstance(self.transport, httpx.AsyncBaseTransport) else None

    def close(self) -> None:
        if self._sync is not None:
            self._sync.close()
            self._sync = None

    async def aclose(self) -> None:
        if self._async is not None:
            await self._async.aclose()
            self._async = None

    # ---- serialization --------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        if self.embedder is not None and not isinstance(self.embedder, HashingEmbedder):
            raise ValueError("only HashingEmbedder embedders can be serialized; pass embedder again after from_dict")
        return default_to_dict(
            self,
            url=self.url,
            api_key=self.api_key.to_dict() if self.api_key else None,
            tenant_id=self.tenant_id,
            collection=self.collection,
            zero_trust_key=self.zero_trust_key.to_dict() if self.zero_trust_key else None,
            embedding_dim=self.embedding_dim,
            embedder={"dim": self.embedder.dim, "bigrams": self.embedder.bigrams} if self.embedder else None,
            text_queries=self.text_queries,
            return_embedding=self.return_embedding,
            list_buckets=self.list_buckets,
            overfetch=self.overfetch,
            max_fetch=self.max_fetch,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WarmDocumentStore":
        params = data.get("init_parameters", {})
        deserialize_secrets_inplace(params, keys=["api_key", "zero_trust_key"])
        if params.get("embedder"):
            params["embedder"] = HashingEmbedder(**params["embedder"])
        return default_from_dict(cls, data)

    # ---- document <-> record --------------------------------------------------------------

    @staticmethod
    def _bucket(doc_id: str, buckets: int) -> int:
        return (zlib.crc32(doc_id.encode("utf-8")) % _BUCKET_SLOTS) % buckets

    def _record(self, doc: Document) -> Record:
        if not isinstance(doc, Document):
            raise ValueError("Please provide a list of Documents.")
        for key in _RESERVED:
            if key in doc.meta:
                raise ValueError(f"metadata key {key!r} is reserved by the Holographic document store")
        full = doc.to_dict(flatten=False)
        payload = {
            "content_none": doc.content is None,
            "blob": full.get("blob"),
            "embedding": doc.embedding,
            "sparse_embedding": full.get("sparse_embedding"),
        }
        meta = {**doc.meta, BUCKET_KEY: self._bucket(doc.id, self.list_buckets), PAYLOAD_KEY: payload}
        return Record(id=doc.id, text=doc.content or "", metadata=meta, vector=doc.embedding)

    def _document(self, rid: str, text: str, metadata: Dict[str, Any], score: Optional[float] = None, embedding: bool = True) -> Document:
        meta = dict(metadata)
        meta.pop(BUCKET_KEY, None)
        payload = meta.pop(PAYLOAD_KEY, None) or {}
        blob, sparse = payload.get("blob"), payload.get("sparse_embedding")
        return Document(
            id=rid,
            content=None if payload.get("content_none") else text,
            blob=ByteStream.from_dict(blob) if blob else None,
            meta=meta,
            score=score,
            embedding=payload.get("embedding") if embedding else None,
            sparse_embedding=SparseEmbedding.from_dict(sparse) if sparse else None,
        )

    def _hit_document(self, hit: Hit, embedding: bool) -> Document:
        return self._document(hit.id, hit.text, hit.metadata, hit.score, embedding)

    # ---- enumeration ----------------------------------------------------------------------

    def _bucket_filters(self, pushed: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [{**pushed, BUCKET_KEY: b} for b in range(self.list_buckets)]

    def _overflow(self, pushed: Dict[str, Any]) -> WarmError:
        return WarmError(
            f"more than max_fetch={self.max_fetch} documents share a hash bucket (list_buckets={self.list_buckets}) "
            "for this query; narrow the filter or create the store with a larger list_buckets"
        )

    def _scan(self, filters: Optional[Dict[str, Any]]) -> List[Document]:
        _filters.validate(filters)
        pushed, _ = _filters.pushdown(filters, _RESERVED)
        page = self.backend.list(pushed or None)
        records = page.records
        if page.truncated:
            records = []
            for flt in self._bucket_filters(pushed):
                part = self.backend.list(flt)
                if part.truncated:
                    raise self._overflow(pushed)
                records.extend(part.records)
        return self._select(records, filters)

    async def _ascan(self, filters: Optional[Dict[str, Any]]) -> List[Document]:
        _filters.validate(filters)
        pushed, _ = _filters.pushdown(filters, _RESERVED)
        page = await self.async_backend.list(pushed or None)
        records = page.records
        if page.truncated:
            parts = await asyncio.gather(*(self.async_backend.list(f) for f in self._bucket_filters(pushed)))
            if any(p.truncated for p in parts):
                raise self._overflow(pushed)
            records = [r for p in parts for r in p.records]
        return self._select(records, filters)

    def _select(self, records: Sequence[Record], filters: Optional[Dict[str, Any]]) -> List[Document]:
        docs = [self._document(r.id, r.text, r.metadata) for r in records]
        if filters:
            docs = [d for d in docs if document_matches_filter(filters=filters, document=d)]
        docs.sort(key=lambda d: d.id)
        return docs

    # ---- DocumentStore protocol -----------------------------------------------------------

    def count_documents(self) -> int:
        return len(self._scan(None))

    async def count_documents_async(self) -> int:
        return len(await self._ascan(None))

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        docs = self._scan(filters)
        return docs if self.return_embedding else [replace(d, embedding=None) for d in docs]

    async def filter_documents_async(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        docs = await self._ascan(filters)
        return docs if self.return_embedding else [replace(d, embedding=None) for d in docs]

    def _plan_write(self, documents: List[Document], policy: DuplicatePolicy) -> Tuple[List[Record], List[str]]:
        if not isinstance(documents, list) or any(not isinstance(d, Document) for d in documents):
            raise ValueError("Please provide a list of Documents.")
        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL
        ids = [d.id for d in documents]
        if policy == DuplicatePolicy.OVERWRITE:
            latest = {d.id: d for d in documents}
            return [self._record(d) for d in latest.values()], []
        return [], ids  # caller resolves existing ids, then calls _finish_write

    def _finish_write(self, documents: List[Document], policy: DuplicatePolicy, existing: set) -> List[Record]:
        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL
        out: Dict[str, Document] = {}
        for d in documents:
            if d.id in existing or d.id in out:
                if policy == DuplicatePolicy.FAIL:
                    raise DuplicateDocumentError(f"ID '{d.id}' already exists.")
                continue
            out[d.id] = d
        return [self._record(d) for d in out.values()]

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        records, check = self._plan_write(documents, policy)
        if check:
            existing = {r.id for r in self.backend.get(list(dict.fromkeys(check))) if r}
            records = self._finish_write(documents, policy, existing)
            written = len(records)
        else:
            written = len(documents)
        if records:
            self.backend.remember(records)
        return written

    async def write_documents_async(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        records, check = self._plan_write(documents, policy)
        if check:
            existing = {r.id for r in await self.async_backend.get(list(dict.fromkeys(check))) if r}
            records = self._finish_write(documents, policy, existing)
            written = len(records)
        else:
            written = len(documents)
        if records:
            await self.async_backend.remember(records)
        return written

    def delete_documents(self, document_ids: List[str]) -> None:
        self.backend.forget(document_ids)

    async def delete_documents_async(self, document_ids: List[str]) -> None:
        await self.async_backend.forget(document_ids)

    # ---- optional management API (mirrors InMemoryDocumentStore) --------------------------

    def delete_all_documents(self) -> None:
        self.backend.forget_where(None)

    async def delete_all_documents_async(self) -> None:
        await self.async_backend.forget_where(None)

    def delete_by_filter(self, filters: Dict[str, Any]) -> int:
        ids = [d.id for d in self._scan(filters)]
        self.backend.forget(ids)
        return len(ids)

    async def delete_by_filter_async(self, filters: Dict[str, Any]) -> int:
        ids = [d.id for d in await self._ascan(filters)]
        await self.async_backend.forget(ids)
        return len(ids)

    def update_by_filter(self, filters: Dict[str, Any], meta: Dict[str, Any]) -> int:
        docs = self._scan(filters)
        for d in docs:
            d.meta.update(meta)
        if docs:
            self.write_documents(docs, DuplicatePolicy.OVERWRITE)
        return len(docs)

    async def update_by_filter_async(self, filters: Dict[str, Any], meta: Dict[str, Any]) -> int:
        docs = await self._ascan(filters)
        for d in docs:
            d.meta.update(meta)
        if docs:
            await self.write_documents_async(docs, DuplicatePolicy.OVERWRITE)
        return len(docs)

    def count_documents_by_filter(self, filters: Dict[str, Any]) -> int:
        return len(self._scan(filters))

    async def count_documents_by_filter_async(self, filters: Dict[str, Any]) -> int:
        return len(await self._ascan(filters))

    @staticmethod
    def _unique_counts(docs: List[Document], fields: List[str]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for f in fields:
            key = _strip(f)
            out[key] = len({_hashable(d.meta[key]) for d in docs if d.meta.get(key) is not None})
        return out

    def count_unique_metadata_by_filter(self, filters: Dict[str, Any], metadata_fields: List[str]) -> Dict[str, int]:
        return self._unique_counts(self._scan(filters), metadata_fields)

    async def count_unique_metadata_by_filter_async(self, filters: Dict[str, Any], metadata_fields: List[str]) -> Dict[str, int]:
        return self._unique_counts(await self._ascan(filters), metadata_fields)

    @staticmethod
    def _fields_info(docs: List[Document]) -> Dict[str, Dict[str, str]]:
        types: Dict[str, str] = {}
        for d in docs:
            for k, v in d.meta.items():
                if v is None:
                    continue
                types[k] = "boolean" if isinstance(v, bool) else "int" if isinstance(v, int) else "float" if isinstance(v, float) else "keyword"
        return {k: {"type": t} for k, t in types.items()}

    def get_metadata_fields_info(self) -> Dict[str, Dict[str, str]]:
        return self._fields_info(self._scan(None))

    async def get_metadata_fields_info_async(self) -> Dict[str, Dict[str, str]]:
        return self._fields_info(await self._ascan(None))

    @staticmethod
    def _min_max(docs: List[Document], field: str) -> Dict[str, Any]:
        key = _strip(field)
        values = [d.meta[key] for d in docs if isinstance(d.meta.get(key), (int, float, str)) and not isinstance(d.meta.get(key), bool)]
        try:
            return {"min": min(values), "max": max(values)} if values else {"min": None, "max": None}
        except TypeError:
            return {"min": None, "max": None}

    def get_metadata_field_min_max(self, metadata_field: str) -> Dict[str, Any]:
        return self._min_max(self._scan(None), metadata_field)

    async def get_metadata_field_min_max_async(self, metadata_field: str) -> Dict[str, Any]:
        return self._min_max(await self._ascan(None), metadata_field)

    @staticmethod
    def _unique_values(docs: List[Document], field: str, search_term: Optional[str], from_: int, size: int) -> Tuple[List[Any], int]:
        key = _strip(field)
        unique: Dict[Tuple[str, str], Any] = {}
        for d in docs:
            v = d.meta.get(key)
            if v is not None:
                unique.setdefault((type(v).__name__, _hashable(v)), v)
        if search_term:
            unique = {k: v for k, v in unique.items() if search_term.lower() in str(v).lower()}
        keys = sorted(unique, key=lambda k: (str(unique[k]), k[0]))
        return [unique[k] for k in keys[from_ : from_ + size]], len(keys)

    def get_metadata_field_unique_values(
        self, metadata_field: str, search_term: Optional[str] = None, from_: int = 0, size: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], int]:
        return self._unique_values(self._scan(filters), metadata_field, search_term, from_, size)

    async def get_metadata_field_unique_values_async(
        self, metadata_field: str, search_term: Optional[str] = None, from_: int = 0, size: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], int]:
        return self._unique_values(await self._ascan(filters), metadata_field, search_term, from_, size)

    # ---- retrieval (used by the retriever components) -------------------------------------

    def _fetch_size(self, filters: Optional[Dict[str, Any]], top_k: int) -> Tuple[Dict[str, Any], int]:
        _filters.validate(filters)
        pushed, complete = _filters.pushdown(filters, _RESERVED)
        fetch = top_k if complete else min(self.max_fetch, top_k * self.overfetch)
        return pushed, max(fetch, top_k)

    def _finish(self, hits: List[Hit], filters: Optional[Dict[str, Any]], top_k: int, scale: bool, embedding: bool) -> List[Document]:
        docs = [self._hit_document(h, embedding) for h in hits]
        if filters:
            docs = [d for d in docs if document_matches_filter(filters=filters, document=d)]
        docs = docs[:top_k]
        if scale and not self.text_queries:
            docs = [replace(d, score=(d.score + 1.0) / 2.0) for d in docs]
        return docs

    def embedding_retrieval(
        self, query_embedding: List[float], filters: Optional[Dict[str, Any]] = None, top_k: int = 10,
        scale_score: bool = False, return_embedding: bool = False,
    ) -> List[Document]:
        """Nearest documents by cosine similarity; `score` is the cosine (or `(cos+1)/2` when scaled)."""
        if top_k == 0:
            return []
        pushed, fetch = self._fetch_size(filters, top_k)
        hits = self.backend.recall(vector=list(query_embedding), k=fetch, filter=pushed or None)
        return self._finish(hits, filters, top_k, scale_score, return_embedding)

    async def embedding_retrieval_async(
        self, query_embedding: List[float], filters: Optional[Dict[str, Any]] = None, top_k: int = 10,
        scale_score: bool = False, return_embedding: bool = False,
    ) -> List[Document]:
        if top_k == 0:
            return []
        pushed, fetch = self._fetch_size(filters, top_k)
        hits = await self.async_backend.recall(vector=list(query_embedding), k=fetch, filter=pushed or None)
        return self._finish(hits, filters, top_k, scale_score, return_embedding)

    def text_retrieval(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, return_embedding: bool = False,
    ) -> List[Document]:
        """Search by text. The query is embedded with the store's embedder, or by the service when `text_queries` is set.

        `score` is cosine similarity with a client embedder and the service's fused rank score with `text_queries`.
        """
        if top_k == 0:
            return []
        pushed, fetch = self._fetch_size(filters, top_k)
        hits = self.backend.recall(query, k=fetch, filter=pushed or None)
        return self._finish(hits, filters, top_k, False, return_embedding)

    async def text_retrieval_async(
        self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 10, return_embedding: bool = False,
    ) -> List[Document]:
        if top_k == 0:
            return []
        pushed, fetch = self._fetch_size(filters, top_k)
        hits = await self.async_backend.recall(query, k=fetch, filter=pushed or None)
        return self._finish(hits, filters, top_k, False, return_embedding)
