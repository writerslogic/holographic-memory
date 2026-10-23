# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""MongoDB (Atlas Vector Search collections): documents read by _id keyset paging."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ..common import redact_uri, uri_secret
from ..model import ImporterError, MissingDependency, Record
from .base import Source, parse_cli_json

_PAGE = 500


def encode_id(value: Any) -> Any:
    """JSON form of an _id that decode_id restores exactly (ObjectId, int, str)."""
    cls = type(value).__name__
    if cls == "ObjectId":
        return {"oid": str(value)}
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return {"v": value}
    raise ImporterError(f"unsupported _id type {cls}; only ObjectId, string and integer _ids can be paged")


def decode_id(token: Any) -> Any:
    if "oid" in token:
        from bson import ObjectId

        return ObjectId(token["oid"])
    return token["v"]


def _vector(value: Any) -> Any:
    as_vector = getattr(value, "as_vector", None)  # bson Binary vector (subtype 9)
    if callable(as_vector):
        return list(as_vector().data)
    return value


class MongoDbSource(Source):
    name = "mongodb"
    summary = "MongoDB / Atlas Vector Search collection (read by _id keyset paging; resumable)"
    seekable = True

    def __init__(self, uri: str, db: str, collection: str, vector_field: str, text_field: Optional[str],
                 id_field: str, namespace_field: Optional[str], created_field: Optional[str],
                 updated_field: Optional[str], metadata_fields: Optional[List[str]], flt: Optional[Dict[str, Any]],
                 page_size: int = _PAGE, collection_obj: Any = None):
        self.uri, self.db, self.collection = uri, db, collection
        self.vector_field, self.text_field, self.id_field = vector_field, text_field, id_field
        self.namespace_field, self.created_field, self.updated_field = namespace_field, created_field, updated_field
        self.metadata_fields, self.filter, self.page = metadata_fields, flt or {}, page_size
        self._coll = collection_obj
        self._client = None
        if id_field != "_id":
            raise ImporterError("only _id is supported as the paging key; use --id-field _id and map another field via metadata")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--uri", help="mongodb:// or mongodb+srv:// URI (default $MONGODB_URI); embedded password is never logged")
        parser.add_argument("--db", required=True)
        parser.add_argument("--collection", required=True)
        parser.add_argument("--vector-field", default="embedding", help="array (or BSON vector) field; default embedding")
        parser.add_argument("--text-field", default="text")
        parser.add_argument("--id-field", default="_id", help="document id field (must be _id)")
        parser.add_argument("--namespace-field")
        parser.add_argument("--created-field")
        parser.add_argument("--updated-field")
        parser.add_argument("--metadata-fields", help="comma-separated; default every other field")
        parser.add_argument("--filter", help="JSON query filter limiting the documents read")
        parser.add_argument("--page-size", type=int, default=_PAGE)

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        uri = args.uri or env.get("MONGODB_URI")
        if not uri:
            raise ImporterError("give --uri or set MONGODB_URI")
        uri_secret(uri)
        meta = [m.strip() for m in args.metadata_fields.split(",")] if args.metadata_fields else None
        return cls(uri, args.db, args.collection, args.vector_field, args.text_field, args.id_field,
                   args.namespace_field, args.created_field, args.updated_field, meta,
                   parse_cli_json(args.filter, "--filter"), args.page_size)

    def describe(self) -> Dict[str, Any]:
        return {"uri": redact_uri(self.uri), "db": self.db, "collection": self.collection,
                "fields": [self.vector_field, self.text_field, self.namespace_field, self.created_field,
                           self.updated_field, self.metadata_fields], "filter": self.filter}

    def _coll_obj(self):
        if self._coll is None:
            try:
                import pymongo
            except ImportError:
                raise MissingDependency("mongodb", "pymongo", "import-mongodb") from None
            self._client = pymongo.MongoClient(self.uri)
            self._coll = self._client[self.db][self.collection]
        return self._coll

    def count(self) -> Optional[int]:
        return int(self._coll_obj().count_documents(self.filter))

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def records(self, cursor: Any = None) -> Iterator[Record]:
        coll = self._coll_obj()
        last = decode_id(cursor) if cursor is not None else None
        while True:
            q: Dict[str, Any] = dict(self.filter)
            if last is not None:
                q = {"$and": [q, {"_id": {"$gt": last}}]} if q else {"_id": {"$gt": last}}
            docs = list(coll.find(q).sort("_id", 1).limit(self.page))
            if not docs:
                return
            for doc in docs:
                last = doc["_id"]
                yield self._record(doc)
            if len(docs) < self.page:
                return

    def _record(self, doc: Dict[str, Any]) -> Record:
        try:
            token = encode_id(doc["_id"])
        except ImporterError as exc:
            return Record(id=str(doc["_id"]), error=str(exc))
        used = {"_id"}

        def take(name: Optional[str]) -> Any:
            if name and name in doc:
                used.add(name)
                return doc[name]
            return None

        vec = _vector(take(self.vector_field))
        text = take(self.text_field)
        ns, created, updated = take(self.namespace_field), take(self.created_field), take(self.updated_field)
        if self.metadata_fields is not None:
            meta = {f: doc[f] for f in self.metadata_fields if f in doc}
        else:
            meta = {f: v for f, v in doc.items() if f not in used}
        ident = str(doc["_id"])
        return Record(id=ident, text=None if text is None else str(text), vector=vec,
                      metadata={k: _plain(v) for k, v in meta.items()}, namespace="" if ns is None else str(ns),
                      created_at=created, updated_at=updated, cursor=token)


def _plain(value: Any) -> Any:
    """BSON scalars without a JSON form (ObjectId, Decimal128, Int64 handled by int) become strings."""
    if type(value).__name__ in ("ObjectId", "Decimal128", "Regex", "Timestamp"):
        return str(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


__all__ = ["MongoDbSource"]
