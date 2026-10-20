# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Milvus and Milvus Lite collections through pymilvus (query iterator, partitions as namespaces)."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ..common import redact_uri, secret_from, uri_secret
from ..model import ImporterError, MissingDependency, Record
from .base import Source

_TEXT_NAMES = ("text", "content", "document", "chunk", "body", "page_content")
_FLOAT_VECTOR = 101


def _quote(value: Any) -> str:
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return str(value)


class MilvusSource(Source):
    name = "milvus"
    summary = "A Milvus (server/Zilliz) or Milvus Lite collection; partitions become namespaces, dynamic fields metadata"
    seekable = True

    def __init__(self, uri: str, collection: str, token: Optional[str] = None, user: Optional[str] = None,
                 password: Optional[str] = None, db_name: Optional[str] = None, vector_field: Optional[str] = None,
                 text_field: Optional[str] = None, namespace_field: Optional[str] = None,
                 partitions: Optional[List[str]] = None, created_at_field: Optional[str] = None,
                 updated_at_field: Optional[str] = None, batch_size: int = 1000):
        uri_secret(uri)
        self.uri, self.collection, self.token, self.user, self.password = uri, collection, token, user, password
        self.db_name, self.vector_field, self.text_field = db_name, vector_field, text_field
        self.namespace_field, self.partitions = namespace_field, partitions or None
        self.created_at_field, self.updated_at_field = created_at_field, updated_at_field
        self.batch_size = max(1, min(batch_size, 16384))
        self._client: Any = None
        self._layout: Any = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--uri", required=True,
                            help="Milvus Lite database file, or http(s)://host:19530 (server, Zilliz Cloud)")
        parser.add_argument("--collection", required=True)
        parser.add_argument("--token", help="discouraged on the command line; prefer $MILVUS_TOKEN")
        parser.add_argument("--user", help="username ($MILVUS_USER)")
        parser.add_argument("--password", help="discouraged on the command line; prefer $MILVUS_PASSWORD")
        parser.add_argument("--db-name", help="Milvus database name")
        parser.add_argument("--vector-field", help="FLOAT_VECTOR field to read (default: the only one)")
        parser.add_argument("--text-field", help="VARCHAR field holding the text (default: text/content/document/...)")
        parser.add_argument("--namespace-field", help="scalar field to use as the namespace (default: the partition)")
        parser.add_argument("--partition", action="append", help="read only this partition; repeatable")
        parser.add_argument("--created-at-field")
        parser.add_argument("--updated-at-field")
        parser.add_argument("--page-size", type=int, default=1000, help="query iterator batch size")

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(
            args.uri, args.collection,
            token=secret_from(args.token, env, "MILVUS_TOKEN"),
            user=args.user or env.get("MILVUS_USER"),
            password=secret_from(args.password, env, "MILVUS_PASSWORD"),
            db_name=args.db_name, vector_field=args.vector_field, text_field=args.text_field,
            namespace_field=args.namespace_field, partitions=args.partition,
            created_at_field=args.created_at_field, updated_at_field=args.updated_at_field,
            batch_size=args.page_size,
        )

    def describe(self) -> Dict[str, Any]:
        return {"uri": redact_uri(self.uri) if "://" in self.uri else os.path.abspath(self.uri),
                "collection": self.collection, "db_name": self.db_name, "vector_field": self.vector_field,
                "text_field": self.text_field, "namespace_field": self.namespace_field,
                "partitions": self.partitions, "created_at_field": self.created_at_field,
                "updated_at_field": self.updated_at_field}

    # -- access ---------------------------------------------------------------------

    def _connect(self):
        if self._client is not None:
            return self._client
        try:
            from pymilvus import MilvusClient
        except ImportError:
            raise MissingDependency("milvus", "pymilvus", "import-milvus") from None
        kwargs: Dict[str, Any] = {}
        if self.token:
            kwargs["token"] = self.token
        if self.user:
            kwargs["user"] = self.user
        if self.password:
            kwargs["password"] = self.password
        if self.db_name:
            kwargs["db_name"] = self.db_name
        self._client = MilvusClient(self.uri, **kwargs)
        return self._client

    def _schema(self):
        if self._layout is not None:
            return self._layout
        client = self._connect()
        if not client.has_collection(self.collection):
            raise ImporterError(f"collection '{self.collection}' does not exist at {self.describe()['uri']}")
        fields = client.describe_collection(self.collection)["fields"]
        pk = next((f for f in fields if f.get("is_primary")), None)
        if pk is None:
            raise ImporterError("collection has no primary key field")
        vectors = [f for f in fields if int(getattr(f["type"], "value", f["type"])) == _FLOAT_VECTOR]
        if self.vector_field:
            vf = next((f for f in fields if f["name"] == self.vector_field), None)
            if vf is None:
                raise ImporterError(f"no field '{self.vector_field}'; fields: {[f['name'] for f in fields]}")
            if int(getattr(vf["type"], "value", vf["type"])) != _FLOAT_VECTOR:
                raise ImporterError(f"field '{vf['name']}' is not a FLOAT_VECTOR; only dense float vectors are supported")
        elif len(vectors) == 1:
            vf = vectors[0]
        else:
            raise ImporterError(
                f"cannot pick the vector field automatically ({len(vectors)} FLOAT_VECTOR fields; binary, fp16, "
                f"bf16 and sparse vectors are not supported); use --vector-field. Fields: {[f['name'] for f in fields]}")
        tf = self.text_field
        if tf is None:
            names = {f["name"].lower(): f["name"] for f in fields}
            tf = next((names[n] for n in _TEXT_NAMES if n in names), None)
        elif tf not in {f["name"] for f in fields}:
            raise ImporterError(f"no field '{tf}'")
        pk_type = int(getattr(pk["type"], "value", pk["type"]))
        partitions = self.partitions or client.list_partitions(self.collection)
        missing = set(partitions) - set(client.list_partitions(self.collection))
        if missing:
            raise ImporterError(f"unknown partition(s): {sorted(missing)}")
        self._layout = (pk["name"], pk_type, vf["name"], tf, list(partitions))
        return self._layout

    def count(self) -> Optional[int]:
        try:
            client = self._connect()
            _, _, _, _, partitions = self._schema()
            total = 0
            for p in partitions:
                total += int(client.get_partition_stats(self.collection, p)["row_count"])
            return total
        except ImporterError:
            raise
        except Exception:
            return None

    def records(self, cursor: Any = None) -> Iterator[Record]:
        client = self._connect()
        pk, _pk_type, vec_field, text_field, partitions = self._schema()
        start_part, last_pk = (cursor or [0, None])
        for pi in range(start_part, len(partitions)):
            part = partitions[pi]
            flt = f"{pk} > {_quote(last_pk)}" if (pi == start_part and last_pk is not None) else ""
            it = client.query_iterator(self.collection, batch_size=self.batch_size, filter=flt,
                                       output_fields=["*"], partition_names=[part])
            try:
                while True:
                    rows = it.next()
                    if not rows:
                        break
                    for row in rows:
                        yield self._record(row, pk, vec_field, text_field, part, pi)
            finally:
                it.close()

    def _record(self, row: Dict[str, Any], pk: str, vec_field: str, text_field: Optional[str], part: str,
                pi: int) -> Record:
        ident = row.get(pk)
        meta = {k: v for k, v in row.items() if k not in (pk, vec_field, text_field)}
        ns = part if part != "_default" else ""
        if self.namespace_field:
            ns = str(row.get(self.namespace_field) or "")
        text = row.get(text_field) if text_field else None
        return Record(
            id=str(ident), text=text, vector=row.get(vec_field), metadata=meta, namespace=ns,
            created_at=row.get(self.created_at_field) if self.created_at_field else None,
            updated_at=row.get(self.updated_at_field) if self.updated_at_field else None,
            cursor=[pi, ident],
        )

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
