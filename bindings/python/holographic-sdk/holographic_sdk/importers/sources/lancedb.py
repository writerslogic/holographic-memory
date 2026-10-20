# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""LanceDB: a local directory or object-store URI (and LanceDB Cloud with an API key)."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ..common import redact_uri, secret_from
from ..model import ImporterError, MissingDependency, Record
from .base import ColumnMap, Source, add_map_argument

PAGE = 500


class LanceDbSource(Source):
    name = "lancedb"
    summary = "LanceDB tables (local path, s3://, gs:// or db:// cloud), pinned to one table version while reading"
    seekable = False

    def __init__(self, uri: str, tables: List[str], vector_column: Optional[str], cmap: ColumnMap,
                 api_key: Optional[str] = None, region: Optional[str] = None, page_size: int = PAGE):
        self.uri, self.tables, self.vector_column, self.cmap = uri, list(tables), vector_column, cmap
        self.api_key, self.region, self.page_size = api_key, region, page_size
        self._db: Any = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--uri", required=True, help="LanceDB directory or URI")
        parser.add_argument("--table", action="append", default=[], help="table name, repeatable (default: all tables)")
        parser.add_argument("--vector-column", help="vector column (default: the first fixed-size list column)")
        parser.add_argument("--region", help="LanceDB Cloud region")
        parser.add_argument("--page-size", type=int, default=PAGE)
        add_map_argument(parser)
        parser.epilog = "LanceDB Cloud key: $LANCEDB_API_KEY. Namespace is the table name unless --map namespace=COLUMN."

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(args.uri, args.table, args.vector_column, ColumnMap(args.map),
                   secret_from(None, env, "LANCEDB_API_KEY"), args.region, args.page_size)

    def describe(self) -> Dict[str, Any]:
        return {"uri": redact_uri(self.uri), "tables": sorted(self.tables), "vector_column": self.vector_column,
                "map": self.cmap.describe()}

    def _connect(self):
        if self._db is None:
            try:
                import lancedb
            except ImportError:
                raise MissingDependency("lancedb", "lancedb", "import-lancedb") from None
            kw: Dict[str, Any] = {}
            if self.api_key:
                kw["api_key"] = self.api_key
            if self.region:
                kw["region"] = self.region
            self._db = lancedb.connect(self.uri, **kw)
        return self._db

    def _table_names(self) -> List[str]:
        if self.tables:
            return sorted(self.tables)
        res = self._connect().table_names()
        return sorted(getattr(res, "tables", res))

    def _open(self, name: str):
        tbl = self._connect().open_table(name)
        try:
            tbl.checkout(tbl.version)  # one snapshot for the whole read
        except Exception:  # remote tables may not support checkout; read the live table
            pass
        return tbl

    def count(self) -> Optional[int]:
        return sum(self._open(n).count_rows() for n in self._table_names())

    def _vector_column(self, schema) -> str:
        import pyarrow as pa

        if self.vector_column:
            if self.vector_column not in schema.names:
                raise ImporterError(f"vector column '{self.vector_column}' not in {schema.names}")
            return self.vector_column
        for field in schema:
            if pa.types.is_fixed_size_list(field.type) or (
                pa.types.is_list(field.type) and pa.types.is_floating(field.type.value_type)
            ):
                return field.name
        return ""

    def records(self, cursor: Any = None) -> Iterator[Record]:
        for name in self._table_names():
            tbl = self._open(name)
            vcol = self._vector_column(tbl.schema)
            spec = [f"{k}={','.join(v) if isinstance(v, list) else v}" for k, v in self.cmap.describe().items()]
            if "id" not in self.cmap.cols and "id" in tbl.schema.names:
                spec.append("id=id")
            cmap = ColumnMap(spec)
            offset, ordinal = 0, 0
            while True:
                rows = tbl.search().limit(self.page_size).offset(offset).to_arrow().to_pylist()
                if not rows:
                    break
                offset += len(rows)
                for row in rows:
                    ordinal += 1
                    vec = row.pop(vcol, None) if vcol else None
                    row.pop("_distance", None)
                    rec = cmap.record(row, ordinal)
                    rec.vector = vec
                    if not rec.namespace:
                        rec.namespace = name
                    yield rec

    def close(self) -> None:
        self._db = None
