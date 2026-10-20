# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""sqlite-vec: a vec0 virtual table, optionally joined to an ordinary side table for text."""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ..model import ImporterError, MissingDependency, Record
from .base import ColumnMap, Source, add_map_argument

PAGE = 500


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def decode_vector(blob: Any, kind: str) -> Optional[List[float]]:
    """float32, int8 or bit vectors as stored by vec0."""
    if blob is None:
        return None
    import numpy as np

    raw = bytes(blob)
    if kind == "int8":
        return np.frombuffer(raw, dtype=np.int8).astype("float64").tolist()
    if kind == "bit":
        return np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).astype("float64").tolist()
    if len(raw) % 4:
        raise ValueError(f"float32 vector blob has {len(raw)} bytes")
    return np.frombuffer(raw, dtype="<f4").astype("float64").tolist()


class SqliteVecSource(Source):
    name = "sqlite-vec"
    summary = "sqlite-vec vec0 table (read-only), optionally joined to a side table holding text/metadata"
    seekable = True  # keyset on rowid

    def __init__(self, db: str, table: str, vector_column: Optional[str], side_table: Optional[str],
                 side_key: str, cmap: ColumnMap, namespace: str = "", page_size: int = PAGE):
        if not os.path.isfile(db):
            raise ImporterError(f"not a file: {db}")
        self.db, self.table, self.vector_column = db, table, vector_column
        self.side_table, self.side_key, self.cmap = side_table, side_key, cmap
        self.namespace, self.page_size = namespace, page_size
        self._kind = "float"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--db", required=True, help="SQLite database file with the sqlite-vec table")
        parser.add_argument("--table", required=True, help="the vec0 virtual table")
        parser.add_argument("--vector-column", help="vector column (default: the first float[N]/int8[N]/bit[N] column)")
        parser.add_argument("--side-table", help="ordinary table joined on --side-key = vec0 rowid; its columns are mappable")
        parser.add_argument("--side-key", default="rowid", help="side-table column equal to the vec0 rowid")
        parser.add_argument("--namespace", default="", help="namespace for every row (or --map namespace=COLUMN)")
        parser.add_argument("--page-size", type=int, default=PAGE)
        add_map_argument(parser)

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(args.db, args.table, args.vector_column, args.side_table, args.side_key,
                   ColumnMap(args.map), args.namespace, args.page_size)

    def describe(self) -> Dict[str, Any]:
        return {"db": os.path.abspath(self.db), "table": self.table, "vector_column": self.vector_column,
                "side_table": self.side_table, "side_key": self.side_key, "map": self.cmap.describe(),
                "namespace": self.namespace}

    def _connect(self) -> sqlite3.Connection:
        try:
            import sqlite_vec
        except ImportError:
            raise MissingDependency("sqlite-vec", "sqlite-vec", "import-sqlite-vec") from None
        con = sqlite3.connect(f"file:{os.path.abspath(self.db)}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            con.enable_load_extension(True)
        except AttributeError:
            raise ImporterError("this Python's sqlite3 cannot load extensions (needs enable_load_extension)") from None
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        return con

    def _columns(self, con: sqlite3.Connection) -> List[str]:
        row = con.execute("SELECT sql FROM sqlite_master WHERE name = ?", (self.table,)).fetchone()
        if row is None or "vec0" not in (row[0] or "").lower():
            raise ImporterError(f"'{self.table}' is not a vec0 virtual table in {self.db}")
        sql = row[0]
        cols = [r["name"] for r in con.execute(f"PRAGMA table_xinfo({_q(self.table)})") if not r["hidden"] and r["name"] != "rowid"]
        if self.vector_column is None:
            m = re.search(r"([A-Za-z_][\w]*)\s+(float|int8|bit)\s*\[", sql)
            if not m:
                raise ImporterError("could not find a vector column in the vec0 definition; use --vector-column")
            self.vector_column = m.group(1)
        if self.vector_column not in cols:
            raise ImporterError(f"vector column '{self.vector_column}' not in {cols}")
        m = re.search(re.escape(self.vector_column) + r"\s+(float|int8|bit)\s*\[", sql)
        self._kind = m.group(1) if m else "float"
        return cols

    def count(self) -> Optional[int]:
        with self._connect() as con:
            return con.execute(f"SELECT COUNT(*) FROM {_q(self.table)}").fetchone()[0]

    def records(self, cursor: Any = None) -> Iterator[Record]:
        con = self._connect()
        try:
            cols = self._columns(con)
            select = ", ".join(_q(c) for c in cols)
            last = int(cursor) if cursor is not None else -(2**63)
            spec = list(self.cmap.describe().items())
            cmap = self.cmap
            if "id" not in self.cmap.cols:
                cmap = ColumnMap([f"{k}={','.join(v) if isinstance(v, list) else v}" for k, v in spec] + ["id=rowid"])
            while True:
                rows = con.execute(
                    f"SELECT rowid AS rowid, {select} FROM {_q(self.table)} WHERE rowid > ? ORDER BY rowid LIMIT ?",
                    (last, self.page_size)).fetchall()
                if not rows:
                    break
                side: Dict[int, Dict[str, Any]] = {}
                if self.side_table:
                    ids = [r["rowid"] for r in rows]
                    marks = ",".join("?" * len(ids))
                    for s in con.execute(
                        f"SELECT * FROM {_q(self.side_table)} WHERE {_q(self.side_key)} IN ({marks})", ids):
                        side[s[self.side_key]] = dict(s)
                for r in rows:
                    row = dict(r)
                    rowid = row["rowid"]
                    blob = row.pop(self.vector_column)
                    for k, v in side.get(rowid, {}).items():
                        row.setdefault(k, v)
                    try:
                        rec = cmap.record(row, rowid, rowid)
                        rec.vector = decode_vector(blob, self._kind)
                    except ValueError as exc:
                        rec = Record(id=str(rowid), error=f"rowid {rowid}: {exc}", cursor=rowid)
                    if self.namespace and not rec.namespace:
                        rec.namespace = self.namespace
                    yield rec
                last = rows[-1]["rowid"]
        finally:
            con.close()
