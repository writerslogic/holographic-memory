# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Generic file and database sources: JSONL, CSV, Parquet, SQLite, DuckDB, NumPy."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sqlite3
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ..common import redact_uri
from ..model import ImporterError, MissingDependency, Record
from .base import ColumnMap, Source, add_map_argument


def _expand(paths: List[str]) -> List[str]:
    out: List[str] = []
    for p in paths:
        hits = sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
        out.extend(hits)
    if not out:
        raise ImporterError(f"no files match {paths}")
    for p in out:
        if not os.path.isfile(p):
            raise ImporterError(f"not a file: {p}")
    return out


class _FileSource(Source):
    def __init__(self, paths: List[str], cmap: ColumnMap):
        self.paths = _expand(paths)
        self.cmap = cmap

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--path", action="append", required=True, help="file or glob; repeatable")
        add_map_argument(parser)

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(args.path, ColumnMap(args.map))

    def describe(self) -> Dict[str, Any]:
        return {"paths": [os.path.abspath(p) for p in self.paths], "map": self.cmap.describe()}


class JsonlSource(_FileSource):
    name = "jsonl"
    summary = "JSON Lines files, one object per line, with --map column mapping (seekable by byte offset)"
    seekable = True

    def records(self, cursor: Any = None) -> Iterator[Record]:
        file_idx, offset, ordinal = cursor or [0, 0, 0]
        for fi, path in enumerate(self.paths):
            if fi < file_idx:
                continue
            with open(path, "rb") as fh:
                if fi == file_idx and offset:
                    fh.seek(offset)
                else:
                    offset = 0
                while True:
                    start = fh.tell()
                    raw = fh.readline()
                    if not raw:
                        break
                    if not raw.strip():
                        continue
                    ordinal += 1
                    cur = [fi, fh.tell(), ordinal]
                    try:
                        row = json.loads(raw)
                        if not isinstance(row, dict):
                            raise ValueError("not a JSON object")
                    except (ValueError, UnicodeDecodeError) as exc:
                        yield Record(id=f"{os.path.basename(path)}@{start}", error=f"{path} byte {start}: {exc}", cursor=cur)
                        continue
                    yield self.cmap.record(row, ordinal if len(self.paths) == 1 else f"{fi}-{ordinal}", cur)


class CsvSource(_FileSource):
    name = "csv"
    summary = "CSV/TSV files with a header row, with --map column mapping (vector columns may hold JSON arrays)"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        super().add_arguments(parser)
        parser.add_argument("--delimiter", default=",", help="field delimiter ('\\t' for TSV)")

    def __init__(self, paths: List[str], cmap: ColumnMap, delimiter: str = ","):
        super().__init__(paths, cmap)
        self.delimiter = "\t" if delimiter == "\\t" else delimiter

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(args.path, ColumnMap(args.map), args.delimiter)

    def records(self, cursor: Any = None) -> Iterator[Record]:
        csv.field_size_limit(2**31 - 1)
        ordinal = 0
        for fi, path in enumerate(self.paths):
            with open(path, "r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh, delimiter=self.delimiter):
                    ordinal += 1
                    row = {k: v for k, v in row.items() if k is not None}
                    yield self.cmap.record(row, ordinal if len(self.paths) == 1 else f"{fi}-{ordinal}")


class ParquetSource(_FileSource):
    name = "parquet"
    summary = "Parquet files (streamed by row group batches) with --map column mapping"

    def count(self) -> Optional[int]:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return None
        return sum(pq.ParquetFile(p).metadata.num_rows for p in self.paths)

    def records(self, cursor: Any = None) -> Iterator[Record]:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise MissingDependency("parquet", "pyarrow", "import-files") from None
        ordinal = 0
        for fi, path in enumerate(self.paths):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    ordinal += 1
                    yield self.cmap.record(row, ordinal if len(self.paths) == 1 else f"{fi}-{ordinal}")


class SqliteSource(Source):
    name = "sqlite"
    summary = "A SQLite table or query with --map column mapping (read-only; seekable by rowid for tables)"

    def __init__(self, db: str, table: Optional[str], query: Optional[str], cmap: ColumnMap):
        if bool(table) == bool(query):
            raise ImporterError("give exactly one of --table or --query")
        if not os.path.isfile(db):
            raise ImporterError(f"not a file: {db}")
        self.db, self.table, self.query, self.cmap = db, table, query, cmap
        self.seekable = table is not None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--db", required=True, help="SQLite database file")
        parser.add_argument("--table")
        parser.add_argument("--query", help="SELECT statement (alternative to --table); rows are ordered as the query orders them")
        add_map_argument(parser)

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(args.db, args.table, args.query, ColumnMap(args.map))

    def describe(self) -> Dict[str, Any]:
        return {"db": os.path.abspath(self.db), "table": self.table, "query": self.query, "map": self.cmap.describe()}

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"file:{os.path.abspath(self.db)}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        return con

    def count(self) -> Optional[int]:
        if self.table is None:
            return None
        with self._connect() as con:
            return con.execute(f'SELECT COUNT(*) FROM "{self.table.replace(chr(34), chr(34) * 2)}"').fetchone()[0]

    def records(self, cursor: Any = None) -> Iterator[Record]:
        con = self._connect()
        try:
            if self.table is not None:
                tbl = '"' + self.table.replace('"', '""') + '"'
                cur = con.execute(f"SELECT rowid AS _warm_rowid, * FROM {tbl} WHERE rowid > ? ORDER BY rowid", (cursor or 0,))
            else:
                cur = con.execute(self.query)
            ordinal = 0
            for row in cur:
                ordinal += 1
                d = dict(row)
                rowid = d.pop("_warm_rowid", None)
                yield self.cmap.record(d, rowid if rowid is not None else ordinal, rowid)
        finally:
            con.close()


class DuckDbSource(Source):
    name = "duckdb"
    summary = "A DuckDB database, or SQL over Parquet/CSV/JSON files, with --map column mapping"

    def __init__(self, db: Optional[str], query: str, cmap: ColumnMap):
        self.db, self.query, self.cmap = db, query, cmap

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--db", help="DuckDB database file (omit to query files only)")
        parser.add_argument("--query", required=True,
                            help="SELECT statement, e.g. \"SELECT * FROM read_parquet('docs/*.parquet')\"; "
                                 "include ORDER BY for a stable resume order")
        add_map_argument(parser)

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(args.db, args.query, ColumnMap(args.map))

    def describe(self) -> Dict[str, Any]:
        return {"db": os.path.abspath(self.db) if self.db else None, "query": self.query, "map": self.cmap.describe()}

    def records(self, cursor: Any = None) -> Iterator[Record]:
        try:
            import duckdb
        except ImportError:
            raise MissingDependency("duckdb", "duckdb", "import-files") from None
        con = duckdb.connect(self.db, read_only=bool(self.db)) if self.db else duckdb.connect()
        try:
            res = con.execute(self.query)
            names = [d[0] for d in res.description]
            ordinal = 0
            while True:
                rows = res.fetchmany(1024)
                if not rows:
                    break
                for row in rows:
                    ordinal += 1
                    yield self.cmap.record(dict(zip(names, row)), ordinal)
        finally:
            con.close()


class NumpySource(Source):
    name = "numpy"
    summary = "NumPy .npy (memory-mapped) or .npz vectors with optional ids/texts/metadata sidecars (seekable by row)"
    seekable = True

    def __init__(self, path: str, vectors_key: Optional[str], ids: Optional[str], texts: Optional[str],
                 metadata: Optional[str], namespace: str):
        if not os.path.isfile(path):
            raise ImporterError(f"not a file: {path}")
        self.path, self.vectors_key, self.ids, self.texts, self.metadata, self.namespace = (
            path, vectors_key, ids, texts, metadata, namespace)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--path", required=True, help=".npy (N x D) or .npz archive")
        parser.add_argument("--vectors-key", help=".npz array holding the N x D matrix (default: vectors, embeddings, vecs or the only array)")
        parser.add_argument("--ids", help=".npz key, or a .npy/.txt/.jsonl file with one id per row (default: row number)")
        parser.add_argument("--texts", help=".npz key, or a .jsonl/.txt file with one text per row")
        parser.add_argument("--metadata", help=".jsonl file with one metadata object per row")
        parser.add_argument("--namespace", default="", help="namespace for every row")

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(args.path, args.vectors_key, args.ids, args.texts, args.metadata, args.namespace)

    def describe(self) -> Dict[str, Any]:
        return {"path": os.path.abspath(self.path), "vectors_key": self.vectors_key, "ids": self.ids,
                "texts": self.texts, "metadata": self.metadata, "namespace": self.namespace}

    def _load(self):
        import numpy as np

        if self.path.endswith(".npz"):
            z = np.load(self.path, allow_pickle=False)
            keys = list(z.keys())
            vk = self.vectors_key or next((k for k in ("vectors", "embeddings", "vecs") if k in keys), None)
            if vk is None and len(keys) == 1:
                vk = keys[0]
            if vk is None or vk not in keys:
                raise ImporterError(f"cannot tell which .npz array holds the vectors; keys: {keys}; use --vectors-key")
            return z, np.asarray(z[vk]), keys
        return None, np.load(self.path, mmap_mode="r", allow_pickle=False), []

    def _side(self, spec: Optional[str], z, nrows: int, what: str, skip: int) -> Optional[Iterator[Any]]:
        """Lazy per-row iterator over an .npz key, a .npy array (memory-mapped) or a line file."""
        if spec is None:
            return None
        if z is not None and spec in z.keys():
            arr = z[spec]
            return (x.item() if hasattr(x, "item") else x for x in arr[skip:])
        if not os.path.isfile(spec):
            raise ImporterError(f"{what} '{spec}' is neither an .npz key nor a file")
        if spec.endswith(".npy"):
            import numpy as np

            arr = np.load(spec, mmap_mode="r", allow_pickle=False)
            return (x.item() if hasattr(x, "item") else x for x in arr[skip:])

        def lines() -> Iterator[Any]:
            with open(spec, "r", encoding="utf-8") as fh:
                for n, line in enumerate(fh):
                    if n < skip:
                        continue
                    line = line.rstrip("\n")
                    yield (json.loads(line) if line.strip() else None) if spec.endswith(".jsonl") else line

        return lines()

    def count(self) -> Optional[int]:
        _, mat, _ = self._load()
        return int(mat.shape[0])

    def records(self, cursor: Any = None) -> Iterator[Record]:
        z, mat, _ = self._load()
        if mat.ndim != 2:
            raise ImporterError(f"vectors must be 2-D (N x D); got shape {tuple(mat.shape)}")
        n = int(mat.shape[0])
        start = int(cursor) + 1 if cursor is not None else 0
        ids = self._side(self.ids, z, n, "--ids", start)
        texts = self._side(self.texts, z, n, "--texts", start)
        metas = self._side(self.metadata, z, n, "--metadata", start)
        for i in range(start, n):
            try:
                ident = next(ids) if ids is not None else i
                text = next(texts) if texts is not None else None
                meta = next(metas) if metas is not None else {}
            except StopIteration:
                raise ImporterError(f"a sidecar (--ids/--texts/--metadata) has fewer rows than the {n} vectors") from None
            yield Record(
                id=str(ident),
                text=None if text is None else str(text),
                vector=mat[i].astype("float64").tolist(),
                metadata=meta if isinstance(meta, dict) else {},
                namespace=self.namespace,
                cursor=i,
            )
        for name, it in (("ids", ids), ("texts", texts), ("metadata", metas)):
            if it is not None and next(it, StopIteration) is not StopIteration:
                raise ImporterError(f"--{name} has more rows than the {n} vectors")


__all__ = [
    "JsonlSource", "CsvSource", "ParquetSource", "SqliteSource", "DuckDbSource", "NumpySource", "redact_uri",
]
