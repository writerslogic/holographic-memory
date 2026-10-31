# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""pgvector on PostgreSQL or Supabase (a Supabase project is a Postgres DSN)."""

from __future__ import annotations

import argparse
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ..common import redact_uri, secret_from, to_jsonable, uri_secret
from ..model import ImporterError, MissingDependency, Record
from .base import ColumnMap, Source, add_map_argument

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_PAGE = 1000


def quote_ident(name: str) -> str:
    """Quote a possibly schema-qualified identifier; refuses anything that is not a plain name."""
    parts = name.split(".")
    if not 1 <= len(parts) <= 2 or not all(_IDENT.match(p) for p in parts):
        raise ImporterError(f"'{name}' is not a plain (schema-qualified) identifier")
    return ".".join('"' + p + '"' for p in parts)


def parse_pg_vector(value: Any) -> Any:
    """vector / halfvec text '[1,2,3]', sparsevec text '{1:0.5,3:1}/5' (densified), or a list."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    m = re.fullmatch(r"\{([^}]*)\}/(\d+)", text)
    if m:
        dense = [0.0] * int(m.group(2))
        for item in filter(None, m.group(1).split(",")):
            idx, _, val = item.partition(":")
            dense[int(idx) - 1] = float(val)  # sparsevec indexes are 1-based
        return dense
    return value  # '[..]' is handled by parse_vector in ColumnMap


class PgVectorSource(Source):
    name = "pgvector"
    summary = "pgvector tables or queries on PostgreSQL/Supabase (keyset-paged; vector, halfvec, sparsevec)"

    def __init__(self, dsn: Optional[str], table: Optional[str], query: Optional[str], key_column: str,
                 cmap: ColumnMap, page_size: int = _PAGE):
        if bool(table) == bool(query):
            raise ImporterError("give exactly one of --table or --query")
        self.dsn, self.table, self.query, self.key = dsn, table, query, key_column
        self.page = page_size
        self.cmap = cmap
        self.seekable = table is not None
        if table is not None:
            quote_ident(table)
            quote_ident(key_column)
            if "id" not in cmap.cols:
                cmap.cols["id"] = key_column

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--dsn", help="PostgreSQL connection string (Supabase: the project's 'Connection string'); "
                                          "default $DATABASE_URL, else libpq PG* variables ($PGPASSWORD is never logged)")
        parser.add_argument("--table", help="[schema.]table to read, paged by --key-column (resumable)")
        parser.add_argument("--query", help="SELECT statement instead of --table (streamed with a server-side cursor; "
                                            "must have a stable ORDER BY for a safe resume)")
        parser.add_argument("--key-column", default="id", help="unique sortable column for keyset paging (default id)")
        parser.add_argument("--page-size", type=int, default=_PAGE)
        add_map_argument(parser)

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        dsn = args.dsn or env.get("DATABASE_URL")
        uri_secret(dsn)
        secret_from(None, dict(env), "PGPASSWORD")
        return cls(dsn, args.table, args.query, args.key_column, ColumnMap(args.map), args.page_size)

    def describe(self) -> Dict[str, Any]:
        return {"dsn": redact_uri(self.dsn) if self.dsn else "libpq-environment", "table": self.table,
                "query": self.query, "key": self.key, "map": self.cmap.describe()}

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError:
            raise MissingDependency("pgvector", "psycopg", "import-pgvector") from None
        return psycopg.connect(self.dsn or "", row_factory=dict_row, autocommit=True)

    def count(self) -> Optional[int]:
        if self.table is None:
            return None
        with self._connect() as con:
            return con.execute(f"SELECT count(*) AS n FROM {quote_ident(self.table)}").fetchone()["n"]

    def _record(self, row: Dict[str, Any], ordinal: int, cursor: Any) -> Record:
        vcol = self.cmap.cols.get("vector")
        if vcol in row:
            row[vcol] = parse_pg_vector(row[vcol])
        rec = self.cmap.record(row, ordinal, cursor)
        cols = self.cmap.cols.get("metadata")
        if cols and cols != ["*"]:  # a jsonb metadata column is spread into keys, other columns copy by name
            spread: Dict[str, Any] = {}
            for c in cols:
                v = row.get(c)
                spread.update(v) if isinstance(v, dict) else spread.__setitem__(c, v)
            rec.metadata = spread
        return rec

    def records(self, cursor: Any = None) -> Iterator[Record]:
        with self._connect() as con:
            if self.table is None:
                with con.transaction(), con.cursor(name="warm_import") as cur:
                    cur.itersize = self.page
                    cur.execute(self.query)
                    for n, row in enumerate(cur, 1):
                        yield self._record(row, n, None)
                return
            tbl, key = quote_ident(self.table), quote_ident(self.key)
            last = cursor
            n = 0
            while True:
                if last is None:
                    rows = con.execute(f"SELECT * FROM {tbl} ORDER BY {key} LIMIT %s", (self.page,)).fetchall()
                else:
                    rows = con.execute(f"SELECT * FROM {tbl} WHERE {key} > %s ORDER BY {key} LIMIT %s",
                                       (last, self.page)).fetchall()
                if not rows:
                    return
                for row in rows:
                    n += 1
                    k = row[self.key]
                    jk = to_jsonable(k)
                    yield self._record(row, n, jk)
                    last = jk if isinstance(jk, (int, str)) else str(jk)
                if len(rows) < self.page:
                    return


__all__: List[str] = ["PgVectorSource"]
