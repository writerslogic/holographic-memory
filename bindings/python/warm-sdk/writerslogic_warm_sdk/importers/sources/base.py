# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Source interface and helpers shared by the concrete sources."""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from ..common import parse_json_maybe, to_jsonable, vector_list
from ..model import ImporterError, Record


class Source:
    """A place records are read from.

    Subclasses yield `Record`s in a deterministic order. Reading must be streaming:
    never hold more than a page of records. A `seekable` source resumes from a cursor;
    any other source is resumed by the pipeline skipping the committed prefix.
    """

    name: str = ""
    summary: str = ""
    seekable: bool = False

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        """JSON description of what is read, with no credentials; feeds the checkpoint key."""
        raise NotImplementedError

    def count(self) -> Optional[int]:
        return None

    def records(self, cursor: Any = None) -> Iterator[Record]:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        return None

    def __enter__(self) -> "Source":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---- column mapping for tabular sources -------------------------------------------

_FIELDS = ("id", "text", "vector", "namespace", "created_at", "updated_at", "metadata")


def add_map_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="FIELD=COLUMN",
        help="column mapping, repeatable. FIELD is id, text, vector, namespace, created_at, updated_at "
        "or metadata. metadata takes comma-separated columns, or '*' for every unmapped column (default). "
        "Without --map id the row number is the id.",
    )


class ColumnMap:
    """Maps table rows (dict-like) to `Record`s."""

    def __init__(self, spec: Sequence[str] = (), default_text: Sequence[str] = ("text", "content", "document")):
        self.cols: Dict[str, Any] = {}
        for item in spec:
            field, sep, col = item.partition("=")
            field = field.strip()
            if not sep or field not in _FIELDS or not col.strip():
                raise ImporterError(f"bad --map '{item}': use FIELD=COLUMN with FIELD in {', '.join(_FIELDS)}")
            self.cols[field] = [c.strip() for c in col.split(",")] if field == "metadata" else col.strip()
        self.default_text = tuple(default_text)

    def describe(self) -> Dict[str, Any]:
        return dict(self.cols)

    def record(self, row: Mapping[str, Any], ordinal: int, cursor: Any = None) -> Record:
        cols = self.cols
        used = set()

        def pick(field: str) -> Any:
            col = cols.get(field)
            if col is None or col not in row:
                return None
            used.add(col)
            return row[col]

        if "text" in cols or any(c in row for c in self.default_text):
            tcol = cols.get("text") or next(c for c in self.default_text if c in row)
            cols_text = tcol
        else:
            cols_text = None
        ident = pick("id")
        text = None
        if cols_text is not None and cols_text in row:
            used.add(cols_text)
            text = row[cols_text]
        vec = pick("vector")
        meta_cols = cols.get("metadata", ["*"])
        used.update(c for c in (cols.get("namespace"), cols.get("created_at"), cols.get("updated_at")) if c)
        if meta_cols == ["*"]:
            meta = {k: v for k, v in row.items() if k not in used}
        else:
            missing = [c for c in meta_cols if c not in row]
            if missing:
                raise ImporterError(f"--map metadata names missing column(s): {', '.join(missing)}")
            meta = {c: row[c] for c in meta_cols}
            if len(meta_cols) == 1:  # a single JSON/dict column holds the metadata object itself
                inner = parse_json_maybe(meta[meta_cols[0]])
                if isinstance(inner, dict):
                    meta = inner
        ns = pick("namespace")
        return Record(
            id=str(ident) if ident is not None else str(ordinal),
            text=None if text is None else str(text),
            vector=parse_vector(vec),
            metadata=meta,
            namespace="" if ns is None else str(ns),
            created_at=row.get(cols["created_at"]) if "created_at" in cols else None,
            updated_at=row.get(cols["updated_at"]) if "updated_at" in cols else None,
            cursor=cursor,
        )


def parse_vector(value: Any) -> Optional[List[float]]:
    """Vector from a list, array, JSON text ('[0.1, 0.2]'), delimited text, or float32 bytes."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text[0] == "[":
            try:
                return vector_list(json.loads(text))
            except ValueError as exc:
                raise ImporterError(f"vector text is not valid JSON: {exc}") from None
        return vector_list([float(x) for x in re.split(r"[,\s;]+", text.strip("()[]")) if x])
    return vector_list(value)


def parse_cli_json(text: Optional[str], what: str) -> Any:
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ImporterError(f"{what} is not valid JSON: {exc}") from None


def clean_meta(meta: Any) -> Dict[str, Any]:
    meta = parse_json_maybe(meta)
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return meta
    return {"value": to_jsonable(meta)}
