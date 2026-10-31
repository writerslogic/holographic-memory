# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Streaming per-row sidecar files for index formats that store only vectors.

FAISS, hnswlib and Annoy indexes hold vectors (and at most integer ids). Texts, string
ids and metadata live in the caller's own files; these helpers read them lazily, in
lockstep with the index rows, so memory stays bounded.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterator, Optional

from ..model import ImporterError


def add_sidecar_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--id-map", help="per-row external ids: .npy, .txt (one per line), .jsonl (one JSON value per line) "
                                         "or .json (a list, or an object mapping index id to external id)")
    parser.add_argument("--texts", help="per-row texts: .jsonl (a JSON string, or an object with a 'text' key, per line) "
                                        "or .txt (one per line; use .jsonl for texts containing newlines)")
    parser.add_argument("--metadata", help=".jsonl with one metadata object per row")
    parser.add_argument("--namespace", default="", help="namespace for every row")
    parser.add_argument("--created-at-key", help="metadata key holding the creation timestamp (also kept in metadata)")
    parser.add_argument("--updated-at-key", help="metadata key holding the update timestamp (also kept in metadata)")


def check_file(path: Optional[str], what: str) -> Optional[str]:
    if path is not None and not os.path.isfile(path):
        raise ImporterError(f"{what}: not a file: {path}")
    return path


def line_values(path: str, skip: int, kind: str) -> Iterator[Any]:
    """Yield one value per row of a sidecar file, starting at row `skip`."""
    if path.endswith(".npy"):
        import numpy as np

        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        for x in arr[skip:]:
            yield x.item() if hasattr(x, "item") else x
        return
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ImporterError(f"{path}: a .json {kind} file must be a list here")
        yield from data[skip:]
        return
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if n < skip:
                continue
            line = line.rstrip("\n")
            if path.endswith(".jsonl"):
                try:
                    yield json.loads(line) if line.strip() else None
                except ValueError as exc:
                    raise ImporterError(f"{path} line {n + 1}: {exc}") from None
            else:
                yield line


def load_id_object(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """A .json id map that is an object {index id -> external id}, else None."""
    if path is None or not path.endswith(".json"):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {str(k): v for k, v in data.items()} if isinstance(data, dict) else None


class Sidecars:
    """Lockstep readers for ids, texts and metadata, positioned at row `start`."""

    def __init__(self, id_map: Optional[str], texts: Optional[str], metadata: Optional[str], start: int):
        self.ids = line_values(id_map, start, "id map") if id_map and load_id_object(id_map) is None else None
        self.texts = line_values(texts, start, "texts") if texts else None
        self.meta = line_values(metadata, start, "metadata") if metadata else None

    def next(self, row: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            if self.ids is not None:
                out["id"] = str(next(self.ids))
            if self.texts is not None:
                t = next(self.texts)
                if isinstance(t, dict):
                    t = t.get("text")
                out["text"] = None if t is None else str(t)
            if self.meta is not None:
                m = next(self.meta)
                out["metadata"] = m if isinstance(m, dict) else ({} if m is None else {"value": m})
        except StopIteration:
            raise ImporterError(f"a sidecar file (--id-map/--texts/--metadata) has fewer rows than the index (row {row})") from None
        return out

    def finish(self, total: int) -> None:
        for name, it in (("--id-map", self.ids), ("--texts", self.texts), ("--metadata", self.meta)):
            if it is not None and next(it, StopIteration) is not StopIteration:
                raise ImporterError(f"{name} has more rows than the {total} rows in the index")


def timestamps(meta: Dict[str, Any], created_key: Optional[str], updated_key: Optional[str]):
    return (meta.get(created_key) if created_key else None, meta.get(updated_key) if updated_key else None)
