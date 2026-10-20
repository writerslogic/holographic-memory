# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Annoy index files: integer item ids and stored vectors, plus caller-supplied sidecars."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterator, Mapping, Optional

from ..model import ImporterError, MissingDependency, Record
from ._sidecar import Sidecars, add_sidecar_arguments, check_file, load_id_object, timestamps
from .base import Source

_METRICS = ("angular", "euclidean", "manhattan", "hamming", "dot")


class AnnoySource(Source):
    name = "annoy"
    summary = "An Annoy index file (items 0..n-1 and stored vectors) with optional id map, texts and metadata sidecars"
    seekable = True

    def __init__(self, index: str, dim: int, metric: str, id_map: Optional[str] = None, texts: Optional[str] = None,
                 metadata: Optional[str] = None, namespace: str = "", created_at_key: Optional[str] = None,
                 updated_at_key: Optional[str] = None):
        if metric not in _METRICS:
            raise ImporterError(f"--metric must be one of {', '.join(_METRICS)}")
        if dim < 1:
            raise ImporterError("--dim must be >= 1")
        self.index = check_file(index, "--index")
        self.dim, self.metric = dim, metric
        self.id_map = check_file(id_map, "--id-map")
        self.texts = check_file(texts, "--texts")
        self.metadata = check_file(metadata, "--metadata")
        self.namespace = namespace
        self.created_at_key, self.updated_at_key = created_at_key, updated_at_key
        self._index: Any = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--index", required=True, help="Annoy index file written by AnnoyIndex.save")
        parser.add_argument("--dim", type=int, required=True, help="vector dimension the index was built with")
        parser.add_argument("--metric", required=True, choices=list(_METRICS), help="metric the index was built with")
        add_sidecar_arguments(parser)

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(args.index, args.dim, args.metric, args.id_map, args.texts, args.metadata, args.namespace,
                   args.created_at_key, args.updated_at_key)

    def describe(self) -> Dict[str, Any]:
        return {"index": os.path.abspath(self.index), "size": os.stat(self.index).st_size, "dim": self.dim,
                "metric": self.metric, "id_map": self.id_map, "texts": self.texts, "metadata": self.metadata,
                "namespace": self.namespace}

    def _load(self):
        if self._index is not None:
            return self._index
        try:
            from annoy import AnnoyIndex
        except ImportError:
            raise MissingDependency("annoy", "annoy", "import-annoy") from None
        idx = AnnoyIndex(self.dim, self.metric)
        try:
            ok = idx.load(self.index)
        except OSError as exc:
            raise ImporterError(f"cannot load {self.index} as an Annoy index of dim {self.dim}, metric "
                                f"{self.metric} (a wrong --dim or --metric is the usual cause): {exc}") from None
        if ok is False:
            raise ImporterError(f"cannot load {self.index} as an Annoy index")
        self._index = idx
        return idx

    def count(self) -> Optional[int]:
        return int(self._load().get_n_items())

    def records(self, cursor: Any = None) -> Iterator[Record]:
        idx = self._load()
        n = idx.get_n_items()
        start = int(cursor) + 1 if cursor is not None else 0
        side = Sidecars(self.id_map, self.texts, self.metadata, start)
        id_obj = load_id_object(self.id_map)
        for item in range(start, n):
            extra = side.next(item)
            meta = extra.get("metadata", {})
            created, updated = timestamps(meta, self.created_at_key, self.updated_at_key)
            ident = extra.get("id") or str(item)
            if id_obj is not None:
                ident = str(id_obj.get(str(item), ident))
            # An item id that was never added reads back as an all-zero vector; the mapper counts it
            # as skipped (no usable vector, no text), it is not dropped here.
            vector = idx.get_item_vector(item)
            if not any(vector):
                vector = None
            yield Record(id=ident, text=extra.get("text"), vector=vector, metadata=meta,
                         namespace=self.namespace, created_at=created, updated_at=updated, cursor=item)
        side.finish(n)

    def close(self) -> None:
        if self._index is not None:
            self._index.unload()
            self._index = None
