# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""hnswlib index files: integer labels and stored vectors, plus caller-supplied sidecars."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterator, Mapping, Optional

from ..model import ImporterError, MissingDependency, Record
from ._sidecar import Sidecars, add_sidecar_arguments, check_file, load_id_object, timestamps
from .base import Source

_BATCH = 1024


class HnswlibSource(Source):
    name = "hnswlib"
    summary = "An hnswlib index file (labels and stored vectors) with optional id map, texts and metadata sidecars"
    seekable = True

    def __init__(self, index: str, dim: int, space: str, id_map: Optional[str] = None, texts: Optional[str] = None,
                 metadata: Optional[str] = None, namespace: str = "", created_at_key: Optional[str] = None,
                 updated_at_key: Optional[str] = None):
        if space not in ("l2", "ip", "cosine"):
            raise ImporterError("--space must be l2, ip or cosine")
        if dim < 1:
            raise ImporterError("--dim must be >= 1")
        self.index = check_file(index, "--index")
        self.dim, self.space = dim, space
        self.id_map = check_file(id_map, "--id-map")
        self.texts = check_file(texts, "--texts")
        self.metadata = check_file(metadata, "--metadata")
        self.namespace = namespace
        self.created_at_key, self.updated_at_key = created_at_key, updated_at_key
        self._loaded: Any = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--index", required=True, help="hnswlib index file written by Index.save_index")
        parser.add_argument("--dim", type=int, required=True, help="vector dimension the index was built with")
        parser.add_argument("--space", required=True, choices=["l2", "ip", "cosine"], help="distance space of the index")
        add_sidecar_arguments(parser)

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(args.index, args.dim, args.space, args.id_map, args.texts, args.metadata, args.namespace,
                   args.created_at_key, args.updated_at_key)

    def describe(self) -> Dict[str, Any]:
        return {"index": os.path.abspath(self.index), "size": os.stat(self.index).st_size, "dim": self.dim,
                "space": self.space, "id_map": self.id_map, "texts": self.texts, "metadata": self.metadata,
                "namespace": self.namespace}

    def _check_header(self) -> None:
        """hnswlib loads an index built with another dimension without complaint and then reads garbage.
        The file header records the per-element byte sizes, from which the dimension follows."""
        import struct

        with open(self.index, "rb") as fh:
            head = fh.read(48)
        if len(head) < 48:
            raise ImporterError(f"{self.index} is too short to be an hnswlib index")
        _off0, _max, _cur, size_per_element, _label_off, offset_data = struct.unpack("<6Q", head)
        data_bytes = size_per_element - offset_data - 8  # label is 8 bytes
        if data_bytes <= 0 or data_bytes % 4:
            raise ImporterError(f"cannot load {self.index} as an hnswlib index: unrecognised header")
        built_dim = data_bytes // 4
        if built_dim != self.dim:
            raise ImporterError(f"cannot load {self.index} as an hnswlib index of dim {self.dim}: it was built with "
                                f"dim {built_dim}; use --dim {built_dim}")

    def _load(self):
        if self._loaded is not None:
            return self._loaded
        try:
            import hnswlib
        except ImportError:
            raise MissingDependency("hnswlib", "hnswlib", "import-hnswlib") from None
        self._check_header()
        p = hnswlib.Index(space=self.space, dim=self.dim)
        try:
            p.load_index(self.index)
        except RuntimeError as exc:
            raise ImporterError(f"cannot load {self.index} as an hnswlib index of dim {self.dim}, space "
                                f"{self.space}: {exc}") from None
        # Labels sorted ascending give a stable order. The list is N Python ints, the same order of
        # magnitude as the index's own label table.
        labels = sorted(int(x) for x in p.get_ids_list())
        self._loaded = (p, labels)
        return self._loaded

    def count(self) -> Optional[int]:
        return len(self._load()[1])

    def records(self, cursor: Any = None) -> Iterator[Record]:
        p, labels = self._load()
        start = int(cursor) + 1 if cursor is not None else 0
        side = Sidecars(self.id_map, self.texts, self.metadata, start)
        id_obj = load_id_object(self.id_map)
        for i in range(start, len(labels), _BATCH):
            chunk = labels[i:i + _BATCH]
            try:
                found = dict(zip(chunk, p.get_items(chunk)))
            except RuntimeError:  # the chunk holds a label that was mark_deleted
                found = {}
                for label in chunk:
                    try:
                        found[label] = p.get_items([label])[0]
                    except RuntimeError:
                        pass
            for j, label in enumerate(chunk):
                row = i + j
                extra = side.next(row)
                meta = extra.get("metadata", {})
                created, updated = timestamps(meta, self.created_at_key, self.updated_at_key)
                ident = extra.get("id") or str(label)
                if id_obj is not None:
                    ident = str(id_obj.get(str(label), ident))
                # A deleted label (mark_deleted) is still listed by the index but has no readable vector; it is
                # yielded vector-less so the pipeline counts it as skipped instead of dropping it unseen.
                vec = found.get(label)
                yield Record(id=ident, text=extra.get("text") if vec is not None else None,
                             vector=None if vec is None else [float(x) for x in vec], metadata=meta,
                             namespace=self.namespace, created_at=created, updated_at=updated, cursor=row)
        side.finish(len(labels))
