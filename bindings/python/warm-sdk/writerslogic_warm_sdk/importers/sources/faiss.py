# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""FAISS index files: vectors, native ids, and caller-supplied id/text/metadata sidecars."""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Iterator, Mapping, Optional

import numpy as np

from ..model import ImporterError, MissingDependency, Record
from ._sidecar import (Sidecars, add_sidecar_arguments, check_file, load_id_object, timestamps)
from .base import Source

_BATCH = 1024


class FaissSource(Source):
    name = "faiss"
    summary = "A FAISS index file (flat, HNSW, IDMap/IDMap2, IVF) with optional id map, texts and metadata sidecars"
    seekable = True

    def __init__(self, index: str, id_map: Optional[str] = None, texts: Optional[str] = None,
                 metadata: Optional[str] = None, namespace: str = "", created_at_key: Optional[str] = None,
                 updated_at_key: Optional[str] = None, mmap: bool = False):
        self.index = check_file(index, "--index")
        self.id_map = check_file(id_map, "--id-map")
        self.texts = check_file(texts, "--texts")
        self.metadata = check_file(metadata, "--metadata")
        self.namespace = namespace
        self.created_at_key, self.updated_at_key, self.mmap = created_at_key, updated_at_key, mmap
        self._idx: Any = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--index", required=True, help="FAISS index file written by faiss.write_index")
        parser.add_argument("--mmap", action="store_true", help="memory-map the index file (flat-style indexes)")
        add_sidecar_arguments(parser)

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(args.index, args.id_map, args.texts, args.metadata, args.namespace,
                   args.created_at_key, args.updated_at_key, args.mmap)

    def describe(self) -> Dict[str, Any]:
        stat = os.stat(self.index)
        return {"index": os.path.abspath(self.index), "size": stat.st_size, "id_map": self.id_map,
                "texts": self.texts, "metadata": self.metadata, "namespace": self.namespace}

    # -- index access ---------------------------------------------------------------

    def _load(self):
        if self._idx is not None:
            return self._idx
        try:
            import faiss
        except ImportError:
            raise MissingDependency("faiss", "faiss-cpu", "import-faiss") from None
        flags = faiss.IO_FLAG_MMAP if self.mmap else 0
        try:
            base = faiss.read_index(self.index, flags)
        except RuntimeError as exc:
            raise ImporterError(f"cannot read FAISS index {self.index}: {exc}") from None
        top = faiss.downcast_index(base)
        native = None
        inner = top
        if hasattr(top, "id_map"):  # IndexIDMap / IndexIDMap2 carry their own ids
            native = faiss.vector_to_array(top.id_map)
            inner = faiss.downcast_index(top.index)
        ivf = faiss.extract_index_ivf(inner) if hasattr(inner, "invlists") or hasattr(inner, "nlist") else None
        self._idx = (faiss, top, inner, native, ivf, base)  # base owns the C++ index; keep it alive
        return self._idx

    def count(self) -> Optional[int]:
        return int(self._load()[1].ntotal)

    def _positional_vectors(self, inner, start: int, n: int):
        try:
            return inner.reconstruct_n(start, n)
        except RuntimeError as exc:
            raise ImporterError(
                f"this FAISS index type cannot reconstruct stored vectors ({type(inner).__name__}: {exc}). "
                "Rebuild it as Flat, HNSWFlat, IDMap or IVF with stored vectors, or import the vectors from "
                "their original source."
            ) from None

    def records(self, cursor: Any = None) -> Iterator[Record]:
        faiss, top, inner, native, ivf, _base = self._load()
        start = int(cursor) + 1 if cursor is not None else 0
        total = int(top.ntotal)
        id_obj = load_id_object(self.id_map)
        if ivf is not None and native is not None:
            raise ImporterError("an IndexIDMap wrapped around an IVF index is not supported; export the IVF index directly")
        if ivf is not None and (self.texts or self.metadata or (self.id_map and id_obj is None)):
            raise ImporterError(
                "an IVF index is enumerated list by list, so row-aligned sidecars (--id-map list, --texts, "
                "--metadata) have no defined order. Use --id-map with a .json object {native id: external id}, "
                "or build the sidecars against a Flat/HNSW/IDMap index."
            )
        side = Sidecars(self.id_map, self.texts, self.metadata, start) if ivf is None else None
        if ivf is not None:
            yield from self._ivf_records(faiss, ivf, start, id_obj)
            return
        for i in range(start, total, _BATCH):
            n = min(_BATCH, total - i)
            vecs = self._positional_vectors(inner, i, n)
            for j in range(n):
                row = i + j
                extra = side.next(row)
                meta = extra.get("metadata", {})
                created, updated = timestamps(meta, self.created_at_key, self.updated_at_key)
                if "id" in extra:
                    ident = extra["id"]
                elif native is not None:
                    ident = str(int(native[row]))
                    if id_obj is not None:
                        ident = str(id_obj.get(ident, ident))
                else:
                    ident = str(row)
                    if id_obj is not None:
                        ident = str(id_obj.get(ident, ident))
                yield Record(id=ident, text=extra.get("text"), vector=vecs[j].astype("float64").tolist(),
                             metadata=meta, namespace=self.namespace, created_at=created, updated_at=updated,
                             cursor=row)
        side.finish(total)

    def _ivf_records(self, faiss, ivf, start: int, id_obj) -> Iterator[Record]:
        il = ivf.invlists
        ordinal = 0
        for lst in range(int(ivf.nlist)):
            size = int(il.list_size(lst))
            if ordinal + size <= start:
                ordinal += size
                continue
            ids = faiss.rev_swig_ptr(il.get_ids(lst), size).copy() if size else []
            for off in range(size):
                if ordinal >= start:
                    vec = np.empty(ivf.d, dtype="float32")
                    try:
                        ivf.reconstruct_from_offset(lst, off, faiss.swig_ptr(vec))
                    except RuntimeError as exc:
                        raise ImporterError(f"cannot reconstruct a vector from the IVF index: {exc}") from None
                    ident = str(int(ids[off]))
                    if id_obj is not None:
                        ident = str(id_obj.get(ident, ident))
                    yield Record(id=ident, vector=vec.astype("float64").tolist(), namespace=self.namespace,
                                 cursor=ordinal)
                ordinal += 1

    def close(self) -> None:
        self._idx = None
