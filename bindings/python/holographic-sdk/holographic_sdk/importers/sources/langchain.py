# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""LangChain stores: FAISS `save_local` folders, `InMemoryVectorStore.dump` JSON,
LocalFileStore docstore directories and JSONL of serialized Documents.

The FAISS `index.pkl` is a pickle. It is read with an allow-list unpickler that maps
the few classes LangChain writes to inert stand-ins, so a hostile pickle cannot run
code, and LangChain itself need not be installed."""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
from typing import Any, Dict, Iterator, Mapping

from ..model import ImporterError, MissingDependency, Record
from .base import Source

_DOC_CLASSES = {
    ("langchain_core.documents.base", "Document"),
    ("langchain_core.documents", "Document"),
    ("langchain.schema", "Document"),
    ("langchain.schema.document", "Document"),
    ("langchain_community.docstore.document", "Document"),
    ("langchain.docstore.document", "Document"),
}
_STORE_CLASSES = {
    ("langchain_community.docstore.in_memory", "InMemoryDocstore"),
    ("langchain.docstore.in_memory", "InMemoryDocstore"),
}
_SAFE_BUILTINS = {"set", "frozenset", "dict", "list", "tuple", "str", "int", "float", "bool", "bytes", "complex"}


class _Inert:
    """Captures pickled state without executing anything."""

    def __setstate__(self, state: Any) -> None:
        self.state = state


class _InertDocument(_Inert):
    pass


class _InertDocstore(_Inert):
    pass


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _DOC_CLASSES:
            return _InertDocument
        if (module, name) in _STORE_CLASSES:
            return _InertDocstore
        if module == "builtins" and name in _SAFE_BUILTINS:
            return getattr(__import__("builtins"), name)
        raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}: not a LangChain docstore class")


def _doc_fields(obj: Any) -> Dict[str, Any]:
    """page_content/metadata/id from an inert pydantic-v1/v2 Document or a dumpd dict."""
    if isinstance(obj, _InertDocument):
        st = obj.state
        d = st.get("__dict__", st) if isinstance(st, dict) else {}
        if not isinstance(d, dict):
            d = {}
        return {"page_content": d.get("page_content"), "metadata": d.get("metadata") or {}, "id": d.get("id")}
    if isinstance(obj, dict):
        kw = obj.get("kwargs", obj) if obj.get("lc") == 1 else obj
        return {"page_content": kw.get("page_content", kw.get("text")), "metadata": kw.get("metadata") or {},
                "id": kw.get("id")}
    raise ImporterError(f"unrecognized LangChain document object: {type(obj).__name__}")


class LangChainSource(Source):
    name = "langchain"
    summary = "LangChain FAISS folder, InMemoryVectorStore JSON, LocalFileStore docstore dir, or Document JSONL"

    def __init__(self, path: str, fmt: str, index_name: str, namespace: str):
        if fmt not in ("auto", "faiss", "inmemory", "jsonl", "filestore"):
            raise ImporterError("--format must be auto, faiss, inmemory, jsonl or filestore")
        if not os.path.exists(path):
            raise ImporterError(f"no such path: {path}")
        if fmt == "auto":
            if os.path.isdir(path):
                fmt = "faiss" if os.path.exists(os.path.join(path, f"{index_name}.faiss")) else "filestore"
            else:
                fmt = "jsonl" if path.endswith((".jsonl", ".ndjson")) else "inmemory"
        self.path, self.fmt, self.index_name, self.namespace = path, fmt, index_name, namespace

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--path", required=True, help="folder, or file, written by LangChain")
        parser.add_argument("--format", default="auto", choices=["auto", "faiss", "inmemory", "jsonl", "filestore"])
        parser.add_argument("--index-name", default="index", help="FAISS save_local index_name")
        parser.add_argument("--namespace", default="")

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(args.path, args.format, args.index_name, args.namespace)

    def describe(self) -> Dict[str, Any]:
        return {"path": os.path.abspath(self.path), "format": self.fmt, "index_name": self.index_name,
                "namespace": self.namespace}

    def records(self, cursor: Any = None) -> Iterator[Record]:
        return getattr(self, f"_{self.fmt}")()

    # -- FAISS ----------------------------------------------------------------------

    def _faiss(self) -> Iterator[Record]:
        try:
            import faiss
        except ImportError:
            raise MissingDependency("langchain", "faiss-cpu", "import-langchain") from None
        pkl = os.path.join(self.path, f"{self.index_name}.pkl")
        idx_path = os.path.join(self.path, f"{self.index_name}.faiss")
        try:
            with open(pkl, "rb") as fh:
                loaded = _RestrictedUnpickler(io.BytesIO(fh.read())).load()
        except pickle.UnpicklingError as exc:
            raise ImporterError(str(exc)) from None
        except OSError as exc:
            raise ImporterError(f"cannot read {pkl}: {type(exc).__name__}") from None
        if not (isinstance(loaded, tuple) and len(loaded) == 2):
            raise ImporterError(f"{pkl} is not a LangChain FAISS (docstore, index_to_docstore_id) pickle")
        docstore, id_map = loaded
        store = getattr(docstore, "state", {}).get("_dict")
        if not isinstance(store, dict) or not isinstance(id_map, dict):
            raise ImporterError(f"{pkl} has an unexpected docstore layout")
        index = faiss.read_index(idx_path)
        total = int(index.ntotal)
        for start in range(0, total, 1024):
            n = min(1024, total - start)
            try:
                vecs = index.reconstruct_n(start, n)
            except RuntimeError as exc:
                raise ImporterError(
                    "this FAISS index type cannot reconstruct stored vectors (IVF needs a direct map; "
                    f"quantized indexes are lossy): {exc}") from None
            for j in range(n):
                pos = start + j
                doc_id = id_map.get(pos)
                doc = store.get(doc_id)
                if doc is None:
                    yield Record(id=str(doc_id if doc_id is not None else pos), vector=vecs[j].tolist(),
                                 error=f"vector {pos} has no document in the docstore")
                    continue
                f = _doc_fields(doc)
                yield Record(id=str(doc_id), text=f["page_content"], vector=vecs[j].tolist(),
                             metadata=f["metadata"], namespace=self.namespace)

    # -- InMemoryVectorStore.dump ------------------------------------------------------

    def _inmemory(self) -> Iterator[Record]:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            raise ImporterError(f"cannot read {self.path}: {type(exc).__name__}") from None
        # current langchain-core dumps {id: item}; older releases wrapped that in {"store": ...}
        store = data.get("store", data) if isinstance(data, dict) else None
        if not isinstance(store, dict) or not all(isinstance(v, dict) and "vector" in v for v in store.values()):
            raise ImporterError(f"{self.path} is not an InMemoryVectorStore dump")
        for key in sorted(store):
            item = store[key]
            yield Record(id=str(item.get("id", key)), text=item.get("text"), vector=item.get("vector"),
                         metadata=item.get("metadata") or {}, namespace=self.namespace)

    # -- JSONL of Documents ---------------------------------------------------------------

    def _jsonl(self) -> Iterator[Record]:
        with open(self.path, "r", encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    f = _doc_fields(json.loads(line))
                except (ValueError, ImporterError) as exc:
                    yield Record(id=f"line-{n}", error=f"{self.path} line {n}: {exc}")
                    continue
                yield Record(id=str(f["id"] if f["id"] is not None else f"line-{n}"), text=f["page_content"],
                             metadata=f["metadata"], namespace=self.namespace)

    # -- LocalFileStore / create_kv_docstore ------------------------------------------------

    def _filestore(self) -> Iterator[Record]:
        root = os.path.realpath(self.path)
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames.sort()
            for fn in sorted(filenames):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                if os.path.islink(full):
                    yield Record(id=rel, error=f"{rel} is a symlink; not followed")
                    continue
                try:
                    with open(full, "rb") as fh:
                        f = _doc_fields(json.loads(fh.read().decode("utf-8")))
                except (OSError, ValueError, ImporterError) as exc:
                    yield Record(id=rel, error=f"{rel}: {type(exc).__name__}")
                    continue
                meta = dict(f["metadata"])
                yield Record(id=rel, text=f["page_content"], metadata=meta, namespace=self.namespace)
