# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""LlamaIndex persisted storage (`StorageContext.persist`): docstore.json plus a SimpleVectorStore JSON.

The files are plain JSON, so no LlamaIndex import and no pickle is involved. LlamaIndex
itself loads these files whole, and so does this source: memory is proportional to the
persisted index, which is the format's own limit."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterator, List, Mapping, Optional

from ..model import ImporterError, Record
from .base import Source

_NODE_TYPES = {"1": "TextNode", "2": "ImageNode", "3": "IndexNode", "4": "Document"}


def _load(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ImporterError(f"cannot read {path}: {type(exc).__name__}") from None
    if not isinstance(data, dict):
        raise ImporterError(f"{path} is not a LlamaIndex JSON store")
    return data


class LlamaIndexSource(Source):
    name = "llamaindex"
    summary = "LlamaIndex persisted storage directory (docstore.json + SimpleVectorStore JSON)"

    def __init__(self, persist_dir: str, vector_store: str, docstore: str, namespace: str, documents: str):
        if not os.path.isdir(persist_dir):
            raise ImporterError(f"not a directory: {persist_dir}")
        if documents not in ("auto", "always", "never"):
            raise ImporterError("--documents must be auto, always or never")
        self.dir, self.vector_store, self.docstore, self.namespace, self.documents = (
            persist_dir, vector_store, docstore, namespace, documents)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--persist-dir", required=True, help="directory written by StorageContext.persist")
        parser.add_argument("--vector-store", default="default__vector_store.json",
                            help="vector store file inside the directory (named stores: '<name>__vector_store.json')")
        parser.add_argument("--docstore", default="docstore.json")
        parser.add_argument("--namespace", default="", help="namespace for every record")
        parser.add_argument("--documents", default="auto", choices=["auto", "always", "never"],
                            help="source Documents as records: auto (default) only when they have no chunks, "
                                 "always, or never")

    @classmethod
    def from_args(cls, args: argparse.Namespace, env: Mapping[str, str]) -> "Source":
        return cls(args.persist_dir, args.vector_store, args.docstore, args.namespace, args.documents)

    def describe(self) -> Dict[str, Any]:
        return {"dir": os.path.abspath(self.dir), "vector_store": self.vector_store, "docstore": self.docstore,
                "namespace": self.namespace, "documents": self.documents}

    def _stores(self):
        dpath, vpath = os.path.join(self.dir, self.docstore), os.path.join(self.dir, self.vector_store)
        data = _load(dpath).get("docstore/data", {}) if os.path.exists(dpath) else {}
        vec = _load(vpath) if os.path.exists(vpath) else {}
        if not data and not vec:
            raise ImporterError(f"neither {self.docstore} nor {self.vector_store} found in {self.dir}")
        return data, vec

    def count(self) -> Optional[int]:
        data, vec = self._stores()
        return len(set(data) | set(vec.get("embedding_dict", {})))

    def records(self, cursor: Any = None) -> Iterator[Record]:
        data, vec = self._stores()
        embeddings: Dict[str, List[float]] = vec.get("embedding_dict", {}) or {}
        ref_of: Dict[str, str] = vec.get("text_id_to_ref_doc_id", {}) or {}
        vmeta: Dict[str, Any] = vec.get("metadata_dict", {}) or {}
        chunked_docs = {n["__data__"]["relationships"]["1"]["node_id"]
                        for n in data.values()
                        if n.get("__type__") != "4" and "1" in (n.get("__data__", {}).get("relationships") or {})}
        for nid in sorted(set(data) | set(embeddings)):
            entry = data.get(nid)
            if entry is None:  # vector-only: external-style store; text may sit in `_node_content`
                meta = dict(vmeta.get(nid) or {})
                text = None
                content = meta.pop("_node_content", None)
                if isinstance(content, str):
                    try:
                        text = json.loads(content).get("text")
                    except ValueError:
                        pass
                yield Record(id=nid, text=text, vector=embeddings.get(nid), metadata=meta, namespace=self.namespace)
                continue
            node = entry.get("__data__", {})
            kind = _NODE_TYPES.get(str(entry.get("__type__")), str(entry.get("__type__")))
            if kind == "Document":
                if self.documents == "never" or (self.documents == "auto" and nid in chunked_docs):
                    continue
            meta = dict(node.get("metadata") or {})
            meta["llamaindex_node_type"] = kind
            rels = node.get("relationships") or {}
            names = {"1": "source", "2": "previous", "3": "next", "4": "parent", "5": "child"}
            for key, rel in rels.items():
                targets = rel if isinstance(rel, list) else [rel]
                ids = [t.get("node_id") for t in targets if isinstance(t, dict)]
                meta[f"llamaindex_{names.get(key, 'rel' + key)}"] = ids if isinstance(rel, list) else (ids[0] if ids else None)
            ref = ref_of.get(nid) or ((rels.get("1") or {}).get("node_id") if isinstance(rels.get("1"), dict) else None)
            if ref:
                meta["ref_doc_id"] = ref
            for k in ("start_char_idx", "end_char_idx", "mimetype"):
                if node.get(k) is not None:
                    meta[k] = node[k]
            for k in ("excluded_embed_metadata_keys", "excluded_llm_metadata_keys"):
                if node.get(k):
                    meta[k] = node[k]
            text = node.get("text")
            if kind == "ImageNode" and not text:
                text = node.get("image_path") or node.get("image_url")
            yield Record(
                id=nid, text=text, vector=embeddings.get(nid) or node.get("embedding"), metadata=meta,
                namespace=self.namespace, created_at=meta.get("creation_date"),
                updated_at=meta.get("last_modified_date"),
            )
