# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Record to WARM document mapping: ids, namespaces, metadata, text splitting, vectors."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .common import Coercion, normalize_ts, stable_hash, to_jsonable, vector_list
from .model import WarmDoc, ImporterError, Record, Reject
from .options import ImportOptions
from .report import Stats

MAX_ID_BYTES = 256
_TENANT_RE = re.compile(r"[^A-Za-z0-9_-]")
RESERVED_PREFIX = "hms_"
MAX_RECORD_PARTS = 1000


@dataclass
class Target:
    """What the destination can accept, learned from the server or given by the caller."""

    input_dim: Optional[int] = None
    server_embeds: Optional[bool] = None  # None: unknown (dry run without a server)
    embedder: Optional[Callable[[List[str]], Sequence[Sequence[float]]]] = None

    @property
    def can_encode(self) -> Optional[bool]:
        if self.embedder is not None:
            return True
        return self.server_embeds


def load_embedder(spec: str) -> Callable[[List[str]], Sequence[Sequence[float]]]:
    """Import `package.module:callable`; the callable maps a list of texts to vectors."""
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ImporterError("--embedder must look like 'package.module:callable'")
    fn = getattr(importlib.import_module(module_name), attr, None)
    if not callable(fn):
        raise ImporterError(f"embedder '{spec}' is not callable")
    return fn


class Projector:
    """Deterministic Gaussian random projection, with zero padding when growing.

    The matrix is generated from `seed` alone, so a re-run (or a query-side client
    configured with the same seed) projects identically. Johnson-Lindenstrauss: cosine
    distortion shrinks as 1/sqrt(target_dim); measure it with the dry-run recall check.
    """

    def __init__(self, source_dim: int, target_dim: int, seed: int):
        import numpy as np

        self.source_dim = source_dim
        self.target_dim = target_dim
        self._matrix = None
        if source_dim > target_dim:
            rng = np.random.default_rng([seed, source_dim, target_dim])
            self._matrix = rng.standard_normal((source_dim, target_dim), dtype=np.float32) / math.sqrt(target_dim)

    def apply(self, vec: Sequence[float]) -> List[float]:
        import numpy as np

        x = np.asarray(vec, dtype=np.float32)
        if self._matrix is None:
            out = np.zeros(self.target_dim, dtype=np.float32)
            out[: x.shape[0]] = x
        else:
            out = x @ self._matrix
        return [float(v) for v in out]


def valid_vector(vec: Optional[List[float]]) -> bool:
    return bool(vec) and all(math.isfinite(v) for v in vec) and any(v != 0.0 for v in vec)


def split_text(text: str, max_words: int, max_bytes: int) -> List[str]:
    """Split at whitespace so that "".join(parts) == text (nothing is lost)."""
    if len(text.encode("utf-8")) <= max_bytes and len(text.split()) <= max_words:
        return [text]
    parts: List[str] = []
    cur: List[str] = []
    words = 0
    size = 0
    pos = 0
    for m in re.finditer(r"\s*\S+", text):
        tok = m.group(0)
        pos = m.end()
        tok_bytes = len(tok.encode("utf-8"))
        if tok_bytes > max_bytes:  # a single unbroken token larger than a part
            if cur:
                parts.append("".join(cur))
                cur, words, size = [], 0, 0
            raw = tok.encode("utf-8")
            start = 0
            while start < len(raw):
                end = min(start + max_bytes, len(raw))
                while end < len(raw) and (raw[end] & 0xC0) == 0x80:
                    end -= 1
                parts.append(raw[start:end].decode("utf-8"))
                start = end
            continue
        if cur and (words + 1 > max_words or size + tok_bytes > max_bytes):
            parts.append("".join(cur))
            cur, words, size = [], 0, 0
        cur.append(tok)
        words += 1
        size += tok_bytes
    tail = text[pos:]
    if cur:
        parts.append("".join(cur) + tail)
    elif tail:
        if parts:
            parts[-1] += tail
        else:
            parts.append(tail)
    return parts


def tenant_for(namespace: str, opts: ImportOptions, default_tenant: Optional[str]) -> Optional[str]:
    """WARM tenant id for a namespace (tenant mode); unsafe or long names get a hash suffix."""
    if opts.namespace_mode != "tenant" or not namespace:
        return default_tenant
    base = opts.tenant_prefix + namespace
    safe = _TENANT_RE.sub("_", base)
    if safe == base and 1 <= len(base) <= 64:
        return base
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:53]}-{digest}"


def valid_warm_id(ident: str) -> bool:
    return (
        0 < len(ident.encode("utf-8")) <= MAX_ID_BYTES
        and not any(ord(c) < 32 or ord(c) == 127 or 0x80 <= ord(c) < 0xA0 for c in ident)
        and not ident.startswith("hms:")
    )


def _composed_id(namespace: str, source_id: str, opts: ImportOptions) -> str:
    prefix = opts.id_prefix_namespace == "always" or (
        opts.id_prefix_namespace == "auto" and bool(namespace) and opts.namespace_mode == "metadata"
    )
    return f"{namespace}::{source_id}" if prefix else source_id


def warm_id(namespace: str, source_id: str, opts: ImportOptions) -> str:
    """Deterministic WARM document id for a source id (idempotent across re-runs)."""
    ident = _composed_id(namespace, source_id, opts)
    if valid_warm_id(ident):
        return ident
    return "h:" + hashlib.sha256(ident.encode("utf-8")).hexdigest()


def mapped_part_id(base_id: str, record_identity: str, mapping: str, parts: int, index: int) -> str:
    if mapping == "v2":
        return "h2:" + stable_hash(["warm-import-document-v2", record_identity, index])
    ident = f"{base_id}#part-{index}" if parts > 1 else base_id
    return ident if valid_warm_id(ident) else "h:" + hashlib.sha256(ident.encode("utf-8")).hexdigest()


class Mapper:
    """Turns `Record`s into `WarmDoc`s under the options; counts every conversion."""

    def __init__(
        self,
        source_name: str,
        opts: ImportOptions,
        target: Target,
        stats: Stats,
        default_tenant: Optional[str] = None,
        source_identity: Optional[str] = None,
    ):
        self.source_name = source_name
        self.opts = opts
        self.target = target
        self.stats = stats
        self.default_tenant = default_tenant
        self.source_identity = source_identity or stable_hash(source_name)
        self._projectors: Dict[int, Projector] = {}
        self._auto_space: Optional[str] = None

    # -- vectors --------------------------------------------------------------------

    def _decide_vector(self, rec: Record, has_text: bool) -> Optional[List[float]]:
        """The vector to send, or None to have the text encoded. Raises Reject/ImporterError."""
        mode, tgt = self.opts.vectors, self.target.input_dim
        vec = vector_list(rec.vector)
        usable = valid_vector(vec)
        if vec is not None:
            self.stats.source_dims[len(vec)] += 1
        if rec.vector is not None and not usable:
            self.stats.count("vectors_unusable")
        if mode in ("reencode", "none"):
            if vec is not None:
                self.stats.count("vectors_ignored")
            return self._needs_encode(rec, has_text)
        if mode == "project":
            if not usable:
                raise Reject("vector_missing", "project mode needs a source vector")
            if tgt is None:
                raise ImporterError("project mode needs the target dimension (server reachable or --target-dim)")
            proj = self._projectors.get(len(vec))
            if proj is None:
                proj = self._projectors[len(vec)] = Projector(len(vec), tgt, self.opts.projection_seed)
            self.stats.count("vectors_projected")
            return proj.apply(vec)
        if mode == "carry":
            if not usable:
                raise Reject("vector_missing" if vec is None else "vector_unusable", "carry mode needs a usable vector")
            if tgt is not None and len(vec) != tgt:
                raise Reject("vector_dim_mismatch", f"source {len(vec)} vs server {tgt}")
            self.stats.count("vectors_carried")
            return vec
        # auto: one embedding space per run, fixed by the first record. Mixing carried and
        # re-encoded vectors in one store would put two embedding models side by side.
        if self._auto_space is None:
            if self.stats.counts.get("vectors_carried"):
                self._auto_space = "carry"
            elif self.stats.counts.get("vectors_encoded"):
                self._auto_space = "encode"
            else:
                self._auto_space = "carry" if usable else "encode"
        if self._auto_space == "carry":
            if not usable:
                raise Reject(
                    "vector_missing_in_carry_run",
                    "this run carries source vectors; encoding this record would mix embedding spaces "
                    "(use --vectors reencode to encode the whole source)",
                )
            if tgt is not None and len(vec) != tgt:
                raise ImporterError(
                    f"source vectors have {len(vec)} dimensions but the WARM server takes {tgt}. "
                    "Nothing is projected silently. Choose --vectors reencode (encode text with the server model "
                    "or --embedder), --vectors project (seeded random projection; approximate), or start the "
                    f"server with --input-dim {len(vec)}."
                )
            self.stats.count("vectors_carried")
            return vec
        if vec is not None:
            self.stats.count("vectors_ignored")
        return self._needs_encode(rec, has_text)

    def _needs_encode(self, rec: Record, has_text: bool) -> None:
        if not has_text:
            raise Reject("no_vector_and_no_text", "nothing to carry or encode")
        if self.target.can_encode is False:
            raise ImporterError(
                "records without a usable vector need text encoding, but the WARM server has no on-device "
                "embedder (warm-server --features local-models --embed-model ...) and no --embedder was given"
            )
        self.stats.count("vectors_encoded")
        return None

    # -- documents ------------------------------------------------------------------

    def map(self, rec: Record) -> List[WarmDoc]:
        opts = self.opts
        sid = str(rec.id)
        if rec.error:
            raise Reject("source_unreadable", rec.error)
        namespace = rec.namespace if rec.namespace not in (None, "") else opts.default_namespace
        namespace = str(namespace)
        text = rec.text if isinstance(rec.text, str) else (None if rec.text is None else str(rec.text))
        if text is not None and not text.strip():
            text = None
        vec = self._decide_vector(rec, text is not None) if (text is not None or rec.vector is not None) else None
        if text is None and vec is None:
            raise Reject("empty", "no text and no vector")

        meta = self._metadata(rec, sid, namespace)
        base_id = warm_id(namespace, sid, opts)
        rewritten = base_id.startswith("h:") and not valid_warm_id(_composed_id(namespace, sid, opts))
        parts = split_text(text, opts.max_text_words, opts.max_text_bytes) if text is not None else [None]
        if len(parts) > MAX_RECORD_PARTS:
            raise Reject("too_many_parts", f"a source record must fit at most {MAX_RECORD_PARTS} document parts")
        record_identity = stable_hash(["warm-import-record-v2", self.source_identity, namespace, sid])
        meta.update({
            "warm_import_source": self.source_identity,
            "warm_import_record": record_identity,
            "warm_import_parts": len(parts),
            "warm_import_base_id": base_id,
            "warm_id_mapping": opts.id_mapping,
        })
        tenant = tenant_for(namespace, opts, self.default_tenant)
        docs: List[WarmDoc] = []
        for index, part in enumerate(parts):
            ident = mapped_part_id(base_id, record_identity, opts.id_mapping, len(parts), index)
            doc_meta = dict(meta)
            doc_meta["warm_import_identity"] = stable_hash(["warm-import-part-v2", record_identity, index])
            if len(parts) > 1:
                doc_meta.update({"warm_parent_id": base_id, "warm_part": index, "warm_parts": len(parts)})
            self._fit_metadata(doc_meta, sid)
            docs.append(WarmDoc(ident, part, list(vec) if vec is not None else None, doc_meta, tenant,
                               (namespace, sid), opts.adopt_legacy))
        if rewritten:
            self.stats.count("ids_rewritten")
        if len(parts) > 1:
            self.stats.count("texts_split")
            self.stats.count("text_parts_created", len(parts))
            if vec is not None:
                self.stats.count("vectors_replicated_across_parts", len(parts) - 1)
        self.stats.namespace(namespace, len(docs))
        return docs

    def _metadata(self, rec: Record, sid: str, namespace: str) -> Dict[str, Any]:
        note = Coercion()
        raw = rec.metadata
        if raw is None:
            raw = {}
        elif not isinstance(raw, dict):
            raw = {"value": raw}
        meta = to_jsonable(raw, note)
        collided = {k: v for k, v in meta.items() if k.startswith(RESERVED_PREFIX)}
        if collided:
            meta = {k: v for k, v in meta.items() if k not in collided}
            meta["warm_orig_meta"] = collided
            self.stats.count("metadata_reserved_key_collisions", len(collided))
        meta["warm_source"] = self.source_name
        meta["warm_source_id"] = sid
        meta["hms_namespace"] = namespace
        for label, value in (("created_at", rec.created_at), ("updated_at", rec.updated_at)):
            iso, raw_ts = normalize_ts(value)
            if iso is not None:
                meta[f"warm_{label}"] = iso
                self.stats.count("timestamps_normalized")
            elif raw_ts is not None:
                meta[f"warm_{label}_raw"] = raw_ts
                self.stats.count("timestamps_unparsed_kept_raw")
        for key, n in note.items():
            self.stats.coercions[key] += n
        return meta

    def _fit_metadata(self, meta: Dict[str, Any], sid: str) -> None:
        limit = self.opts.max_metadata_bytes
        size = len(json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        if size <= limit:
            return
        if self.opts.oversize_metadata == "reject":
            raise Reject("metadata_too_large", f"{size} bytes > {limit}")
        dropped: List[str] = []
        sizes = {
            k: len(json.dumps(v, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
            for k, v in meta.items()
            if not k.startswith(RESERVED_PREFIX)
        }
        for key in sorted(sizes, key=sizes.get, reverse=True):  # type: ignore[arg-type]
            dropped.append(key)
            del meta[key]
            meta["warm_dropped_keys"] = dropped
            size = len(json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
            if size <= limit:
                self.stats.count("metadata_keys_dropped", len(dropped))
                return
        raise Reject("metadata_too_large", f"reserved keys alone are {size} bytes")
