# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Old-versus-new fidelity check on a sample of queries.

Ground truth is exact cosine top-k over the source's own vectors, computed by streaming
the source once (bounded memory), so it does not depend on the old system's ANN index.
The new side is the live HMS server after an import, or, in a dry run, exact cosine over
the vectors that would be sent (which measures the loss from projection).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .common import scrub, vector_list
from .mapping import Target, valid_vector
from .model import HmsDoc, ImporterError, Record
from .options import ImportOptions
from .sink import BatchRejected, HmsSink

SourceKey = Tuple[str, str]  # (namespace, source id)
_BLOCK = 2048


@dataclass
class Sample:
    key: SourceKey
    vector: Optional[List[float]]  # source vector (None when the source has none)
    docs: List[HmsDoc]  # what was sent for this record
    text: Optional[str]


class Reservoir:
    """Uniform sample (algorithm R) of imported records, bounded by `size`."""

    def __init__(self, size: int, seed: int):
        self.size = size
        self.items: List[Sample] = []
        self._seen = 0
        self._rng = random.Random(seed)

    def offer(self, rec: Record, docs: List[HmsDoc]) -> None:
        if self.size <= 0 or not docs:
            return
        self._seen += 1
        vec = vector_list(rec.vector)
        item = Sample(docs[0].source_key, vec if valid_vector(vec) else None, docs, rec.text if isinstance(rec.text, str) else None)
        if len(self.items) < self.size:
            self.items.append(item)
        else:
            j = self._rng.randrange(self._seen)
            if j < self.size:
                self.items[j] = item


def _ground_truth(
    records: Iterable[Record],
    samples: List[Sample],
    k: int,
    new_space: Optional[Callable[[List[float]], Optional[List[float]]]],
    default_ns: str,
    limit: Optional[int],
) -> Tuple[List[List[SourceKey]], List[List[SourceKey]]]:
    """Stream the source once; return per-sample top-k keys in source space and in new space."""
    import numpy as np

    def norm(m):
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return m / n

    queries = [i for i, s in enumerate(samples) if s.vector is not None]
    old_top: List[List[Tuple[float, SourceKey]]] = [[] for _ in samples]
    new_top: List[List[Tuple[float, SourceKey]]] = [[] for _ in samples]
    block: List[Tuple[SourceKey, List[float]]] = []
    q_new: Dict[int, Optional[List[float]]] = {}
    if new_space is not None:
        q_new = {qi: new_space(samples[qi].vector) for qi in queries}

    def merge(top, scores, keys, qi):
        cand = top[qi] + [(float(sc), key) for sc, key in zip(scores, keys)]
        cand.sort(key=lambda t: (-t[0], t[1]))
        top[qi] = cand[:k]

    def flush():
        if not block:
            return
        by_dim: Dict[int, List[Tuple[SourceKey, List[float]]]] = {}
        for key, vec in block:
            by_dim.setdefault(len(vec), []).append((key, vec))
        for dim, rows in by_dim.items():
            keys = [r[0] for r in rows]
            mat = norm(np.asarray([r[1] for r in rows], dtype=np.float64))
            for qi in queries:
                q = samples[qi]
                if len(q.vector) != dim:
                    continue
                allowed = [j for j, key in enumerate(keys) if key[0] == q.key[0]]
                if not allowed:
                    continue
                qv = norm(np.asarray([q.vector], dtype=np.float64))[0]
                scores = mat[allowed] @ qv
                merge(old_top, scores, [keys[j] for j in allowed], qi)
        if new_space is not None:
            moved: List[Tuple[SourceKey, List[float]]] = []
            for key, vec in block:
                nv = new_space(vec)
                if nv is not None:
                    moved.append((key, nv))
            by_dim = {}
            for key, vec in moved:
                by_dim.setdefault(len(vec), []).append((key, vec))
            for dim, rows in by_dim.items():
                keys = [r[0] for r in rows]
                mat = norm(np.asarray([r[1] for r in rows], dtype=np.float64))
                for qi in queries:
                    q = samples[qi]
                    qn = q_new.get(qi)
                    if qn is None or len(qn) != dim:
                        continue
                    allowed = [j for j, key in enumerate(keys) if key[0] == q.key[0]]
                    if not allowed:
                        continue
                    qv = norm(np.asarray([qn], dtype=np.float64))[0]
                    merge(new_top, mat[allowed] @ qv, [keys[j] for j in allowed], qi)
        block.clear()

    n = 0
    for rec in records:
        if limit is not None and n >= limit:
            break
        n += 1
        vec = vector_list(rec.vector)
        if not valid_vector(vec):
            continue
        ns = str(rec.namespace) if rec.namespace not in (None, "") else default_ns
        block.append(((ns, str(rec.id)), vec))
        if len(block) >= _BLOCK:
            flush()
    flush()
    return [[key for _, key in t] for t in old_top], [[key for _, key in t] for t in new_top]


def _collapse(matches: List[Dict[str, Any]], k: int) -> List[SourceKey]:
    seen: List[SourceKey] = []
    for m in matches:
        meta = m.get("metadata") or {}
        key = (str(meta.get("hms_namespace", "")), str(meta.get("hms_source_id", m.get("id"))))
        if key not in seen:
            seen.append(key)
        if len(seen) >= k:
            break
    return seen


def run_verification(
    *,
    source: Any,
    sink: Optional[HmsSink],
    opts: ImportOptions,
    target: Target,
    reservoir: Reservoir,
    new_space: Optional[Callable[[List[float]], Optional[List[float]]]],
    source_name: str,
    default_tenant: Optional[str],
) -> Dict[str, Any]:
    samples = reservoir.items
    k = opts.verify_k
    out: Dict[str, Any] = {"performed": False, "k": k}
    if opts.verify_sample == 0:
        out["reason"] = "disabled (--verify-sample 0)"
        return out
    if not samples:
        out["reason"] = "no imported records to sample"
        return out
    dry = sink is None
    with_vec = [s for s in samples if s.vector is not None]

    old, new_local = ([[] for _ in samples], [[] for _ in samples])
    if with_vec:
        old, new_local = _ground_truth(source.records(None), samples, k, new_space if dry else None, opts.default_namespace, opts.limit)

    recalls: List[float] = []
    self_hits = 0
    self_total = 0
    mismatches: List[Dict[str, Any]] = []
    mismatch_count = 0
    queried = 0
    for i, s in enumerate(samples):
        if dry:
            if s.vector is None or not new_local[i]:
                continue
            queried += 1
            got = new_local[i][:k]
        else:
            first = s.docs[0]
            qvec = first.vector
            if qvec is None and target.embedder is not None and s.text:
                qvec = vector_list(target.embedder([s.text])[0])
            query_kwargs: Dict[str, Any] = {"top_k": min(100, k * 3)}
            if qvec is not None:
                query_kwargs["vector"] = qvec
            elif s.text and target.server_embeds:
                query_kwargs["text"] = s.text[:8000]
            else:
                continue
            flt = {"hms_source": source_name, "hms_namespace": s.key[0]}
            try:
                matches = sink.query(filter=flt, tenant=first.tenant, **query_kwargs)
            except BatchRejected as exc:
                raise ImporterError(scrub(f"verification query refused: {exc}")) from None
            queried += 1
            got = _collapse(matches, k)
            by_id = {m.get("id"): m for m in matches}
            for doc in s.docs:
                hit = by_id.get(doc.id)
                if hit is None:
                    continue
                problems = []
                if (hit.get("text") or "").strip() != (doc.text or "").strip():  # the server trims surrounding whitespace
                    problems.append("text")
                if (hit.get("metadata") or {}) != doc.metadata:
                    problems.append("metadata")
                if problems:
                    mismatch_count += 1
                    if len(mismatches) < 10:
                        mismatches.append({"id": doc.id, "fields": problems})
        self_total += 1
        if s.key in got:
            self_hits += 1
        if s.vector is not None and old[i]:
            recalls.append(len(set(old[i]) & set(got)) / len(old[i]))

    out.update(
        {
            "performed": queried > 0,
            "queries": queried,
            "basis": (
                "dry run: exact cosine over the vectors that would be sent vs the source vectors"
                if dry
                else "live HMS query vs exact cosine over the source vectors"
            ),
            "queries_with_source_vectors": len(with_vec),
            "recall_at_k_mean": round(sum(recalls) / len(recalls), 4) if recalls else None,
            "recall_at_k_min": round(min(recalls), 4) if recalls else None,
            "self_hit_rate": round(self_hits / self_total, 4) if self_total else None,
            "roundtrip_mismatches": None if dry else mismatch_count,
            "roundtrip_mismatch_examples": mismatches,
        }
    )
    if not recalls and with_vec == []:
        out["note"] = "the source has no vectors, so recall against the old system is not defined; only self-retrieval was checked"
    if not queried:
        out["reason"] = "no sampled record could be queried in this mode"
    return out
