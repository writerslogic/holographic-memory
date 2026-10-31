#!/usr/bin/env python3
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Holographic query batching (HQB) kill test.

Question
--------
Can K concurrent queries be superposed into ONE d-vector so that a single
pass over the corpus yields candidates for all K, which are then reranked
exactly?  If so the scan costs d*N/K multiply-accumulates (MACs) per query
instead of d*N.  This harness measures whether the candidate pool keeps the
exact top-10 (recall@10 after rerank) at the FLOP budget that would make the
trick worth having, against the ordinary batching baseline that any
superposition must beat.

Variants (one corpus, one ground truth, one timing harness)
--------
single  exact scan, one query per pass: s = X q, top-10 directly.
v2      ordinary batching, the honest baseline: S = X [q_1..q_K], exact
        top-10 per column.  Same MACs per query as `single`, but the corpus
        is streamed once per K queries, so wall time per query drops.
v1      keyed / plain superposition.  Q = sum_i P_i q_i with P_i a fixed
        seeded signed permutation of coordinates (mode "key", the WARM
        permute/bind construction) or P_i = I (mode "sum").  One matvec
        s = X Q gives a mixed score per doc.  A shared pool of C docs is
        selected by |s| ("abs") or by s ("pos"); every query of the group
        reranks the whole pool exactly (C*d MACs per query).  A scalar
        cannot be unbound, so "key" is expected to carry no signal; "sum"
        carries the sum of the K per-query scores.
v3      sign-split superposition.  The coordinates are cut into K disjoint
        blocks of m = d/K; query i keeps only block i.  One pass computes
        the K per-block partial dot products (K matvecs of N x m, d*N MACs
        in total, corpus read once); each query takes its own top-C by its
        partial score and reranks exactly.  Mode "axis" uses a seeded
        permutation of the raw coordinates; mode "rot" first applies a
        seeded Haar-random rotation R (QR of a Gaussian matrix) to docs and
        queries so the embedding spectrum is spread evenly over the blocks.
        v3 is exactly a per-query random m-dimensional subspace sketch; the
        superposition only packs K sketches into one d-vector.

Data layout
-----------
X    (N, d) float32 C-order, rows l2-normalised (cosine == dot).
Qs   (nq, d) float32 l2-normalised held-out queries.
S    (nq, N) float32 exact scores: ground truth, and the recall arithmetic
     of every variant (the exact rerank of a candidate set is a lookup into
     S; the timing harness performs the real gather + matmul rerank and
     reports the recall it observed on the timed groups as a cross-check).
Xb   (K, N, m) float32 block-major corpus for v3 at batch width K, so the
     k-th block matvec reads a contiguous N x m matrix.
Groups: queries are shuffled once (seeded) and cut into consecutive groups
of K; the last group may be short and is handled as a group of K' < K
(v1 superposes K' queries, v3 uses the first K' blocks).  Timing uses only
full groups.

Invariants
----------
* Every random draw comes from a numpy Generator seeded from --seed; all
  outputs except wall times are reproducible for a fixed numpy version.
* BLAS / OpenMP thread caps are set to 1 before numpy is imported.
* Timing: --warmup passes, then --runs passes over the same groups; the
  median of per-run microseconds per query is reported with min and runs.
* FLOPs follow the d*N convention of the brief (MACs, not 2*MACs).
* recall@10 = |exact top-10 and returned top-10| / 10, averaged over queries.
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_var] = "1"

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA = "hqb_v1"
TOP = 10
KS_DEFAULT = (1, 2, 4, 8, 16)
CS_DEFAULT = (200, 1000, 5000)
V1_MODES = ("key", "sum")
V1_SELECTS = ("abs", "pos")
V3_MODES = ("axis", "rot")
PASS_RULE = (
    "a v1 or v3 row with K >= 4, C <= 1000, recall_at_10 >= 0.95 and "
    "scan_flop_reduction_vs_single >= 2 exists on every evaluated dataset"
)
ROW_CHUNK = 64


# --------------------------------------------------------------------------
# configuration and data containers
# --------------------------------------------------------------------------


@dataclass
class Config:
    """Run parameters shared by all datasets."""

    seed: int
    ks: Tuple[int, ...]
    cs: Tuple[int, ...]
    runs: int
    warmup: int
    timing_queries: int
    n: int
    n_queries: int


@dataclass
class Dataset:
    """A corpus plus held-out queries, both row-major and l2-normalised."""

    name: str
    source: str
    X: np.ndarray
    Qs: np.ndarray
    n_real: int
    n_queries_requested: int
    load_ms: float
    extra: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------
# datasets (recipe copied from benchmarks/research/scan/src/data.rs)
# --------------------------------------------------------------------------


def normalize_rows(a: np.ndarray) -> None:
    """l2-normalise rows in place; zero rows stay zero."""
    n = np.linalg.norm(a, axis=1, keepdims=True)
    np.maximum(n, np.float32(1e-12), out=n)
    a /= n


def synthetic(
    n: int,
    d: int,
    clusters: int,
    n_queries: int,
    seed: int,
    spread: float,
    alpha: float,
) -> Dataset:
    """Fixed-seed anisotropic Gaussian clusters.

    Centres c_k = normalize(w * g), w_j = (j+1)^-alpha; doc i belongs to
    cluster i mod C: x_i = normalize(c_k + spread * (w * g') / |w|).  Queries
    are fresh draws from random clusters.  Same model as the scan crate; the
    PRNG is numpy PCG64 rather than xoshiro, so rows are not bit-identical.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed ^ 0x5EED_0000_0000_0001)
    w = (np.arange(1, d + 1, dtype=np.float64) ** (-float(alpha))).astype(np.float32)
    noise_scale = np.float32(spread / float(np.linalg.norm(w)))
    centres = rng.standard_normal((clusters, d), dtype=np.float32) * w
    normalize_rows(centres)
    docs = np.empty((n, d), dtype=np.float32)
    step = 8192
    for a in range(0, n, step):
        b = min(n, a + step)
        k = np.arange(a, b) % clusters
        g = rng.standard_normal((b - a, d), dtype=np.float32)
        docs[a:b] = centres[k] + noise_scale * (w * g)
        normalize_rows(docs[a:b])
    qk = rng.integers(0, clusters, size=n_queries)
    g = rng.standard_normal((n_queries, d), dtype=np.float32)
    queries = centres[qk] + noise_scale * (w * g)
    normalize_rows(queries)
    return Dataset(
        name="synthetic",
        source=(
            f"synthetic gaussian clusters: seed={seed} clusters={clusters} "
            f"spread={spread} spectrum_alpha={alpha} (numpy PCG64)"
        ),
        X=np.ascontiguousarray(docs),
        Qs=np.ascontiguousarray(queries),
        n_real=n,
        n_queries_requested=n_queries,
        load_ms=(time.perf_counter() - t0) * 1e3,
        extra={"clusters": clusters, "spread": spread, "spectrum_alpha": alpha},
    )


def load_f32_rows(path: Path, dim: int) -> np.ndarray:
    """Little-endian f32 row-major file -> (rows, dim) float32 array."""
    arr = np.fromfile(str(path), dtype="<f4")
    if arr.size == 0 or arr.size % dim != 0:
        raise ValueError(f"{path}: {arr.size} floats is not a positive multiple of dim {dim}")
    return np.ascontiguousarray(arr.reshape(-1, dim).astype(np.float32))


def nfcorpus(directory: Path, n_total: int, n_queries: int, noise: float, seed: int) -> Dataset:
    """NFCorpus MiniLM rows padded with on-manifold distractors.

    Distractor rows are normalize(x_j + noise * g / sqrt(D)) around random
    real rows.  Queries are the first n_queries real queries after a seeded
    shuffle (all 323 when n_queries exceeds the file).
    """
    t0 = time.perf_counter()
    meta_path = directory / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    dim = int(meta["dim"])
    if dim <= 0:
        raise ValueError(f"{meta_path}: dim must be positive")
    docs = load_f32_rows(directory / "corpus.f32", dim)
    all_queries = load_f32_rows(directory / "queries.f32", dim)
    n_real = docs.shape[0]
    nq_all = all_queries.shape[0]
    if n_total < n_real:
        raise ValueError(f"--n {n_total} is smaller than the real corpus ({n_real} rows)")
    normalize_rows(docs)
    rng = np.random.default_rng(seed ^ 0x4E46_434F_5250_5553)
    X = np.empty((n_total, dim), dtype=np.float32)
    X[:n_real] = docs
    n_pad = n_total - n_real
    if n_pad > 0:
        src = rng.integers(0, n_real, size=n_pad)
        scale = np.float32(noise / float(np.sqrt(dim)))
        step = 8192
        for a in range(0, n_pad, step):
            b = min(n_pad, a + step)
            g = rng.standard_normal((b - a, dim), dtype=np.float32)
            X[n_real + a : n_real + b] = docs[src[a:b]] + scale * g
            normalize_rows(X[n_real + a : n_real + b])
    picked = rng.permutation(nq_all)[: min(n_queries, nq_all)]
    Qs = np.ascontiguousarray(all_queries[picked])
    normalize_rows(Qs)
    return Dataset(
        name="nfcorpus",
        source=(
            f"BEIR NFCorpus {meta.get('model', '?')} {dim}-d, {n_real} real rows + "
            f"{n_pad} on-manifold distractors (noise={noise})"
        ),
        X=X,
        Qs=Qs,
        n_real=n_real,
        n_queries_requested=n_queries,
        load_ms=(time.perf_counter() - t0) * 1e3,
        extra={"model": meta.get("model"), "distractor_noise": noise, "n_queries_in_file": nq_all},
    )


# --------------------------------------------------------------------------
# shared numerics
# --------------------------------------------------------------------------


def exact_scores(X: np.ndarray, Qs: np.ndarray) -> np.ndarray:
    """(nq, N) exact dot products, query rows contiguous."""
    nq = Qs.shape[0]
    S = np.empty((nq, X.shape[0]), dtype=np.float32)
    for a in range(0, nq, ROW_CHUNK):
        b = min(nq, a + ROW_CHUNK)
        S[a:b] = Qs[a:b] @ X.T
    return S


def topk_rows(M: np.ndarray, k: int) -> np.ndarray:
    """Indices (rows, k) of the k largest entries per row, unsorted."""
    r, c = M.shape
    k = min(k, c)
    out = np.empty((r, k), dtype=np.int64)
    for a in range(0, r, ROW_CHUNK):
        b = min(r, a + ROW_CHUNK)
        if k >= c:
            out[a:b] = np.arange(c, dtype=np.int64)
        else:
            out[a:b] = np.argpartition(M[a:b], c - k, axis=1)[:, c - k :]
    return out


def topk_rows_sorted(M: np.ndarray, k: int) -> np.ndarray:
    """Indices (rows, k) of the k largest entries per row, descending."""
    idx = topk_rows(M, k)
    vals = np.take_along_axis(M, idx, axis=1)
    order = np.argsort(-vals, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)


def recall_for_query(S_row: np.ndarray, gt_set: set, cand_sorted: np.ndarray, C: int) -> float:
    """Exact rerank of the first C candidates (a lookup into S) vs ground truth."""
    c = cand_sorted[:C]
    n = c.size
    if n > TOP:
        sc = S_row[c]
        c = c[np.argpartition(sc, n - TOP)[n - TOP :]]
    return len(gt_set.intersection(c.tolist())) / TOP


def make_groups(order: np.ndarray, K: int) -> List[np.ndarray]:
    """Consecutive groups of K query indices; the last may be short."""
    return [order[a : a + K] for a in range(0, order.size, K)]


def full_groups_for_timing(groups: Sequence[np.ndarray], K: int, timing_queries: int) -> List[np.ndarray]:
    full = [g for g in groups if g.size == K]
    return full[: max(2, timing_queries // K)]


def random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed orthogonal (d, d) float32 matrix."""
    A = rng.standard_normal((d, d))
    Qm, Rm = np.linalg.qr(A)
    Qm = Qm * np.sign(np.diag(Rm))[None, :]
    return np.ascontiguousarray(Qm.astype(np.float32))


def superpose_group(V: np.ndarray, mode: str, perms: Sequence[np.ndarray], signs: np.ndarray) -> np.ndarray:
    """Q = sum_s P_s V[s] with P_s a signed permutation ("key") or I ("sum")."""
    if mode == "sum":
        return V.sum(axis=0)
    Q = np.zeros(V.shape[1], dtype=np.float32)
    for s in range(V.shape[0]):
        Q += signs[s] * V[s, perms[s]]
    return Q


# --------------------------------------------------------------------------
# recall passes (batched arithmetic, identical selection rules to the timed path)
# --------------------------------------------------------------------------


def v1_recall(
    X: np.ndarray,
    Qs: np.ndarray,
    S: np.ndarray,
    gt_sets: List[set],
    groups: Sequence[np.ndarray],
    K: int,
    mode: str,
    perms: Sequence[np.ndarray],
    signs: np.ndarray,
    cs: Sequence[int],
) -> Dict[Tuple[str, int], float]:
    N, d = X.shape
    G = len(groups)
    cmax = min(max(cs), N)
    Qmat = np.zeros((G, d), dtype=np.float32)
    for s in range(K):
        gids = [g for g in range(G) if groups[g].size > s]
        if not gids:
            continue
        qidx = np.array([groups[g][s] for g in gids])
        V = Qs[qidx]
        if mode == "sum":
            Qmat[gids] += V
        else:
            Qmat[gids] += signs[s] * V[:, perms[s]]
    hits = {(sel, C): 0.0 for sel in V1_SELECTS for C in cs}
    for a in range(0, G, ROW_CHUNK):
        b = min(G, a + ROW_CHUNK)
        Smix = Qmat[a:b] @ X.T
        for sel in V1_SELECTS:
            key = np.abs(Smix) if sel == "abs" else Smix
            cand = topk_rows_sorted(key, cmax)
            for gi in range(a, b):
                for q in groups[gi].tolist():
                    for C in cs:
                        hits[(sel, C)] += recall_for_query(S[q], gt_sets[q], cand[gi - a], C)
    nq = sum(g.size for g in groups)
    return {k: v / nq for k, v in hits.items()}


def v3_recall(
    XT: np.ndarray,
    QT: np.ndarray,
    S: np.ndarray,
    gt_sets: List[set],
    groups: Sequence[np.ndarray],
    K: int,
    cs: Sequence[int],
) -> Tuple[Dict[int, float], np.ndarray]:
    """Per-block partial scores; returns recall per C and the block-major corpus."""
    N, d = XT.shape
    m = d // K
    cmax = min(max(cs), N)
    Xb = np.ascontiguousarray(XT.reshape(N, K, m).transpose(1, 0, 2))
    hits = {C: 0.0 for C in cs}
    for s in range(K):
        qidx = np.array([g[s] for g in groups if g.size > s])
        if qidx.size == 0:
            continue
        Vb = np.ascontiguousarray(QT[qidx][:, s * m : (s + 1) * m])
        XbT = Xb[s].T
        for a in range(0, qidx.size, ROW_CHUNK):
            b = min(qidx.size, a + ROW_CHUNK)
            P = Vb[a:b] @ XbT
            cand = topk_rows_sorted(P, cmax)
            for i in range(a, b):
                q = int(qidx[i])
                for C in cs:
                    hits[C] += recall_for_query(S[q], gt_sets[q], cand[i - a], C)
    nq = sum(g.size for g in groups)
    return {C: v / nq for C, v in hits.items()}, Xb


# --------------------------------------------------------------------------
# timed code paths (one group per call; return one top-10 array per query)
# --------------------------------------------------------------------------

GroupFn = Callable[[np.ndarray], List[np.ndarray]]


def single_fn(X: np.ndarray, Qs: np.ndarray) -> GroupFn:
    N = X.shape[0]

    def fn(g: np.ndarray) -> List[np.ndarray]:
        s = X @ Qs[g[0]]
        return [np.argpartition(s, N - TOP)[N - TOP :]]

    return fn


def v2_fn(X: np.ndarray, Qs: np.ndarray) -> GroupFn:
    N = X.shape[0]

    def fn(g: np.ndarray) -> List[np.ndarray]:
        V = Qs[g]
        S2 = X @ V.T
        top = np.argpartition(S2, N - TOP, axis=0)[N - TOP :]
        return [top[:, i] for i in range(g.size)]

    return fn


def v1_fn(
    X: np.ndarray,
    Qs: np.ndarray,
    mode: str,
    sel: str,
    C: int,
    perms: Sequence[np.ndarray],
    signs: np.ndarray,
) -> GroupFn:
    N = X.shape[0]

    def fn(g: np.ndarray) -> List[np.ndarray]:
        V = Qs[g]
        Q = superpose_group(V, mode, perms, signs)
        s = X @ Q
        key = np.abs(s) if sel == "abs" else s
        cand = np.argpartition(key, N - C)[N - C :]
        R = X[cand] @ V.T
        top = np.argpartition(R, C - TOP, axis=0)[C - TOP :]
        return [cand[top[:, i]] for i in range(g.size)]

    return fn


def v3_fn(
    X: np.ndarray,
    Qs: np.ndarray,
    Xb: np.ndarray,
    m: int,
    C: int,
    transform: Callable[[np.ndarray], np.ndarray],
) -> GroupFn:
    N = X.shape[0]

    def fn(g: np.ndarray) -> List[np.ndarray]:
        V = Qs[g]
        VT = transform(V)
        out: List[np.ndarray] = []
        for s in range(g.size):
            p = Xb[s] @ VT[s, s * m : (s + 1) * m]
            cand = np.argpartition(p, N - C)[N - C :]
            r = X[cand] @ V[s]
            out.append(cand[np.argpartition(r, C - TOP)[C - TOP :]])
        return out

    return fn


def axis_transform(perm: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return lambda V: V[:, perm]


def rot_transform(R: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    return lambda V: V @ R


def time_path(
    fn: GroupFn,
    groups: Sequence[np.ndarray],
    K: int,
    runs: int,
    warmup: int,
    gt_sets: List[set],
) -> Optional[Dict[str, object]]:
    """Warmup passes (first one cross-checks recall), then timed passes."""
    if not groups:
        return None
    path_hits = 0.0
    path_n = 0
    for w in range(max(1, warmup)):
        for g in groups:
            outs = fn(g)
            if w == 0:
                for q, top in zip(g.tolist(), outs):
                    path_hits += len(gt_sets[q].intersection(np.asarray(top).tolist())) / TOP
                    path_n += 1
    per_run: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for g in groups:
            fn(g)
        per_run.append((time.perf_counter() - t0) * 1e6 / (len(groups) * K))
    return {
        "us_per_query_median": float(np.median(per_run)),
        "us_per_query_min": float(min(per_run)),
        "us_per_query_runs": [float(x) for x in per_run],
        "timing_groups": len(groups),
        "timing_queries": len(groups) * K,
        "path_recall_at_10_timing_groups": path_hits / max(1, path_n),
    }


# --------------------------------------------------------------------------
# rows, verdict, evaluation
# --------------------------------------------------------------------------


def make_row(
    variant: str,
    mode: str,
    select: str,
    K: int,
    C: int,
    groups: Sequence[np.ndarray],
    recall: float,
    N: int,
    d: int,
    nq: int,
    timing: Optional[Dict[str, object]],
    single_us: Optional[float],
) -> Dict[str, object]:
    exact = variant in ("single", "v2")
    G = len(groups)
    scan_single = float(d) * float(N)
    if exact:
        scan_nom = scan_single
        scan_act = scan_single
        rerank = 0.0
        cand = 0
    elif variant == "v1":
        scan_nom = scan_single / K
        scan_act = scan_single * G / nq
        rerank = float(C) * float(d)
        cand = C
    else:
        scan_nom = scan_single / K
        scan_act = scan_nom
        rerank = float(C) * float(d)
        cand = C
    total = scan_nom + rerank
    row: Dict[str, object] = {
        "variant": variant,
        "mode": mode,
        "select": select,
        "K": int(K),
        "C": int(C),
        "groups": int(G),
        "exact_scan": exact,
        "recall_at_10": float(recall),
        "rerank_candidates_per_query": int(cand),
        "scan_flops_per_query": scan_nom,
        "scan_flops_per_query_actual": scan_act,
        "rerank_flops_per_query": rerank,
        "total_flops_per_query": total,
        "scan_flop_reduction_vs_single": scan_single / scan_nom,
        "recall_per_scan_flop_vs_single": float(recall) * scan_single / scan_nom,
        "recall_per_total_flop_vs_single": float(recall) * scan_single / total,
        "us_per_query_median": None,
        "us_per_query_min": None,
        "us_per_query_runs": [],
        "timing_groups": 0,
        "timing_queries": 0,
        "path_recall_at_10_timing_groups": None,
        "speedup_vs_single_wall": None,
    }
    if timing is not None:
        row.update(timing)
        us = float(timing["us_per_query_median"])
        if single_us is not None and us > 0:
            row["speedup_vs_single_wall"] = single_us / us
    return row


def row_passes(r: Dict[str, object]) -> bool:
    return (
        r["variant"] in ("v1", "v3")
        and int(r["K"]) >= 4
        and int(r["C"]) <= 1000
        and float(r["recall_at_10"]) >= 0.95
        and float(r["scan_flop_reduction_vs_single"]) >= 2.0
    )


def fmt_row(r: Dict[str, object]) -> str:
    us = r["us_per_query_median"]
    sp = r["speedup_vs_single_wall"]
    sp = float("nan") if sp is None else sp
    pr = r["path_recall_at_10_timing_groups"]
    pr = float("nan") if pr is None else pr
    return (
        f"  {r['variant']:6} {r['mode'] or '-':4} {r['select']:5} K={r['K']:2d} C={r['C']:5d} "
        f"recall@10={r['recall_at_10']:.4f} cand={r['rerank_candidates_per_query']:5d} "
        f"scanMAC/q={r['scan_flops_per_query']:.3e} (x{r['scan_flop_reduction_vs_single']:4.1f}) "
        f"total={r['total_flops_per_query']:.3e} "
        f"us/q={us:9.1f} speedup={sp:5.2f} path_recall={pr:.3f}"
        if us is not None
        else f"  {r['variant']:6} {r['mode'] or '-':4} {r['select']:5} K={r['K']:2d} C={r['C']:5d} "
        f"recall@10={r['recall_at_10']:.4f} (no full group to time)"
    )


def evaluate_dataset(ds: Dataset, cfg: Config) -> Dict[str, object]:
    X, Qs = ds.X, ds.Qs
    N, d = X.shape
    nq = Qs.shape[0]
    log(f"[{ds.name}] N={N} d={d} queries={nq} load={ds.load_ms:.0f} ms")

    t0 = time.perf_counter()
    S = exact_scores(X, Qs)
    gt = topk_rows(S, TOP)
    gt_sets = [set(r.tolist()) for r in gt]
    gt_ms = (time.perf_counter() - t0) * 1e3
    log(f"[{ds.name}] exact top-{TOP} ground truth in {gt_ms:.0f} ms")

    krng = np.random.default_rng(cfg.seed ^ 0x4B45_5953_0000_0001)
    kmax = max(cfg.ks)
    perms = [krng.permutation(d) for _ in range(kmax)]
    signs = krng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(kmax, d))
    axis_perm = krng.permutation(d)
    t0 = time.perf_counter()
    R = random_rotation(d, krng)
    rot_ms = (time.perf_counter() - t0) * 1e3
    order = krng.permutation(nq)

    rows: List[Dict[str, object]] = []
    notes: List[str] = []

    # single-query exact scan (the FLOP and wall-time reference)
    groups1 = make_groups(order, 1)
    tg = full_groups_for_timing(groups1, 1, cfg.timing_queries)
    tm = time_path(single_fn(X, Qs), tg, 1, cfg.runs, cfg.warmup, gt_sets)
    single_us = float(tm["us_per_query_median"]) if tm else None
    rows.append(make_row("single", "", "exact", 1, 0, groups1, 1.0, N, d, nq, tm, single_us))
    log(fmt_row(rows[-1]))

    # v2: ordinary batching
    for K in cfg.ks:
        groups = make_groups(order, K)
        tg = full_groups_for_timing(groups, K, cfg.timing_queries)
        tm = time_path(v2_fn(X, Qs), tg, K, cfg.runs, cfg.warmup, gt_sets)
        rows.append(make_row("v2", "", "exact", K, 0, groups, 1.0, N, d, nq, tm, single_us))
        log(fmt_row(rows[-1]))

    # v1: keyed and plain superposition with a shared candidate pool
    for mode in V1_MODES:
        for K in cfg.ks:
            groups = make_groups(order, K)
            tg = full_groups_for_timing(groups, K, cfg.timing_queries)
            t0 = time.perf_counter()
            rec = v1_recall(X, Qs, S, gt_sets, groups, K, mode, perms, signs, cfg.cs)
            rec_ms = (time.perf_counter() - t0) * 1e3
            for sel in V1_SELECTS:
                for C in cfg.cs:
                    Ceff = min(C, N)
                    fn = v1_fn(X, Qs, mode, sel, Ceff, perms, signs)
                    tm = time_path(fn, tg, K, cfg.runs, cfg.warmup, gt_sets)
                    rows.append(make_row("v1", mode, sel, K, Ceff, groups, rec[(sel, C)], N, d, nq, tm, single_us))
                    log(fmt_row(rows[-1]))
            log(f"[{ds.name}] v1/{mode} K={K}: recall pass {rec_ms:.0f} ms")

    # v3: sign-split superposition (disjoint coordinate blocks)
    for mode in V3_MODES:
        t0 = time.perf_counter()
        if mode == "axis":
            XT = np.ascontiguousarray(X[:, axis_perm])
            QT = np.ascontiguousarray(Qs[:, axis_perm])
            transform = axis_transform(axis_perm)
        else:
            XT = np.ascontiguousarray(X @ R)
            QT = np.ascontiguousarray(Qs @ R)
            transform = rot_transform(R)
        xf_ms = (time.perf_counter() - t0) * 1e3
        log(f"[{ds.name}] v3/{mode} corpus transform {xf_ms:.0f} ms (rotation fit {rot_ms:.0f} ms)")
        for K in cfg.ks:
            if d % K != 0:
                notes.append(f"v3/{mode} K={K} skipped: d={d} not divisible by K")
                continue
            m = d // K
            groups = make_groups(order, K)
            tg = full_groups_for_timing(groups, K, cfg.timing_queries)
            t0 = time.perf_counter()
            rec, Xb = v3_recall(XT, QT, S, gt_sets, groups, K, cfg.cs)
            rec_ms = (time.perf_counter() - t0) * 1e3
            for C in cfg.cs:
                Ceff = min(C, N)
                fn = v3_fn(X, Qs, Xb, m, Ceff, transform)
                tm = time_path(fn, tg, K, cfg.runs, cfg.warmup, gt_sets)
                rows.append(make_row("v3", mode, "own", K, Ceff, groups, rec[C], N, d, nq, tm, single_us))
                log(fmt_row(rows[-1]))
            del Xb
            log(f"[{ds.name}] v3/{mode} K={K}: recall pass {rec_ms:.0f} ms")
        del XT, QT

    pass_rows = [r for r in rows if row_passes(r)]
    eligible = [r for r in rows if r["variant"] in ("v1", "v3") and int(r["K"]) >= 4 and int(r["C"]) <= 1000]
    best: Dict[str, object] = {}
    for variant in ("v1", "v3"):
        cand = [r for r in eligible if r["variant"] == variant]
        if cand:
            b = max(cand, key=lambda r: (float(r["recall_at_10"]), -int(r["C"])))
            best[variant] = {k: b[k] for k in ("mode", "select", "K", "C", "recall_at_10", "scan_flop_reduction_vs_single")}
    return {
        "name": ds.name,
        "source": ds.source,
        "n_docs": int(N),
        "n_real_docs": int(ds.n_real),
        "dim": int(d),
        "n_queries": int(nq),
        "n_queries_requested": int(ds.n_queries_requested),
        "load_ms": ds.load_ms,
        "ground_truth_ms": gt_ms,
        "extra": ds.extra,
        "single_us_per_query": single_us,
        "rows": rows,
        "best_eligible": best,
        "pass_rows": [
            {k: r[k] for k in ("variant", "mode", "select", "K", "C", "recall_at_10", "scan_flop_reduction_vs_single")}
            for r in pass_rows
        ],
        "pass": bool(pass_rows),
        "notes": notes,
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def blas_info() -> object:
    try:
        info = np.show_config(mode="dicts")
        return info.get("Build Dependencies", {}).get("blas", info)
    except Exception as exc:  # numpy < 1.25 has no dict mode
        return {"unavailable": str(exc)}


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.write("\n")
    os.replace(tmp, path)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Holographic query batching kill test (numpy, single core).")
    p.add_argument("--datasets", default="synthetic,nfcorpus", help="comma list of synthetic,nfcorpus")
    p.add_argument("--nfcorpus-dir", default=str(Path.home() / ".cache" / "warm-bench" / "nfcorpus"))
    p.add_argument("--out", default=None, help=f"output JSON (default {here / 'out' / 'hqb_v1.json'})")
    p.add_argument("--n", type=int, default=50_000, help="corpus size (synthetic: generated; nfcorpus: real + distractors)")
    p.add_argument("--queries", type=int, default=500, help="held-out queries (nfcorpus: capped at the 323 in the file)")
    p.add_argument("--dim", type=int, default=768, help="synthetic dimension")
    p.add_argument("--clusters", type=int, default=200)
    p.add_argument("--spread", type=float, default=0.75)
    p.add_argument("--spectrum-alpha", type=float, default=0.5)
    p.add_argument("--distractor-noise", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ks", default=",".join(str(k) for k in KS_DEFAULT))
    p.add_argument("--cs", default=",".join(str(c) for c in CS_DEFAULT))
    p.add_argument("--runs", type=int, default=5, help="timed repeats per configuration (median reported)")
    p.add_argument("--warmup", type=int, default=1, help="warmup passes per configuration")
    p.add_argument("--timing-queries", type=int, default=16, help="queries per timed pass (>= 2 groups)")
    p.add_argument("--quick", action="store_true", help="smoke run: n=5000, queries=50, runs=3, out/hqb_quick.json")
    args = p.parse_args(argv)
    if args.quick:
        args.n = min(args.n, 5000)
        args.queries = min(args.queries, 50)
        args.runs = min(args.runs, 3)
        args.timing_queries = min(args.timing_queries, 8)
        if args.out is None:
            args.out = str(here / "out" / "hqb_quick.json")
    if args.out is None:
        args.out = str(here / "out" / "hqb_v1.json")
    return args


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    ks = tuple(sorted({int(k) for k in args.ks.split(",") if k.strip()}))
    cs = tuple(sorted({int(c) for c in args.cs.split(",") if c.strip()}))
    if not ks or min(ks) < 1 or not cs or min(cs) <= TOP:
        log(f"invalid --ks {args.ks!r} or --cs {args.cs!r} (C must exceed {TOP})")
        return 2
    if args.runs < 1 or args.n < 1 or args.queries < 1 or args.seed < 0:
        log("--runs, --n and --queries must be positive and --seed non-negative")
        return 2
    cfg = Config(
        seed=args.seed,
        ks=ks,
        cs=cs,
        runs=args.runs,
        warmup=args.warmup,
        timing_queries=args.timing_queries,
        n=args.n,
        n_queries=args.queries,
    )
    names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    unknown = [s for s in names if s not in ("synthetic", "nfcorpus")]
    if unknown or not names:
        log(f"unknown dataset(s) {unknown}; choose from synthetic,nfcorpus")
        return 2

    t_start = time.perf_counter()
    log(f"[hqb] seed={cfg.seed} n={cfg.n} queries={cfg.n_queries} ks={ks} cs={cs} runs={cfg.runs} threads=1")
    results: Dict[str, object] = {}
    for name in names:
        try:
            if name == "synthetic":
                ds = synthetic(cfg.n, args.dim, args.clusters, cfg.n_queries, cfg.seed, args.spread, args.spectrum_alpha)
            else:
                ds = nfcorpus(Path(args.nfcorpus_dir).expanduser(), cfg.n, cfg.n_queries, args.distractor_noise, cfg.seed)
        except (OSError, ValueError, KeyError) as exc:
            log(f"[hqb] cannot load dataset {name}: {exc}")
            return 2
        results[name] = evaluate_dataset(ds, cfg)
        del ds

    overall = all(bool(r["pass"]) for r in results.values())
    payload: Dict[str, object] = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": cfg.seed,
        "config": {
            "ks": list(ks),
            "cs": list(cs),
            "top": TOP,
            "runs": cfg.runs,
            "warmup": cfg.warmup,
            "timing_queries": cfg.timing_queries,
            "n": cfg.n,
            "queries": cfg.n_queries,
            "synthetic": {"dim": args.dim, "clusters": args.clusters, "spread": args.spread, "spectrum_alpha": args.spectrum_alpha},
            "nfcorpus": {"dir": str(Path(args.nfcorpus_dir).expanduser()), "distractor_noise": args.distractor_noise},
            "quick": bool(args.quick),
            "flop_convention": "multiply-accumulates; single-query scan = d*N per query",
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "blas": blas_info(),
        },
        "datasets": results,
        "verdict": {
            "pass": overall,
            "rule": PASS_RULE,
            "per_dataset": {name: bool(r["pass"]) for name, r in results.items()},
        },
        "total_ms": (time.perf_counter() - t_start) * 1e3,
    }
    try:
        write_json(Path(args.out), payload)
    except OSError as exc:
        log(f"[hqb] cannot write {args.out}: {exc}")
        return 1
    log(f"[hqb] wrote {args.out} in {payload['total_ms'] / 1e3:.1f} s")
    for name, r in results.items():
        verdict = "PASS" if r["pass"] else "FAIL"
        best = r["best_eligible"]
        log(f"[hqb] {name}: {verdict}; best eligible rows (K>=4, C<=1000): {json.dumps(best, default=str)}")
    log(f"[hqb] rule: {PASS_RULE}")
    log("PASS" if overall else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
