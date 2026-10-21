#!/usr/bin/env python3
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""relevance: query-time mechanisms above static hybrid fusion on NFCorpus (round-2 kill test).

Question. Static z-score fusion of BM25 and MiniLM-L6 cosine (dense weight tuned on the odd
queries) scores nDCG@10 0.3459 on the even queries; the best-channel-per-query oracle is 0.3774
and the best-weight-per-query oracle 0.3970 (roc_v1.json). Which label-free mechanism, using only
what the engine holds at query time (BM25 scores, corpus and query embeddings, term statistics),
captures a large part of that headroom?

Mechanisms (every one ranks inside a per-query candidate pool and fuses by z-score sum):
  zsum            baseline: (1-w) z(BM25) + w z(cos) on the pool BM25 top-200 (positive) | cos top-200.
  rrf_k60         competitor baseline: 1/(60+rank_bm25) + 1/(60+rank_cos), no tuning.
  prf_dense       (a) q' = normalize(q + alpha * mean(D[top-m of the baseline fused list])), m in {3,5,10},
                  alpha in {0.5,1,2}; cos' = D q' over the whole corpus; pool |= cos' top-200; re-fuse
                  with a second-stage weight w' from the grid.
  gate_jaccard20  (c) Jaccard of BM25 top-20 (positive) and cos top-20 chooses w: queries are bucketed
                  by tune-fold quantiles (2, 3 or 4 buckets); each bucket takes the weight that
                  maximises the tune-fold mean nDCG@10 of its members.
  gate_*_margin   (d) same bucketing on cos top1-top2 margin, relative BM25 margin (s1-s2)/s1, their
                  difference in tune-fold midrank ECDF units, and a 2-D (cos x BM25) margin grid.
  centroid_rerank (e) score = (1-w) z(BM25) + w z(cos) + beta z(cos(doc, c_b)) where c_b is the
                  normalized mean embedding of the positive BM25 top-m (m in {3,5}), beta in
                  {0.25,0.5,1,2}; c_b = q when no BM25 hit exists.
  two_stage_prf   (f) baseline fused top-100 re-ranked by (1-lambda) z(fused) + lambda z(cos'),
                  lambda in {0.5,0.75,1}; lambda = 1 is the pure cosine-to-PRF-query re-rank.
  prf_gate_*      (g) prf_dense tables with the second-stage weight chosen by the (c)/(d) gates
                  computed on the first-pass channels.

Protocol. Queries with no judged relevant document are dropped (IDCG = 0). Fold "primary" tunes on
odd query indices (queries.jsonl order, 0-based) and evaluates on even; fold "swapped" mirrors it.
Every parameter (baseline weight w0, PRF m/alpha/w', bucket count and per-bucket weights,
beta/m, lambda) is chosen by the tune-fold mean nDCG@10; the held-out number is reported once.
The winner is the non-baseline family with the highest tune-fold score. Paired percentile
bootstrap (2000 resamples, seed 0) of family minus zsum over the held-out queries. Oracle gap
closed = (family - zsum) / (oracle - zsum) on the held-out fold for both oracles. Verdict WIN if
the chosen family beats zsum by >= 0.01 on the primary held-out fold with the CI excluding 0.

Data layout. S_b, S_c, S_c' are dense (n_q, n_d) float64 score matrices. A pool is a compact
(n_q, width) index matrix with a validity mask (roc.Candidates); channel scores are gathered on
it, z-normalised over the valid entries, fused, and the top-100 taken with the document-index
tie-break. A Table holds per-weight, per-query nDCG@10 and recall@100 of one dense channel.

Invariants. Pools always contain at least 200 valid entries (the dense top-200 is always
present), so top-100 is defined. Nothing that touches eval-fold labels enters any choice. All
randomness is the bootstrap generator with a fixed seed. Costs are counted in extra dim-384 dot
products per query beyond the baseline's one corpus pass.

Reuses the sibling module roc.py for input parsing, BM25, metrics, pools and bootstrap.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import roc  # noqa: E402
from roc import BOOT_N, BOOT_SEED, CAND_K, NDCG_K, RRF_K, WEIGHTS, Candidates  # noqa: E402

SCHEMA = "hms.research.relevance/1"
RECALL_K = 100
AGREE_TOP = 20
PRF_M = (3, 5, 10)
PRF_ALPHA = (0.5, 1.0, 2.0)
BUCKETS_1D = (2, 3, 4)
BUCKETS_2D = (2, 3)
CENTROID_M = (3, 5)
CENTROID_BETA = (0.25, 0.5, 1.0, 2.0)
STAGE2_LAMBDA = (0.5, 0.75, 1.0)
STAGE2_TOP = 100
WIN_MARGIN = 0.01
MARGIN_STATS = ("cos_margin", "bm25_margin", "margin_diff")
INPUT_FILES = ("meta.json", "corpus.jsonl", "queries.jsonl", "qrels.tsv", "corpus.f32", "queries.f32")


# ----------------------------------------------------------------------------------------------
# pools, tables, sweeps
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Table:
    """Per-weight (rows follow WEIGHTS), per-query nDCG@10 and recall@100 of one fused channel pair."""

    nd: np.ndarray
    rc: np.ndarray


def pool_from_mask(mask: np.ndarray) -> Candidates:
    """Compact candidate pool from a boolean (n_q, n_d) membership matrix (ascending document index)."""
    width = int(mask.sum(axis=1).max())
    idx = np.zeros((mask.shape[0], width), dtype=np.int64)
    valid = np.zeros((mask.shape[0], width), dtype=bool)
    for q in range(mask.shape[0]):
        docs = np.flatnonzero(mask[q])
        idx[q, : docs.size] = docs
        valid[q, : docs.size] = True
    return Candidates(idx, valid)


def z_channels(S_b: np.ndarray, S_dense: np.ndarray, pool: Candidates) -> tuple[np.ndarray, np.ndarray]:
    """z-scored BM25 and dense scores on the pool."""
    return roc.z_norm(roc.gather(S_b, pool), pool.valid), roc.z_norm(roc.gather(S_dense, pool), pool.valid)


def fused_scores(Bz: np.ndarray, Cz: np.ndarray, w: float, pool: Candidates, extra: np.ndarray | None = None, beta: float = 0.0) -> np.ndarray:
    """(1-w) Bz + w Cz (+ beta extra) on valid entries, -inf elsewhere."""
    F = (1.0 - w) * Bz + w * Cz
    if extra is not None:
        F = F + beta * extra
    return np.where(pool.valid, F, -np.inf)


def top_docs(F: np.ndarray, pool: Candidates, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k (document indices, scores) of a fused pool matrix, descending, index tie-break."""
    cols = roc.topk(F, k)
    return np.take_along_axis(pool.idx, cols, axis=1), np.take_along_axis(F, cols, axis=1)


def sweep(Bz: np.ndarray, Cz: np.ndarray, pool: Candidates, lab: dict, extra: np.ndarray | None = None, beta: float = 0.0,
          keep_top: bool = False) -> tuple[Table, np.ndarray | None, np.ndarray | None]:
    """Score every weight in WEIGHTS; optionally return the fused top-100 docs and scores per weight."""
    nd, rc, tops, fsc = [], [], [], []
    for w in WEIGHTS:
        docs, sc = top_docs(fused_scores(Bz, Cz, w, pool, extra, beta), pool, RECALL_K)
        nd.append(roc.ndcg10(docs, lab["R"], lab["idcg"]))
        rc.append(roc.recall_at(docs, lab["rel"], RECALL_K))
        if keep_top:
            tops.append(docs)
            fsc.append(sc)
    table = Table(np.stack(nd), np.stack(rc))
    if keep_top:
        return table, np.stack(tops), np.stack(fsc)
    return table, None, None


# ----------------------------------------------------------------------------------------------
# query-time statistics and gates
# ----------------------------------------------------------------------------------------------


def first_pass_stats(S_b: np.ndarray, S_c: np.ndarray, order_b: np.ndarray, order_c: np.ndarray, n_d: int) -> dict[str, np.ndarray]:
    """Label-free per-query statistics of the two first-pass channels (no extra dot products)."""
    mb = roc.topk_mask(order_b, n_d, AGREE_TOP, S_b)
    mc = roc.topk_mask(order_c, n_d, AGREE_TOP)
    inter = (mb & mc).sum(axis=1).astype(np.float64)
    union = (mb | mc).sum(axis=1).astype(np.float64)
    jaccard = np.where(union > 0, inter / np.maximum(union, 1.0), 0.0)
    c12 = np.take_along_axis(S_c, order_c[:, :2], axis=1)
    b12 = np.take_along_axis(S_b, order_b[:, :2], axis=1)
    b1 = np.where(b12[:, 0] > 0, b12[:, 0], 1.0)
    bm25_margin = np.where(b12[:, 0] > 0, (b12[:, 0] - b12[:, 1]) / b1, 0.0)
    return {"jaccard20": jaccard, "cos_margin": c12[:, 0] - c12[:, 1], "bm25_margin": bm25_margin}


def assign_buckets(stat: np.ndarray, tune: np.ndarray, nb: int) -> tuple[np.ndarray, np.ndarray]:
    """Bucket index 0..nb-1 per query from tune-fold quantile edges (ties fall into the lower bucket)."""
    edges = np.quantile(stat[tune], np.linspace(0.0, 1.0, nb + 1)[1:-1])
    return np.searchsorted(edges, stat, side="right"), edges


def bucket_gate(table: Table, bucket: np.ndarray, n_cells: int, tune: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float], list[int]]:
    """Per-cell dense weight = argmax of the tune-fold mean nDCG@10 of the cell's tune members.

    Cells without tune members take the global tune-best weight. Returns per-query nDCG@10,
    recall@100, the weight chosen per cell and the tune member count per cell.
    """
    glob = int(np.argmax(table.nd[:, tune].mean(axis=1)))
    w_idx = np.full(bucket.size, glob, dtype=np.int64)
    chosen: list[float] = []
    sizes: list[int] = []
    for c in range(n_cells):
        members = tune[bucket[tune] == c]
        bi = glob if members.size == 0 else int(np.argmax(table.nd[:, members].mean(axis=1)))
        w_idx[bucket == c] = bi
        chosen.append(WEIGHTS[bi])
        sizes.append(int(members.size))
    cols = np.arange(bucket.size)
    return table.nd[w_idx, cols], table.rc[w_idx, cols], chosen, sizes


def gate_variants(stats: dict[str, np.ndarray], names: tuple[str, ...], tune: np.ndarray) -> list[tuple[dict, np.ndarray, int]]:
    """All (params, bucket assignment, n_cells) candidates for the named statistics; "margin_2d" is the grid."""
    out: list[tuple[dict, np.ndarray, int]] = []
    for name in names:
        if name == "margin_2d":
            for nb in BUCKETS_2D:
                bc, ec = assign_buckets(stats["cos_margin"], tune, nb)
                bb, eb = assign_buckets(stats["bm25_margin"], tune, nb)
                out.append(({"statistic": name, "buckets_per_axis": nb, "edges_cos": ec.tolist(), "edges_bm25": eb.tolist()}, bc * nb + bb, nb * nb))
        else:
            for nb in BUCKETS_1D:
                b, e = assign_buckets(stats[name], tune, nb)
                out.append(({"statistic": name, "buckets": nb, "edges": e.tolist()}, b, nb))
    return out


def select_gated(tables: dict, variants: list[tuple[dict, np.ndarray, int]], tune: np.ndarray) -> tuple[dict, int]:
    """Best (table, gate variant) by tune mean; tables maps a params dict (as tuple key) to a Table."""
    best: dict | None = None
    n = 0
    for tkey, table in tables.items():
        for params, bucket, cells in variants:
            nd, rc, chosen, sizes = bucket_gate(table, bucket, cells, tune)
            n += 1
            t = float(nd[tune].mean())
            if best is None or t > best["tune"]:
                best = {"params": {**dict(tkey), **params, "weight_per_cell": chosen, "tune_members_per_cell": sizes}, "tune": t, "nd": nd, "rc": rc}
    assert best is not None
    return best, n


def select_static(tables: dict, tune: np.ndarray) -> tuple[dict, int]:
    """Best (table, weight) by tune mean over every table and every weight."""
    best: dict | None = None
    n = 0
    for tkey, table in tables.items():
        means = table.nd[:, tune].mean(axis=1)
        n += len(WEIGHTS)
        bi = int(np.argmax(means))
        if best is None or means[bi] > best["tune"]:
            best = {"params": {**dict(tkey), "dense_weight": WEIGHTS[bi]}, "tune": float(means[bi]), "nd": table.nd[bi], "rc": table.rc[bi]}
    assert best is not None
    return best, n


# ----------------------------------------------------------------------------------------------
# dense PRF and two-stage rerank
# ----------------------------------------------------------------------------------------------


def normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2-normalise rows (zero rows stay zero)."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 0, n, 1.0)


def prf_tables(ctx: dict, w0i: int, cache: dict) -> dict:
    """PRF tables for every (m, alpha), seeded by the baseline fused top-m at weight index w0i (memoised)."""
    if w0i in cache:
        return cache[w0i]
    D, Qk, S_b, lab = ctx["D"], ctx["Qk"], ctx["S_b"], ctx["lab"]
    seed_top = ctx["T0_top"][w0i]
    out: dict = {}
    for m in PRF_M:
        cent = D[seed_top[:, :m]].mean(axis=1)
        for a in PRF_ALPHA:
            qp = normalize_rows(Qk + a * cent)
            S_cp = qp @ D.T
            order_cp = roc.argsort_desc(S_cp)
            pool = pool_from_mask(ctx["mask0"] | roc.topk_mask(order_cp, D.shape[0], CAND_K))
            Bz, Cz = z_channels(S_b, S_cp, pool)
            table, _, _ = sweep(Bz, Cz, pool, lab)
            out[(("prf_m", m), ("prf_alpha", a))] = {"table": table, "S_cp": S_cp, "pool_width": float(pool.valid.sum(axis=1).mean())}
    cache[w0i] = out
    return out


def two_stage(ctx: dict, w0i: int, S_cp: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Re-rank the baseline fused top-100 by (1-lam) z(fused) + lam z(cos'); recall@100 is unchanged."""
    top = ctx["T0_top"][w0i]
    F = ctx["T0_score"][w0i]
    C = np.take_along_axis(S_cp, top, axis=1)
    full = np.ones_like(top, dtype=bool)
    score = (1.0 - lam) * roc.z_norm(F, full) + lam * roc.z_norm(C, full)
    order = np.argsort(-roc.tie_broken(score), axis=1, kind="stable")[:, :NDCG_K]
    docs = np.take_along_axis(top, order, axis=1)
    return roc.ndcg10(docs, ctx["lab"]["R"], ctx["lab"]["idcg"]), ctx["T0"].rc[w0i]


def centroid_tables(ctx: dict) -> dict:
    """Tables of (e): baseline channels plus beta z(cos(doc, centroid of positive BM25 top-m))."""
    D, Qk, S_b, order_b, pool, lab = ctx["D"], ctx["Qk"], ctx["S_b"], ctx["order_b"], ctx["pool0"], ctx["lab"]
    Bz, Cz = ctx["Bz0"], ctx["Cz0"]
    out: dict = {}
    for m in CENTROID_M:
        topb = order_b[:, :m]
        pos = (np.take_along_axis(S_b, topb, axis=1) > 0).astype(np.float64)
        cnt = pos.sum(axis=1)
        cent = (D[topb] * pos[:, :, None]).sum(axis=1) / np.maximum(cnt, 1.0)[:, None]
        cent = normalize_rows(np.where(cnt[:, None] > 0, cent, Qk))
        Ez = roc.z_norm(roc.gather(cent @ D.T, pool), pool.valid)
        for beta in CENTROID_BETA:
            table, _, _ = sweep(Bz, Cz, pool, lab, extra=Ez, beta=beta)
            out[(("centroid_m", m), ("beta", beta))] = table
    return out


# ----------------------------------------------------------------------------------------------
# folds
# ----------------------------------------------------------------------------------------------


def evaluate_fold(name: str, tune: np.ndarray, evl: np.ndarray, ctx: dict, prf_cache: dict) -> dict:
    """Select every family on `tune`, report on `evl`, bootstrap each family against zsum on `evl`."""
    T0, n_d = ctx["T0"], ctx["D"].shape[0]
    w0i = int(np.argmax(T0.nd[:, tune].mean(axis=1)))
    w0 = WEIGHTS[w0i]
    zs_nd, zs_rc = T0.nd[w0i], T0.rc[w0i]
    zs_eval = float(zs_nd[evl].mean())
    oracle_bc = float(np.maximum(ctx["nd_bm25"], ctx["nd_cos"])[evl].mean())
    oracle_bw = float(T0.nd.max(axis=0)[evl].mean())
    families: dict[str, dict] = {}
    per_query: dict[str, np.ndarray] = {}

    def gap(x: float, oracle: float) -> float | None:
        return (x - zs_eval) / (oracle - zs_eval) if oracle > zs_eval else None

    def add(fname: str, params: dict, nd: np.ndarray, rc: np.ndarray, n_configs: int, cost: dict) -> None:
        e = float(nd[evl].mean())
        per_query[fname] = nd
        families[fname] = {
            "params": params,
            "ndcg10_tune": float(nd[tune].mean()),
            "ndcg10_eval": e,
            "recall100_tune": float(rc[tune].mean()),
            "recall100_eval": float(rc[evl].mean()),
            "delta_vs_zsum_eval": e - zs_eval,
            "gap_closed_best_weight_eval": gap(e, oracle_bw),
            "gap_closed_best_channel_eval": gap(e, oracle_bc),
            "bootstrap_vs_zsum_eval": roc.paired_bootstrap(nd[evl] - zs_nd[evl]),
            "n_configs": n_configs,
            "cost": cost,
        }

    add("zsum", {"dense_weight": w0, "norm": "z"}, zs_nd, zs_rc, len(WEIGHTS), {"extra_dots": 0, "note": "baseline"})
    add("rrf_k60", {"k": RRF_K, "dense_weight": 0.5}, ctx["rrf_nd"], ctx["rrf_rc"], 1, {"extra_dots": 0, "note": "competitor baseline, untuned"})

    prf = prf_tables(ctx, w0i, prf_cache)
    prf_t = {k: v["table"] for k, v in prf.items()}
    best, n = select_static(prf_t, tune)
    width = prf[(("prf_m", best["params"]["prf_m"]), ("prf_alpha", best["params"]["prf_alpha"]))]["pool_width"]
    add("prf_dense", {**best["params"], "seed_weight": w0, "seed": "baseline fused top-m", "mean_pool_width": width}, best["nd"], best["rc"], n,
        {"extra_dots": n_d, "note": "one centroid (m vector adds) + full-corpus re-score with q'; pool-restricted variant would cost mean_pool_width"})

    stats = dict(ctx["stats"])
    stats["margin_diff"] = roc.ecdf_midrank(stats["cos_margin"][tune], stats["cos_margin"]) - roc.ecdf_midrank(stats["bm25_margin"][tune], stats["bm25_margin"])
    base_t = {(("channel", "cos"),): T0}
    best, n = select_gated(base_t, gate_variants(stats, ("jaccard20",), tune), tune)
    add("gate_jaccard20", best["params"], best["nd"], best["rc"], n, {"extra_dots": 0, "note": "set overlap of two top-20 lists"})
    for stat in MARGIN_STATS:
        best, n = select_gated(base_t, gate_variants(stats, (stat,), tune), tune)
        add(f"gate_{stat}", best["params"], best["nd"], best["rc"], n, {"extra_dots": 0, "note": "two scalar reads per channel"})
    best, n = select_gated(base_t, gate_variants(stats, ("margin_2d",), tune), tune)
    add("gate_margin_2d", best["params"], best["nd"], best["rc"], n, {"extra_dots": 0, "note": "two scalar reads per channel"})

    best, n = select_static(ctx["cent_tables"], tune)
    add("centroid_rerank", best["params"], best["nd"], best["rc"], n,
        {"extra_dots": int(round(ctx["pool0_width"])), "note": "one centroid + one dot product per pool document (mean pool width)"})

    best_s2: dict | None = None
    n = 0
    for key, entry in prf.items():
        for lam in STAGE2_LAMBDA:
            nd, rc = two_stage(ctx, w0i, entry["S_cp"], lam)
            n += 1
            t = float(nd[tune].mean())
            if best_s2 is None or t > best_s2["tune"]:
                best_s2 = {"params": {**dict(key), "lambda": lam, "seed_weight": w0, "rerank_depth": STAGE2_TOP}, "tune": t, "nd": nd, "rc": rc}
    assert best_s2 is not None
    add("two_stage_prf", best_s2["params"], best_s2["nd"], best_s2["rc"], n, {"extra_dots": STAGE2_TOP, "note": "one centroid + 100 dot products with q'"})

    best, n = select_gated(prf_t, gate_variants(stats, ("jaccard20",), tune), tune)
    add("prf_gate_jaccard20", {**best["params"], "seed_weight": w0}, best["nd"], best["rc"], n, {"extra_dots": n_d, "note": "prf_dense cost; gate is free"})
    best, n = select_gated(prf_t, gate_variants(stats, MARGIN_STATS + ("margin_2d",), tune), tune)
    add("prf_gate_margin", {**best["params"], "seed_weight": w0}, best["nd"], best["rc"], n, {"extra_dots": n_d, "note": "prf_dense cost; gate is free"})

    candidates = [k for k in families if k not in ("zsum", "rrf_k60")]
    chosen = max(candidates, key=lambda k: families[k]["ndcg10_tune"])
    ch = families[chosen]
    boot = ch["bootstrap_vs_zsum_eval"]
    passes = ch["delta_vs_zsum_eval"] >= WIN_MARGIN and boot["ci95"][0] > 0
    return {
        "fold": name,
        "n_tune": int(tune.size),
        "n_eval": int(evl.size),
        "bm25": {"ndcg10_tune": float(ctx["nd_bm25"][tune].mean()), "ndcg10_eval": float(ctx["nd_bm25"][evl].mean()), "recall100_eval": float(ctx["rc_bm25"][evl].mean())},
        "cosine": {"ndcg10_tune": float(ctx["nd_cos"][tune].mean()), "ndcg10_eval": float(ctx["nd_cos"][evl].mean()), "recall100_eval": float(ctx["rc_cos"][evl].mean())},
        "oracle_eval": {"best_channel": oracle_bc, "best_weight_z": oracle_bw, "headroom_best_channel": oracle_bc - zs_eval, "headroom_best_weight": oracle_bw - zs_eval},
        "families": families,
        "chosen": {"family": chosen, "rule": "highest ndcg10_tune among non-baseline families", **ch,
                   "bootstrap_vs_rrf_eval": roc.paired_bootstrap(per_query[chosen][evl] - ctx["rrf_nd"][evl]),
                   "passes_win_rule": bool(passes)},
        "best_family_by_eval_info_only": max(candidates, key=lambda k: families[k]["ndcg10_eval"]),
    }


# ----------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------


def protocol_text() -> dict:
    """Exact procedure text, stored in the output."""
    return {
        "inputs": "NFCorpus cache; queries with IDCG@10 = 0 dropped; BM25 k1=1.2 b=0.75 HMS tokenizer (roc.py); cosine = Q D^T on l2-normalised f32 rows cast to f64.",
        "pool": "BM25 top-200 (positive scores only) | cosine top-200; PRF adds | cos' top-200. Channels z-scored over the pool; fused = (1-w) z(BM25) + w z(dense); top-100 by score, ties to the lower document index.",
        "metrics": "nDCG@10 linear gain over judged grades (1, 2); recall@100 = judged-relevant in the fused top-100 / judged-relevant.",
        "folds": "primary: tune = odd 0-based query index in queries.jsonl order, eval = even; swapped: mirrored. Every parameter chosen by tune-fold mean nDCG@10; eval reported once per family.",
        "prf_dense": "seed = baseline (zsum, tune weight w0) fused top-m; q' = normalize(q + alpha mean(D[seed])); cos' over the corpus; second-stage weight w' from the 21-point grid; grid m x alpha x w' = 189.",
        "gates": "statistic computed from first-pass channels; tune-fold quantile edges (2,3,4 buckets; 2x2 or 3x3 for margin_2d); per-bucket weight = argmax tune mean nDCG@10 over its members; empty bucket -> global tune-best weight; margin_diff = ECDF_tune(cos_margin) - ECDF_tune(bm25_margin).",
        "centroid_rerank": "c_b = normalize(mean D[positive BM25 top-m]) or q when none; score = fused + beta z(cos(doc, c_b)) over the baseline pool; grid m x beta x w = 168.",
        "two_stage_prf": "baseline fused top-100 at w0; score = (1-lambda) z(fused) + lambda z(cos'); top-10 by score; recall@100 equals zsum's; grid (m, alpha) x lambda = 27.",
        "prf_gate": "prf_dense tables (all m, alpha) with w' chosen per bucket by the gate; grid (m, alpha) x gate variants.",
        "selection": "chosen = non-baseline family with the highest tune-fold nDCG@10 (ties: first in insertion order).",
        "bootstrap": "paired percentile bootstrap of per-query (family - zsum) on the eval fold; 2000 resamples; numpy default_rng(0).",
        "oracle_gap": "(family_eval - zsum_eval) / (oracle_eval - zsum_eval); oracles: best channel per query, best z-sum weight per query; null if the oracle does not exceed zsum.",
        "verdict": "WIN iff chosen family on the primary fold has delta_vs_zsum_eval >= 0.01 and the bootstrap CI95 lower bound > 0; swapped fold reported as a replication check.",
        "cost": "extra dim-384 dot products per query beyond the baseline (one corpus pass for cosine + BM25 postings).",
    }


def load_context(cache: Path) -> dict:
    """Read inputs, build BM25 and cosine channels, the baseline pool and tables, first-pass statistics."""
    meta_path = cache / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read {meta_path}: {exc}") from exc
    n_corpus, n_queries, dim = int(meta["n_corpus"]), int(meta["n_queries"]), int(meta["dim"])
    doc_ids, doc_texts = roc.read_jsonl(cache / "corpus.jsonl")
    q_ids, q_texts = roc.read_jsonl(cache / "queries.jsonl")
    if len(doc_ids) != n_corpus or len(q_ids) != n_queries:
        raise SystemExit(f"jsonl rows ({len(doc_ids)}, {len(q_ids)}) differ from meta.json ({n_corpus}, {n_queries})")
    D = roc.read_f32(cache / "corpus.f32", n_corpus, dim).astype(np.float64)
    Q = roc.read_f32(cache / "queries.f32", n_queries, dim).astype(np.float64)
    did_index = {d: i for i, d in enumerate(doc_ids)}
    qid_index = {q: i for i, q in enumerate(q_ids)}
    if len(did_index) != n_corpus or len(qid_index) != n_queries:
        raise SystemExit("duplicate ids in corpus.jsonl or queries.jsonl")
    R_all, qrel_counts = roc.read_qrels(cache / "qrels.tsv", qid_index, did_index)
    idcg_all = roc.ideal_dcg(R_all)
    keep = idcg_all > 0
    R, Qk, idcg = R_all[keep], Q[keep], idcg_all[keep]
    q_texts_k = [t for t, k in zip(q_texts, keep) if k]
    lab = {"R": R, "idcg": idcg, "rel": R > 0}
    n_q, n_d = R.shape
    t_load = time.perf_counter()

    index = roc.build_bm25(doc_texts)
    S_b = roc.bm25_scores(index, q_texts_k)
    S_c = Qk @ D.T
    order_b, order_c = roc.argsort_desc(S_b), roc.argsort_desc(S_c)
    mask0 = roc.union_mask(order_b, order_c, n_d, CAND_K, S_b)
    pool0 = pool_from_mask(mask0)
    Bz0, Cz0 = z_channels(S_b, S_c, pool0)
    T0, T0_top, T0_score = sweep(Bz0, Cz0, pool0, lab, keep_top=True)
    rrf_b, rrf_c = roc.gather(roc.rrf_norm(order_b), pool0), roc.gather(roc.rrf_norm(order_c), pool0)
    rrf_docs, _ = top_docs(fused_scores(rrf_b, rrf_c, 0.5, pool0), pool0, RECALL_K)
    t_channels = time.perf_counter()

    ctx = {
        "D": D, "Qk": Qk, "S_b": S_b, "S_c": S_c, "order_b": order_b, "order_c": order_c, "lab": lab,
        "mask0": mask0, "pool0": pool0, "pool0_width": float(pool0.valid.sum(axis=1).mean()), "Bz0": Bz0, "Cz0": Cz0,
        "T0": T0, "T0_top": T0_top, "T0_score": T0_score,
        "nd_bm25": roc.ndcg10(order_b, R, idcg, S_b), "nd_cos": roc.ndcg10(order_c, R, idcg),
        "rc_bm25": roc.recall_at(order_b, lab["rel"], RECALL_K, S_b), "rc_cos": roc.recall_at(order_c, lab["rel"], RECALL_K),
        "rrf_nd": roc.ndcg10(rrf_docs, R, idcg), "rrf_rc": roc.recall_at(rrf_docs, lab["rel"], RECALL_K),
        "stats": first_pass_stats(S_b, S_c, order_b, order_c, n_d),
        "data": {
            "n_corpus": n_corpus, "n_queries_file": n_queries, "n_queries_scored": int(n_q), "n_queries_dropped_no_relevant": int((~keep).sum()),
            "dim": dim, "qrels": qrel_counts, "bm25_vocab": len(index.vocab), "bm25_postings": int(index.post_doc.size),
            "pool_width_mean": float(pool0.valid.sum(axis=1).mean()), "pool_width_max": int(pool0.idx.shape[1]),
            "corpus_row_norm_range": [float(np.linalg.norm(D, axis=1).min()), float(np.linalg.norm(D, axis=1).max())],
            "input_sha256": {name: roc.sha256_file(cache / name) for name in INPUT_FILES},
            "reference": "benchmarks/research/roc/out/roc_v1.json (round 1: zsum w=0.7 eval 0.3459, oracles 0.3774 / 0.3970 on all queries)",
        },
        "timing": {"load_s": t_load, "channels_s": t_channels},
    }
    ctx["cent_tables"] = centroid_tables(ctx)
    ctx["timing"]["centroid_s"] = time.perf_counter()
    return ctx


def run(cache: Path, out_path: Path) -> dict:
    """Execute both folds and write the result document."""
    t0 = time.perf_counter()
    ctx = load_context(cache)
    n_q = ctx["lab"]["R"].shape[0]
    idx = np.arange(n_q)
    odd, even = idx[idx % 2 == 1], idx[idx % 2 == 0]
    prf_cache: dict = {}
    folds = {
        "primary": evaluate_fold("primary: tune=odd, eval=even", odd, even, ctx, prf_cache),
        "swapped": evaluate_fold("swapped: tune=even, eval=odd", even, odd, ctx, prf_cache),
    }
    t_folds = time.perf_counter()

    p = folds["primary"]
    s = folds["swapped"]
    chosen = p["chosen"]
    same_family = chosen["family"] == s["chosen"]["family"]
    verdict = "WIN" if chosen["passes_win_rule"] else "NO_WIN"
    doc = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": protocol_text(),
        "constants": {
            "ndcg_k": NDCG_K, "recall_k": RECALL_K, "candidate_k": CAND_K, "rrf_k": RRF_K, "weights": list(WEIGHTS), "agreement_top": AGREE_TOP,
            "prf_m": list(PRF_M), "prf_alpha": list(PRF_ALPHA), "buckets_1d": list(BUCKETS_1D), "buckets_2d": list(BUCKETS_2D),
            "centroid_m": list(CENTROID_M), "centroid_beta": list(CENTROID_BETA), "stage2_lambda": list(STAGE2_LAMBDA), "stage2_top": STAGE2_TOP,
            "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED}, "win_margin": WIN_MARGIN, "tie_eps": repr(roc.TIE_EPS),
        },
        "data": ctx["data"],
        "all_queries": {
            "bm25_ndcg10": float(ctx["nd_bm25"].mean()), "cosine_ndcg10": float(ctx["nd_cos"].mean()),
            "bm25_recall100": float(ctx["rc_bm25"].mean()), "cosine_recall100": float(ctx["rc_cos"].mean()),
            "pool_recall100_union_top100": float(((roc.union_mask(ctx["order_b"], ctx["order_c"], ctx["D"].shape[0], RECALL_K, ctx["S_b"]) & ctx["lab"]["rel"]).sum(axis=1) / ctx["lab"]["rel"].sum(axis=1)).mean()),
            "zsum_curve_ndcg10": ctx["T0"].nd.mean(axis=1).tolist(),
            "zsum_curve_recall100": ctx["T0"].rc.mean(axis=1).tolist(),
            "rrf_k60_ndcg10": float(ctx["rrf_nd"].mean()),
            "oracle_best_channel": float(np.maximum(ctx["nd_bm25"], ctx["nd_cos"]).mean()),
            "oracle_best_weight_z": float(ctx["T0"].nd.max(axis=0).mean()),
            "stat_quantiles_10_50_90": {k: np.quantile(v, [0.1, 0.5, 0.9]).tolist() for k, v in ctx["stats"].items()},
        },
        "folds": folds,
        "verdict": {
            "result": verdict,
            "rule": f"chosen family (by primary tune fold) beats zsum by >= {WIN_MARGIN} on the primary eval fold with bootstrap CI95 excluding 0",
            "chosen_family": chosen["family"],
            "delta_vs_zsum_eval": chosen["delta_vs_zsum_eval"],
            "ci95": chosen["bootstrap_vs_zsum_eval"]["ci95"],
            "swapped_fold": {"chosen_family": s["chosen"]["family"], "same_family": same_family, "delta_vs_zsum_eval": s["chosen"]["delta_vs_zsum_eval"],
                             "ci95": s["chosen"]["bootstrap_vs_zsum_eval"]["ci95"], "passes_win_rule": s["chosen"]["passes_win_rule"]},
            "cross_fit_delta_vs_zsum": (chosen["delta_vs_zsum_eval"] * even.size + s["chosen"]["delta_vs_zsum_eval"] * odd.size) / n_q,
        },
        "timing_s": {
            "load": ctx["timing"]["load_s"] - t0, "channels_and_baseline": ctx["timing"]["channels_s"] - ctx["timing"]["load_s"],
            "centroid_tables": ctx["timing"]["centroid_s"] - ctx["timing"]["channels_s"], "folds": t_folds - ctx["timing"]["centroid_s"],
            "total": time.perf_counter() - t0,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path.write_text(json.dumps(roc.rounded(doc), indent=1) + "\n", encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot write {out_path}: {exc}") from exc
    return doc


def fmt(x, nd: int = 4) -> str:
    """Fixed-width float or '-' for None."""
    return "-" if x is None else f"{x:.{nd}f}"


def ci(b: dict) -> str:
    """Bootstrap CI95 as a compact signed pair."""
    return f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]"


def print_summary(doc: dict, out_path: Path) -> None:
    """One-screen summary ending in the VERDICT line."""
    print(f"relevance  {doc['schema']}  queries={doc['data']['n_queries_scored']}  docs={doc['data']['n_corpus']}  total={doc['timing_s']['total']:.1f}s")
    for fname in ("primary", "swapped"):
        f = doc["folds"][fname]
        o = f["oracle_eval"]
        print(f"\n[{f['fold']}]  n_tune={f['n_tune']} n_eval={f['n_eval']}  BM25 eval {fmt(f['bm25']['ndcg10_eval'])}  cos {fmt(f['cosine']['ndcg10_eval'])}  "
              f"oracle ch {fmt(o['best_channel'])} / w {fmt(o['best_weight_z'])}")
        print(f"  {'family':<20} {'tune':>7} {'eval':>7} {'R@100':>7} {'d_zsum':>7} {'ci95_lo':>8} {'ci95_hi':>8} {'gap_w':>6} {'gap_ch':>6} {'dots':>5} {'cfg':>4}")
        for k, v in f["families"].items():
            b = v["bootstrap_vs_zsum_eval"]
            print(f"  {k:<20} {fmt(v['ndcg10_tune']):>7} {fmt(v['ndcg10_eval']):>7} {fmt(v['recall100_eval']):>7} {v['delta_vs_zsum_eval']:>+7.4f} "
                  f"{b['ci95'][0]:>+8.4f} {b['ci95'][1]:>+8.4f} {fmt(v['gap_closed_best_weight_eval'], 2):>6} {fmt(v['gap_closed_best_channel_eval'], 2):>6} "
                  f"{v['cost']['extra_dots']:>5} {v['n_configs']:>4}")
        c = f["chosen"]
        print(f"  chosen by tune: {c['family']}  eval {fmt(c['ndcg10_eval'])}  d_zsum {c['delta_vs_zsum_eval']:+.4f} ci {ci(c['bootstrap_vs_zsum_eval'])}  "
              f"vs rrf {c['bootstrap_vs_rrf_eval']['mean']:+.4f} ci {ci(c['bootstrap_vs_rrf_eval'])}  (best by eval, info only: {f['best_family_by_eval_info_only']})")
    v = doc["verdict"]
    p = doc["folds"]["primary"]["chosen"]
    print(f"\nwritten: {out_path}")
    print(f"VERDICT: {v['result']}  (chosen {v['chosen_family']}: delta {v['delta_vs_zsum_eval']:+.4f} >= {WIN_MARGIN}: {'yes' if v['delta_vs_zsum_eval'] >= WIN_MARGIN else 'no'}; "
          f"ci95 {ci(p['bootstrap_vs_zsum_eval'])} excludes 0: {'yes' if v['ci95'][0] > 0 else 'no'}; swapped fold {v['swapped_fold']['chosen_family']} "
          f"delta {v['swapped_fold']['delta_vs_zsum_eval']:+.4f} passes: {'yes' if v['swapped_fold']['passes_win_rule'] else 'no'})")


def main() -> None:
    """CLI entry: --cache <dir> --out <json>."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "hms-bench" / "nfcorpus", help="NFCorpus cache directory")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out" / "relevance_v1.json", help="result JSON path")
    args = ap.parse_args()
    for name in INPUT_FILES:
        if not (args.cache / name).is_file():
            raise SystemExit(f"missing input: {args.cache / name}")
    doc = run(args.cache, args.out)
    print_summary(doc, args.out)


if __name__ == "__main__":
    main()
