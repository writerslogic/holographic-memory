# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Dense-channel relevance gate on BEIR NFCorpus: may a MiniLM-L6 cosine channel vote next to
BM25, and by how much?

Mechanism
---------
Two exhaustive scorers over the 3633-document corpus for every query: BM25 (k1 1.2, b 0.75, the
WARM tokenizer: lowercase, split on runs outside [0-9A-Za-z\\u00C0-\\uFFFF], no stemming, no stop
list, each unique query term counted once, idf ln(1 + (N - df + 0.5) / (df + 0.5))) and cosine of
the cached l2-normalized MiniLM-L6 vectors. Four measurements follow, numpy only, deterministic:

1. nDCG@10 of each channel alone and of fusions ranked inside the union of the two per-channel
   top-200 lists: min-max linear (dense-weight sweep), weighted reciprocal-rank fusion (k = 60,
   the equal-weight point is standard RRF), z-score sum (sweep), and per-query gated fusion whose
   dense weight is a function of the dispersion of the query's top-50 BM25 scores: a label-free
   "rank-calibrated information gain" gate (dense weight = 1 - midrank ECDF of the KL-from-uniform
   statistic, ECDF taken over the tune split without labels) and a sigmoid gate tuned by grid
   search. Parameters are chosen on the tune split and reported on the disjoint eval split.
2. Judged-pair ROC: Mann-Whitney AUC of cosine for ordering (relevant, non-relevant) pairs that
   fall in the same BM25 score band (four rank-quantile bands of the positive-score BM25 top-200)
   and for pairs outside the BM25 top-50. BM25's own within-band AUC is the control.
3. Candidate recall@100 / @200 of BM25, cosine, and their union (top-K of each, up to 2K docs).
4. Oracle: mean over queries of the better channel's nDCG@10 (upper bound on any one-channel-per-
   query gate) and the best static min-max weight per query (upper bound on any weight gate).

Data layout
-----------
<cache>/corpus.jsonl ({"id","text"}; the title is already joined to the text by a newline),
<cache>/queries.jsonl ({"id","text"}), <cache>/qrels.tsv (qid, did, grade; tab separated, no
header; grades 1 and 2 only, so non-relevant == unjudged), <cache>/corpus.f32 and queries.f32
(row-major float32, rows in jsonl order; byte sizes are checked against meta.json).
R[q, d] is the grade (int16, 0 = unjudged). Score matrices are dense (n_queries, n_docs) float64.

Invariants
----------
* One tie-break everywhere: score - 1e-12 * doc_index, so equal scores order by document index
  and every ranking is reproducible bit for bit on one numpy build.
* nDCG@10: linear gain, log2(rank + 1) discount, ideal list from all judged documents
  (trec_eval ndcg_cut_10). Queries without a positive judgment are dropped and counted.
* Fusions never rank a document outside the candidate union (its fused score is -inf).
* Gate statistics read no labels. The tune split enters the principled gate only through a
  label-free ECDF; labels on the tune split are used only to pick grid points.
* BEIR train qrels exist in the download but there are no embeddings for train queries, so the
  disjoint split is by 0-based query index parity: odd -> tune, even -> eval (primary fold). The
  swapped fold and the count-weighted cross-fit mean are also reported.
* The cache holds 323 queries / 12334 judgments, which is the BEIR *test* split by count; the
  reference numbers quoted in the output were measured on dev (324 queries) and are not directly
  comparable. The gate verdict uses only numbers measured here.

Run: uv run --no-project --with numpy python benchmarks/research/roc/roc.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "warm.research.roc/1"
K1 = 1.2
B = 0.75
TOKEN_SPLIT = re.compile(r"[^0-9A-Za-zÀ-￿]+")
NDCG_K = 10
CAND_K = 200
RRF_K = 60
GATE_TOP = 50
BAND_TOP = 200
N_BANDS = 4
OUTSIDE_TOP = 50
RECALL_KS = (100, 200)
WEIGHTS = tuple(round(0.05 * i, 2) for i in range(21))
SIGMOID_W_LO = (0.0, 0.15, 0.3, 0.45)
SIGMOID_W_HI = (0.55, 0.7, 0.85, 1.0)
SIGMOID_C_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
SIGMOID_TAU_SCALES = (0.1, 0.3, 1.0)
GATE_STATS = ("ig_kl", "ig_cv", "z_margin")
BOOT_N = 2000
BOOT_SEED = 0
AUC_MIN = 0.55
GAIN_MIN = 0.02
TIE_EPS = 1e-12
DISCOUNT = 1.0 / np.log2(np.arange(2, NDCG_K + 2))
REFERENCE_DEV = {"bm25_warm": 0.2916, "minilm_l6": 0.3024, "gte_small": 0.3398}


# ----------------------------------------------------------------------------------------------
# input
# ----------------------------------------------------------------------------------------------


def read_jsonl(path: Path) -> tuple[list[str], list[str]]:
    """Return (ids, texts) of a jsonl file with "id" and "text" keys, in file order."""
    ids: list[str] = []
    texts: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "id" not in row or "text" not in row:
                raise SystemExit(f"{path}:{n}: missing id/text keys")
            ids.append(str(row["id"]))
            texts.append(str(row["text"]))
    return ids, texts


def read_f32(path: Path, rows: int, dim: int) -> np.ndarray:
    """Read a row-major float32 matrix after checking its byte size against (rows, dim)."""
    expect = rows * dim * 4
    size = path.stat().st_size
    if size != expect:
        raise SystemExit(f"{path}: {size} bytes, expected {expect} = {rows} x {dim} x 4")
    return np.fromfile(path, dtype=np.float32).reshape(rows, dim)


def read_qrels(path: Path, qid_index: dict[str, int], did_index: dict[str, int]) -> tuple[np.ndarray, dict]:
    """Return R[q, d] = grade (int16, max over duplicates) and counts of kept / dropped judgments."""
    R = np.zeros((len(qid_index), len(did_index)), dtype=np.int16)
    kept = dropped_q = dropped_d = 0
    grades: Counter[int] = Counter()
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise SystemExit(f"{path}:{n}: expected 3 tab-separated fields, got {len(parts)}")
            qid, did, grade = parts
            if qid == "query-id":
                continue
            qi = qid_index.get(qid)
            di = did_index.get(did)
            if qi is None:
                dropped_q += 1
                continue
            if di is None:
                dropped_d += 1
                continue
            g = int(grade)
            R[qi, di] = max(int(R[qi, di]), g)
            grades[g] += 1
            kept += 1
    counts = {
        "judgments_kept": kept,
        "dropped_unknown_query": dropped_q,
        "dropped_unknown_doc": dropped_d,
        "grade_counts": {str(k): v for k, v in sorted(grades.items())},
    }
    return R, counts


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
# BM25
# ----------------------------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """WARM tokenizer: lowercase, split on runs outside [0-9A-Za-z\\u00C0-\\uFFFF]; no stemming."""
    return [t for t in TOKEN_SPLIT.split(text.lower()) if t]


@dataclass(frozen=True)
class Bm25Index:
    """CSR postings with precomputed BM25 term-document weights.

    Term t's postings are post_doc[indptr[t]:indptr[t+1]] (document ids, ascending) with weights
    post_w at the same offsets, already multiplied by idf and tf-saturated.
    """

    vocab: dict[str, int]
    indptr: np.ndarray
    post_doc: np.ndarray
    post_w: np.ndarray
    n_docs: int
    avgdl: float


def build_bm25(texts: list[str], k1: float = K1, b: float = B) -> Bm25Index:
    """Index documents: idf = ln(1 + (N - df + 0.5)/(df + 0.5)), weight = idf * tf (k1+1) / (tf + k1 (1 - b + b dl/avgdl))."""
    vocab: dict[str, int] = {}
    term_docs: list[list[int]] = []
    term_tfs: list[list[int]] = []
    dl = np.zeros(len(texts), dtype=np.float64)
    for di, text in enumerate(texts):
        counts = Counter(tokenize(text))
        dl[di] = float(sum(counts.values()))
        for term, tf in counts.items():
            tid = vocab.get(term)
            if tid is None:
                tid = len(vocab)
                vocab[term] = tid
                term_docs.append([])
                term_tfs.append([])
            term_docs[tid].append(di)
            term_tfs[tid].append(tf)
    n = len(texts)
    if n == 0 or dl.sum() <= 0:
        raise SystemExit("empty corpus")
    avgdl = float(dl.mean())
    df = np.array([len(d) for d in term_docs], dtype=np.int64)
    idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
    indptr = np.zeros(len(term_docs) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(df)
    total = int(indptr[-1])
    post_doc = np.fromiter((d for docs in term_docs for d in docs), dtype=np.int32, count=total)
    post_tf = np.fromiter((t for tfs in term_tfs for t in tfs), dtype=np.float64, count=total)
    norm = k1 * (1.0 - b + b * dl[post_doc] / avgdl)
    post_w = np.repeat(idf, df) * post_tf * (k1 + 1.0) / (post_tf + norm)
    return Bm25Index(vocab, indptr, post_doc, post_w, n, avgdl)


def bm25_scores(index: Bm25Index, queries: list[str]) -> np.ndarray:
    """Dense (n_queries, n_docs) BM25 matrix; each unique query term contributes once, in sorted order."""
    S = np.zeros((len(queries), index.n_docs), dtype=np.float64)
    for qi, q in enumerate(queries):
        row = S[qi]
        for term in sorted(set(tokenize(q))):
            tid = index.vocab.get(term)
            if tid is None:
                continue
            a, z = int(index.indptr[tid]), int(index.indptr[tid + 1])
            row[index.post_doc[a:z]] += index.post_w[a:z]
    return S


# ----------------------------------------------------------------------------------------------
# rankings and metrics
# ----------------------------------------------------------------------------------------------


def tie_broken(S: np.ndarray) -> np.ndarray:
    """Scores with the deterministic tie-break: lower document index wins among equal scores."""
    return S - TIE_EPS * np.arange(S.shape[1], dtype=np.float64)


def argsort_desc(S: np.ndarray) -> np.ndarray:
    """Full descending ordering of every row (tie-broken)."""
    return np.argsort(-tie_broken(S), axis=1, kind="stable")


def topk(S: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, in descending order (tie-broken)."""
    T = -tie_broken(S)
    part = np.argpartition(T, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(T, part, axis=1)
    return np.take_along_axis(part, np.argsort(vals, axis=1, kind="stable"), axis=1)


def ranks_from_order(order: np.ndarray) -> np.ndarray:
    """1-based rank of every document given each row's full ordering."""
    n_q, n = order.shape
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, n + 1), (n_q, n)), axis=1)
    return ranks


def ideal_dcg(R: np.ndarray) -> np.ndarray:
    """IDCG@10 per query from all judged grades (linear gain)."""
    top = -np.sort(-R.astype(np.float64), axis=1)[:, :NDCG_K]
    return top @ DISCOUNT


def ndcg10(top_idx: np.ndarray, R: np.ndarray, idcg: np.ndarray, S: np.ndarray | None = None) -> np.ndarray:
    """nDCG@10 per query for ranked index rows (at least 10 columns); idcg must be > 0.

    With S (full score matrix), documents whose score is not positive count as unretrieved
    (gain 0): a BM25 run never returns a document that matches no query term.
    """
    cols = top_idx[:, :NDCG_K]
    gains = np.take_along_axis(R, cols, axis=1).astype(np.float64)
    if S is not None:
        gains *= np.take_along_axis(S, cols, axis=1) > 0
    return (gains @ DISCOUNT) / idcg


def topk_mask(order: np.ndarray, n_docs: int, k: int, S: np.ndarray | None = None) -> np.ndarray:
    """Boolean (n_q, n_docs) membership in the top-k; with S, only positively scored documents."""
    m = np.zeros((order.shape[0], n_docs), dtype=bool)
    np.put_along_axis(m, order[:, :k], True, axis=1)
    if S is not None:
        m &= S > 0
    return m


def recall_at(top_idx: np.ndarray, rel: np.ndarray, k: int, S: np.ndarray | None = None) -> np.ndarray:
    """Fraction of relevant documents inside the first k ranked columns, per query (S as in ndcg10)."""
    hits = topk_mask(top_idx, rel.shape[1], k, S) & rel
    return hits.sum(axis=1) / rel.sum(axis=1)


def union_mask(order_a: np.ndarray, order_b: np.ndarray, n_docs: int, k: int, S_a: np.ndarray | None = None) -> np.ndarray:
    """Boolean (n_q, n_docs) membership in top-k(a) union top-k(b); S_a drops unscored a-entries."""
    return topk_mask(order_a, n_docs, k, S_a) | topk_mask(order_b, n_docs, k)


@dataclass(frozen=True)
class Candidates:
    """Compact per-query candidate union.

    idx[q, j] is a document index, ascending within each row so that the compact-column tie-break
    equals the document-index tie-break; valid[q, j] marks real entries (rows are padded with
    index 0 and valid = False). Every row holds at least k valid entries.
    """

    idx: np.ndarray
    valid: np.ndarray


def build_candidates(order_a: np.ndarray, order_b: np.ndarray, n_docs: int, k: int, S_a: np.ndarray | None = None) -> Candidates:
    """Union of the two per-channel top-k lists in compact form (S_a drops unscored a-entries)."""
    mask = union_mask(order_a, order_b, n_docs, k, S_a)
    width = int(mask.sum(axis=1).max())
    idx = np.zeros((mask.shape[0], width), dtype=np.int64)
    valid = np.zeros((mask.shape[0], width), dtype=bool)
    for q in range(mask.shape[0]):
        docs = np.flatnonzero(mask[q])
        idx[q, : docs.size] = docs
        valid[q, : docs.size] = True
    return Candidates(idx, valid)


def gather(S: np.ndarray, cand: Candidates) -> np.ndarray:
    """Scores of the candidates, (n_q, width); padded columns carry the score of document 0."""
    return np.take_along_axis(S, cand.idx, axis=1)


def minmax_norm(S: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-row min-max over the masked candidates (constant rows map to 0); 0 outside the mask."""
    M = np.where(mask, S, np.nan)
    lo = np.nanmin(M, axis=1, keepdims=True)
    hi = np.nanmax(M, axis=1, keepdims=True)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    return np.where(mask, (S - lo) / span, 0.0)


def z_norm(S: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-row z-score over the masked candidates (zero-variance rows map to 0); 0 outside the mask."""
    M = np.where(mask, S, np.nan)
    mu = np.nanmean(M, axis=1, keepdims=True)
    sd = np.nanstd(M, axis=1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return np.where(mask, (S - mu) / sd, 0.0)


def rrf_norm(order: np.ndarray, k: int = RRF_K) -> np.ndarray:
    """Reciprocal-rank score 1 / (k + rank) over the full ordering."""
    return 1.0 / (k + ranks_from_order(order).astype(np.float64))


def fuse(Bn: np.ndarray, Cn: np.ndarray, w, mask: np.ndarray) -> np.ndarray:
    """(1 - w) * BM25 + w * cosine on the candidates, -inf elsewhere; w is a scalar or (n_q, 1)."""
    return np.where(mask, (1.0 - w) * Bn + w * Cn, -np.inf)


def fused_top_docs(Bc: np.ndarray, Cc: np.ndarray, w, cand: Candidates, k: int = NDCG_K) -> np.ndarray:
    """Document indices of the fused top-k over compact candidate channels (n_q, width)."""
    return np.take_along_axis(cand.idx, topk(fuse(Bc, Cc, w, cand.valid), k), axis=1)


# ----------------------------------------------------------------------------------------------
# gate statistics, ranks, AUC
# ----------------------------------------------------------------------------------------------


def gate_statistics(S_b: np.ndarray, order_b: np.ndarray, top: int = GATE_TOP) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Label-free dispersion of each query's positive BM25 scores among its top-`top`.

    ig_kl: KL(softmax(scores) || uniform) / ln(n) in [0, 1] (BM25 scores read as log-odds);
    ig_cv: std / mean of the scores (normalized-query-commitment style spread);
    z_margin: (s1 - mean(s2..n)) / std(s2..n), the top hit's z-score margin.
    Fewer than two positive scores give 0 for every statistic.
    """
    head = np.take_along_axis(S_b, order_b[:, :top], axis=1)
    out = {k: np.zeros(S_b.shape[0], dtype=np.float64) for k in GATE_STATS}
    n_pos = np.zeros(S_b.shape[0], dtype=np.int64)
    for qi in range(S_b.shape[0]):
        s = head[qi][head[qi] > 0]
        n = int(s.size)
        n_pos[qi] = n
        if n < 2:
            continue
        p = np.exp(s - s.max())
        p /= p.sum()
        entropy = -float(np.sum(p * np.log(np.maximum(p, 1e-300))))
        out["ig_kl"][qi] = (math.log(n) - entropy) / math.log(n)
        mean = float(s.mean())
        out["ig_cv"][qi] = float(s.std()) / mean if mean > 0 else 0.0
        if n >= 3:
            tail = s[1:]
            sd = float(tail.std())
            out["z_margin"][qi] = (float(s[0]) - float(tail.mean())) / sd if sd > 0 else 0.0
    return out, n_pos


def ecdf_midrank(reference: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Empirical CDF of `reference` evaluated at x with midrank treatment of ties, in [0, 1]."""
    ref = np.sort(reference)
    lo = np.searchsorted(ref, x, side="left")
    hi = np.searchsorted(ref, x, side="right")
    return (lo + hi) / (2.0 * ref.size)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-free logistic function."""
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def gate_weights_sigmoid(g: np.ndarray, c: float, tau: float, w_lo: float, w_hi: float) -> np.ndarray:
    """Dense weight decreasing in the BM25 statistic g: w_lo + (w_hi - w_lo) * sigmoid(-(g - c)/tau)."""
    return w_lo + (w_hi - w_lo) * sigmoid(-(g - c) / tau)


def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given their average rank."""
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    first = np.cumsum(counts) - counts + 1.0
    return (first + (counts - 1) / 2.0)[inverse.reshape(-1)]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (average ranks); nan when either side is constant."""
    ra, rb = average_ranks(a), average_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def auc_pairs(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Mann-Whitney AUC of scores x for positives y (bool) over negatives; (nan, 0) without pairs."""
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan"), 0
    r = average_ranks(x)
    auc = (float(r[y].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc, n_pos * n_neg


def summarize_cells(cos_aucs: list[float], bm25_aucs: list[float], pairs: list[int]) -> dict:
    """Pair-pooled and cell-macro AUCs for a set of (query, band) cells."""
    if not pairs:
        return {"n_cells": 0, "n_pairs": 0, "cos_pooled_auc": None, "cos_macro_auc": None, "bm25_pooled_auc": None, "bm25_macro_auc": None}
    p = np.array(pairs, dtype=np.float64)
    c = np.array(cos_aucs, dtype=np.float64)
    b = np.array(bm25_aucs, dtype=np.float64)
    return {
        "n_cells": len(pairs),
        "n_pairs": int(p.sum()),
        "cos_pooled_auc": float((c * p).sum() / p.sum()),
        "cos_macro_auc": float(c.mean()),
        "bm25_pooled_auc": float((b * p).sum() / p.sum()),
        "bm25_macro_auc": float(b.mean()),
    }


def judged_pair_roc(S_b: np.ndarray, S_c: np.ndarray, order_b: np.ndarray, rel: np.ndarray) -> dict:
    """AUC of cosine (and BM25 as control) over (relevant, non-relevant) pairs inside BM25 bands.

    Bands: the query's BM25 top-200 restricted to positive scores, cut into four equal-count rank
    quantiles (band1 = highest scores). Also: pairs among documents outside the BM25 top-50, and
    pairs among all documents. Pooled = pair-weighted, macro = unweighted mean over cells.
    """
    n_q = S_b.shape[0]
    cells: dict[str, dict[str, list]] = {f"band{i + 1}": {"cos": [], "bm25": [], "pairs": []} for i in range(N_BANDS)}
    cells["outside_top50"] = {"cos": [], "bm25": [], "pairs": []}
    cells["all_docs"] = {"cos": [], "bm25": [], "pairs": []}
    all_bands: dict[str, list] = {"cos": [], "bm25": [], "pairs": []}
    per_query_within: list[float] = []
    for qi in range(n_q):
        ob = order_b[qi]
        head = ob[:BAND_TOP]
        head = head[S_b[qi, head] > 0]
        q_conc = 0.0
        q_pairs = 0
        for bi, idx in enumerate(np.array_split(np.arange(head.size), N_BANDS)):
            docs = head[idx]
            if docs.size < 2:
                continue
            y = rel[qi, docs]
            a_c, n = auc_pairs(S_c[qi, docs], y)
            if n == 0:
                continue
            a_b, _ = auc_pairs(S_b[qi, docs], y)
            for store in (cells[f"band{bi + 1}"], all_bands):
                store["cos"].append(a_c)
                store["bm25"].append(a_b)
                store["pairs"].append(n)
            q_conc += a_c * n
            q_pairs += n
        if q_pairs > 0:
            per_query_within.append(q_conc / q_pairs)
        tail = ob[OUTSIDE_TOP:]
        a_c, n = auc_pairs(S_c[qi, tail], rel[qi, tail])
        if n > 0:
            a_b, _ = auc_pairs(S_b[qi, tail], rel[qi, tail])
            cells["outside_top50"]["cos"].append(a_c)
            cells["outside_top50"]["bm25"].append(a_b)
            cells["outside_top50"]["pairs"].append(n)
        a_c, n = auc_pairs(S_c[qi], rel[qi])
        if n > 0:
            a_b, _ = auc_pairs(S_b[qi], rel[qi])
            cells["all_docs"]["cos"].append(a_c)
            cells["all_docs"]["bm25"].append(a_b)
            cells["all_docs"]["pairs"].append(n)
    result = {name: summarize_cells(c["cos"], c["bm25"], c["pairs"]) for name, c in cells.items()}
    result["all_bands"] = summarize_cells(all_bands["cos"], all_bands["bm25"], all_bands["pairs"])
    result["all_bands"]["cos_query_macro_auc"] = float(np.mean(per_query_within)) if per_query_within else None
    result["all_bands"]["n_queries_with_pairs"] = len(per_query_within)
    return result


# ----------------------------------------------------------------------------------------------
# fusion selection
# ----------------------------------------------------------------------------------------------


def sweep_static(Bc: np.ndarray, Cc: np.ndarray, cand: Candidates, R: np.ndarray, idcg: np.ndarray) -> np.ndarray:
    """Per-query nDCG@10 for every dense weight in WEIGHTS; shape (len(WEIGHTS), n_q)."""
    return np.stack([ndcg10(fused_top_docs(Bc, Cc, w, cand), R, idcg) for w in WEIGHTS])


def select_sigmoid_gate(stats: dict[str, np.ndarray], norms: dict[str, tuple[np.ndarray, np.ndarray]], cand: Candidates,
                        R: np.ndarray, idcg: np.ndarray, tune: np.ndarray, evl: np.ndarray) -> tuple[dict, int]:
    """Grid search (statistic, normalization, centre quantile, temperature, w_lo, w_hi) on the tune split.

    Returns the first configuration with the highest tune mean (params, tune, eval, per_query) and
    the number of configurations scored. Centres are quantiles of the statistic on the tune split
    and temperatures are multiples of its tune-split standard deviation.
    """
    best: dict | None = None
    n_configs = 0
    for stat in GATE_STATS:
        g = stats[stat]
        sd = float(g[tune].std())
        if sd == 0:
            continue
        centres = np.quantile(g[tune], SIGMOID_C_QUANTILES)
        for norm_name, (Bn, Cn) in norms.items():
            for qc, c in zip(SIGMOID_C_QUANTILES, centres):
                for ts in SIGMOID_TAU_SCALES:
                    tau = ts * sd
                    for w_lo in SIGMOID_W_LO:
                        for w_hi in SIGMOID_W_HI:
                            w = gate_weights_sigmoid(g, float(c), tau, w_lo, w_hi)
                            nd = ndcg10(fused_top_docs(Bn, Cn, w[:, None], cand), R, idcg)
                            n_configs += 1
                            t = float(nd[tune].mean())
                            if best is None or t > best["tune"]:
                                best = {
                                    "params": {"statistic": stat, "norm": norm_name, "centre_quantile": qc, "centre": float(c),
                                               "tau": tau, "tau_scale": ts, "w_lo": w_lo, "w_hi": w_hi,
                                               "mean_dense_weight": float(w.mean())},
                                    "tune": t,
                                    "eval": float(nd[evl].mean()),
                                    "per_query": nd,
                                }
    if best is None:
        raise SystemExit("every gate statistic is constant on the tune split")
    return best, n_configs


def evaluate_fold(name: str, tune: np.ndarray, evl: np.ndarray, ctx: dict) -> tuple[dict, np.ndarray]:
    """Select every fusion family on `tune`, report on `evl`; returns (summary, best per-query nDCG)."""
    R, idcg, cand, norms, stats = ctx["R"], ctx["idcg"], ctx["cand"], ctx["norms"], ctx["stats"]
    families: dict[str, dict] = {}
    per_query: dict[str, np.ndarray] = {}

    def add(fname: str, params: dict, nd: np.ndarray, n_configs: int) -> None:
        families[fname] = {"params": params, "ndcg10_tune": float(nd[tune].mean()), "ndcg10_eval": float(nd[evl].mean()), "n_configs": n_configs}
        per_query[fname] = nd

    for sname in ("minmax", "z", "rrf"):
        table = ctx["static"][sname]
        bi = int(np.argmax(table[:, tune].mean(axis=1)))
        add(sname, {"dense_weight": WEIGHTS[bi]}, table[bi], len(WEIGHTS))
    add("rrf_k60_equal", {"dense_weight": 0.5, "k": RRF_K}, ctx["static"]["rrf"][WEIGHTS.index(0.5)], 1)

    g = stats["ig_kl"]
    w = 1.0 - ecdf_midrank(g[tune], g)
    Bn, Cn = norms["minmax"]
    nd = ndcg10(fused_top_docs(Bn, Cn, w[:, None], cand), R, idcg)
    add("gate_rankcal_kl_minmax", {"statistic": "ig_kl", "norm": "minmax", "calibration": "1 - midrank ECDF over tune split (label-free)",
                                   "mean_dense_weight": float(w.mean())}, nd, 1)

    best_rc: tuple | None = None
    for stat in GATE_STATS:
        for norm_name, (Bn, Cn) in norms.items():
            w = 1.0 - ecdf_midrank(stats[stat][tune], stats[stat])
            nd = ndcg10(fused_top_docs(Bn, Cn, w[:, None], cand), R, idcg)
            t = float(nd[tune].mean())
            if best_rc is None or t > best_rc[0]:
                best_rc = (t, stat, norm_name, nd, float(w.mean()))
    assert best_rc is not None
    add("gate_rankcal_selected", {"statistic": best_rc[1], "norm": best_rc[2], "calibration": "1 - midrank ECDF over tune split",
                                  "mean_dense_weight": best_rc[4]}, best_rc[3], len(GATE_STATS) * len(norms))

    w = 1.0 - stats["ig_kl"]
    Bn, Cn = norms["minmax"]
    nd = ndcg10(fused_top_docs(Bn, Cn, w[:, None], cand), R, idcg)
    add("gate_kl_raw_minmax", {"statistic": "ig_kl", "norm": "minmax", "calibration": "none (w = 1 - ig_kl)", "mean_dense_weight": float(w.mean())}, nd, 1)

    best_sig, n_cfg = select_sigmoid_gate(stats, norms, cand, R, idcg, tune, evl)
    add("gate_sigmoid", best_sig["params"], best_sig["per_query"], n_cfg)

    bm25_eval = float(ctx["ndcg_bm25"][evl].mean())
    cos_eval = float(ctx["ndcg_cos"][evl].mean())
    best_name = max(families, key=lambda k: families[k]["ndcg10_tune"])
    static_names = ("minmax", "z", "rrf", "rrf_k60_equal")
    best_static = max(static_names, key=lambda k: families[k]["ndcg10_tune"])
    gated_names = tuple(k for k in families if k.startswith("gate_"))
    best_gated = max(gated_names, key=lambda k: families[k]["ndcg10_tune"])
    summary = {
        "fold": name,
        "n_tune": int(tune.size),
        "n_eval": int(evl.size),
        "bm25": {"ndcg10_tune": float(ctx["ndcg_bm25"][tune].mean()), "ndcg10_eval": bm25_eval},
        "cosine": {"ndcg10_tune": float(ctx["ndcg_cos"][tune].mean()), "ndcg10_eval": cos_eval},
        "families": families,
        "best": {"family": best_name, **families[best_name], "delta_vs_bm25_eval": families[best_name]["ndcg10_eval"] - bm25_eval,
                 "delta_vs_cosine_eval": families[best_name]["ndcg10_eval"] - cos_eval},
        "best_static": {"family": best_static, **families[best_static], "delta_vs_bm25_eval": families[best_static]["ndcg10_eval"] - bm25_eval},
        "best_gated": {"family": best_gated, **families[best_gated], "delta_vs_bm25_eval": families[best_gated]["ndcg10_eval"] - bm25_eval,
                       "delta_vs_best_static_eval": families[best_gated]["ndcg10_eval"] - families[best_static]["ndcg10_eval"]},
    }
    return summary, per_query[best_name]


def paired_bootstrap(diff: np.ndarray, n: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """Percentile bootstrap of the mean paired difference (fixed seed)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n, diff.size))
    means = diff[idx].mean(axis=1)
    return {
        "mean": float(diff.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "p_mean_le_0": float((means <= 0).mean()),
        "n_resamples": n,
        "seed": seed,
        "n_queries": int(diff.size),
    }


# ----------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------


def rounded(obj):
    """JSON-ready copy: floats to 6 decimals, nan to null, numpy scalars/arrays to Python."""
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): rounded(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rounded(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return rounded(obj.tolist())
    return obj


def protocol_text() -> dict:
    """Exact protocol, written next to the numbers."""
    return {
        "bm25": f"k1={K1}, b={B}, idf=ln(1+(N-df+0.5)/(df+0.5)), tf*(k1+1)/(tf+k1*(1-b+b*dl/avgdl)); tokenizer: lowercase, split on runs outside [0-9A-Za-z\\u00C0-\\uFFFF], no stemming, no stop list; unique query terms; field = corpus.jsonl text (title already joined by newline)",
        "cosine": "float64 dot product of the cached l2-normalized MiniLM-L6 rows (corpus.f32 x queries.f32), row order = jsonl order (byte sizes checked)",
        "ndcg10": "linear gain, discount 1/log2(rank+1), ideal from all judged grades (trec_eval ndcg_cut_10); queries without a positive judgment dropped and counted; ties broken by lower doc index everywhere; a BM25 score of 0 counts as not retrieved (gain 0, not in top-K, not a candidate)",
        "candidates": f"fusions rank only the union of BM25 top-{CAND_K} (positive scores) and cosine top-{CAND_K} (true scores of both channels on every candidate; -inf elsewhere)",
        "minmax": "per query, each channel min-max normalized over the candidate union; fused = (1-w)*bm25 + w*cos; w in WEIGHTS",
        "z": "per query, each channel z-scored over the candidate union; fused = (1-w)*bm25 + w*cos; w in WEIGHTS",
        "rrf": f"fused = (1-w)/({RRF_K}+rank_bm25) + w/({RRF_K}+rank_cos), ranks over the full corpus; w=0.5 is standard RRF k={RRF_K}",
        "gate_statistics": f"on the positive BM25 scores among the query's top-{GATE_TOP}: ig_kl = KL(softmax(raw scores)||uniform)/ln(n); ig_cv = std/mean; z_margin = (s1-mean(s2..n))/std(s2..n); no labels",
        "gate_rankcal_kl_minmax": "principled, label-free: dense weight = 1 - midrank ECDF(ig_kl) with the ECDF over the tune split's ig_kl values; min-max base",
        "gate_rankcal_selected": "same calibration, statistic and base normalization chosen on the tune split (6 configs)",
        "gate_kl_raw_minmax": "dense weight = 1 - ig_kl, no calibration",
        "gate_sigmoid": "dense weight = w_lo + (w_hi-w_lo)*sigmoid(-(g-c)/tau); grid: statistic x {minmax,z} x centre quantile x tau scale x w_lo x w_hi, selected on the tune split",
        "split": "BEIR train qrels exist in the download but there are no train-query embeddings, so: 0-based query index parity in queries.jsonl order; primary fold tune=odd, eval=even; swapped fold reported; cross_fit = count-weighted mean of the two eval scores",
        "selection": "within each family the first configuration with the highest tune-split mean nDCG@10; 'best' = family with the highest tune mean, reported on eval",
        "roc": f"per query, BM25 top-{BAND_TOP} restricted to positive scores cut into {N_BANDS} equal-count rank quantiles (band1 highest); AUC = Mann-Whitney over (rel>0, rel==0) pairs inside the cell; pooled = pair-weighted across cells, macro = unweighted; outside_top50 = documents at BM25 rank > {OUTSIDE_TOP}; BM25's own AUC in the same cells is the control; all 323 queries (labels used for measurement only)",
        "recall": "recall@K = |relevant in top-K| / |relevant|, macro over queries; union = BM25 top-K union cosine top-K (up to 2K docs)",
        "oracle": "best_channel = mean over queries of max(nDCG bm25, nDCG cosine); best_minmax_weight = mean of per-query max over the min-max sweep; best_any_static = max over the three static sweeps",
        "bootstrap": f"paired percentile bootstrap of eval-split (best fusion - BM25) nDCG@10, {BOOT_N} resamples, numpy default_rng({BOOT_SEED})",
        "verdict": f"GATE_OPEN iff roc.all_bands.cos_pooled_auc >= {AUC_MIN} and primary-fold best fusion eval nDCG@10 >= BM25 eval nDCG@10 + {GAIN_MIN}; else GATE_CLOSED",
        "reference_points_dev": "measured elsewhere on the BEIR dev split (324 queries), not on this cache (test split by count); shown for orientation only",
    }


def run(cache: Path, out_path: Path) -> dict:
    """Execute every measurement and return the result document."""
    t0 = time.perf_counter()
    meta_path = cache / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read {meta_path}: {exc}") from exc
    n_corpus, n_queries, dim = int(meta["n_corpus"]), int(meta["n_queries"]), int(meta["dim"])

    doc_ids, doc_texts = read_jsonl(cache / "corpus.jsonl")
    q_ids, q_texts = read_jsonl(cache / "queries.jsonl")
    if len(doc_ids) != n_corpus or len(q_ids) != n_queries:
        raise SystemExit(f"jsonl rows ({len(doc_ids)}, {len(q_ids)}) differ from meta.json ({n_corpus}, {n_queries})")
    D = read_f32(cache / "corpus.f32", n_corpus, dim)
    Q = read_f32(cache / "queries.f32", n_queries, dim)
    d_norm = np.linalg.norm(D.astype(np.float64), axis=1)
    q_norm = np.linalg.norm(Q.astype(np.float64), axis=1)
    did_index = {d: i for i, d in enumerate(doc_ids)}
    qid_index = {q: i for i, q in enumerate(q_ids)}
    if len(did_index) != n_corpus or len(qid_index) != n_queries:
        raise SystemExit("duplicate ids in corpus.jsonl or queries.jsonl")
    R_all, qrel_counts = read_qrels(cache / "qrels.tsv", qid_index, did_index)

    idcg_all = ideal_dcg(R_all)
    keep = idcg_all > 0
    n_dropped = int((~keep).sum())
    R = R_all[keep]
    Qk = Q[keep]
    q_texts_k = [t for t, k in zip(q_texts, keep) if k]
    q_ids_k = [q for q, k in zip(q_ids, keep) if k]
    idcg = idcg_all[keep]
    rel = R > 0
    n_q, n_d = R.shape
    t_load = time.perf_counter()

    index = build_bm25(doc_texts)
    S_b = bm25_scores(index, q_texts_k)
    t_bm25 = time.perf_counter()
    S_c = Qk.astype(np.float64) @ D.astype(np.float64).T

    order_b = argsort_desc(S_b)
    order_c = argsort_desc(S_c)
    cand = build_candidates(order_b, order_c, n_d, CAND_K, S_b)
    Sb_c, Sc_c = gather(S_b, cand), gather(S_c, cand)
    norms = {
        "minmax": (minmax_norm(Sb_c, cand.valid), minmax_norm(Sc_c, cand.valid)),
        "z": (z_norm(Sb_c, cand.valid), z_norm(Sc_c, cand.valid)),
    }
    rrf_b, rrf_c = gather(rrf_norm(order_b), cand), gather(rrf_norm(order_c), cand)

    ndcg_bm25 = ndcg10(order_b, R, idcg, S_b)
    ndcg_cos = ndcg10(order_c, R, idcg)
    static = {
        "minmax": sweep_static(norms["minmax"][0], norms["minmax"][1], cand, R, idcg),
        "z": sweep_static(norms["z"][0], norms["z"][1], cand, R, idcg),
        "rrf": sweep_static(rrf_b, rrf_c, cand, R, idcg),
    }
    t_static = time.perf_counter()

    recall = {}
    for k in RECALL_KS:
        rb, rc = recall_at(order_b, rel, k, S_b), recall_at(order_c, rel, k)
        um = union_mask(order_b, order_c, n_d, k, S_b)
        ru = (um & rel).sum(axis=1) / rel.sum(axis=1)
        recall[f"@{k}"] = {"bm25": float(rb.mean()), "cosine": float(rc.mean()), "union": float(ru.mean()),
                           "union_gain_over_bm25": float(ru.mean() - rb.mean()), "union_candidates_max": 2 * k}

    oracle_channel = np.maximum(ndcg_bm25, ndcg_cos)
    oracle = {
        "best_channel": float(oracle_channel.mean()),
        "best_channel_gain_over_bm25": float(oracle_channel.mean() - ndcg_bm25.mean()),
        "cosine_wins": int((ndcg_cos > ndcg_bm25).sum()),
        "bm25_wins": int((ndcg_bm25 > ndcg_cos).sum()),
        "ties": int((ndcg_bm25 == ndcg_cos).sum()),
        "best_minmax_weight": float(static["minmax"].max(axis=0).mean()),
        "best_any_static": float(np.maximum.reduce([static[k].max(axis=0) for k in static]).mean()),
        "channel_ndcg_spearman": spearman(ndcg_bm25, ndcg_cos),
    }

    stats, n_pos_head = gate_statistics(S_b, order_b)
    gain = ndcg_cos - ndcg_bm25
    diagnostics = {}
    for name, g in stats.items():
        med = float(np.median(g))
        low, high = g <= med, g > med
        diagnostics[name] = {
            "spearman_vs_cos_minus_bm25": spearman(g, gain),
            "spearman_vs_bm25_ndcg": spearman(g, ndcg_bm25),
            "spearman_vs_cos_ndcg": spearman(g, ndcg_cos),
            "quantiles_10_50_90": [float(x) for x in np.quantile(g, [0.1, 0.5, 0.9])],
            "mean_cos_minus_bm25_low_half": float(gain[low].mean()) if low.any() else None,
            "mean_cos_minus_bm25_high_half": float(gain[high].mean()) if high.any() else None,
        }
    diagnostics["positive_bm25_scores_in_top50"] = {"min": int(n_pos_head.min()), "median": float(np.median(n_pos_head)), "max": int(n_pos_head.max())}

    ctx = {"R": R, "idcg": idcg, "cand": cand, "norms": norms, "stats": stats, "static": static, "ndcg_bm25": ndcg_bm25, "ndcg_cos": ndcg_cos}
    idx = np.arange(n_q)
    odd, even = idx[idx % 2 == 1], idx[idx % 2 == 0]
    primary, best_pq = evaluate_fold("primary: tune=odd, eval=even", odd, even, ctx)
    swapped, _ = evaluate_fold("swapped: tune=even, eval=odd", even, odd, ctx)
    cross_best = (primary["best"]["ndcg10_eval"] * even.size + swapped["best"]["ndcg10_eval"] * odd.size) / n_q
    cross_bm25 = float(ndcg_bm25.mean())
    boot = paired_bootstrap(best_pq[even] - ndcg_bm25[even])
    t_fusion = time.perf_counter()

    roc = judged_pair_roc(S_b, S_c, order_b, rel)
    t_roc = time.perf_counter()

    auc_head = roc["all_bands"]["cos_pooled_auc"]
    bm25_eval = primary["bm25"]["ndcg10_eval"]
    best_eval = primary["best"]["ndcg10_eval"]
    auc_ok = auc_head is not None and auc_head >= AUC_MIN
    gain_ok = best_eval >= bm25_eval + GAIN_MIN
    verdict = {
        "gate": "GATE_OPEN" if (auc_ok and gain_ok) else "GATE_CLOSED",
        "auc_headline": auc_head,
        "auc_threshold": AUC_MIN,
        "auc_ok": bool(auc_ok),
        "best_fusion_family": primary["best"]["family"],
        "best_fusion_eval_ndcg10": best_eval,
        "bm25_eval_ndcg10": bm25_eval,
        "fusion_gain": best_eval - bm25_eval,
        "gain_threshold": GAIN_MIN,
        "gain_ok": bool(gain_ok),
        "gain_bootstrap_ci95": boot["ci95"],
        "note": "verdict from the primary fold point estimates; see bootstrap and swapped fold for stability",
    }

    doc = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": protocol_text(),
        "constants": {"k1": K1, "b": B, "cand_k": CAND_K, "rrf_k": RRF_K, "gate_top": GATE_TOP, "band_top": BAND_TOP, "n_bands": N_BANDS,
                      "outside_top": OUTSIDE_TOP, "weights": list(WEIGHTS), "sigmoid_grid": {"w_lo": list(SIGMOID_W_LO), "w_hi": list(SIGMOID_W_HI),
                      "centre_quantiles": list(SIGMOID_C_QUANTILES), "tau_scales": list(SIGMOID_TAU_SCALES), "statistics": list(GATE_STATS)},
                      "tie_eps": TIE_EPS},
        "data": {
            "cache": str(cache),
            "meta": meta,
            "n_corpus": n_corpus, "n_queries_file": n_queries, "n_queries_scored": n_q, "n_queries_dropped_no_positive": n_dropped,
            "qrels": qrel_counts,
            "split_identification": "323 queries / 12334 judgments = BEIR nfcorpus test split by count (dev has 324 / 11385)",
            "embedding_row_norms": {"corpus_min": float(d_norm.min()), "corpus_max": float(d_norm.max()), "queries_min": float(q_norm.min()), "queries_max": float(q_norm.max())},
            "bm25_index": {"n_terms": len(index.vocab), "n_postings": int(index.post_doc.size), "avgdl": index.avgdl},
            "sha256": {f: sha256_file(cache / f) for f in ("meta.json", "corpus.jsonl", "queries.jsonl", "qrels.tsv", "corpus.f32", "queries.f32")},
            "first_query_ids": q_ids_k[:3],
        },
        "baselines_all_queries": {
            "bm25_ndcg10": float(ndcg_bm25.mean()),
            "cosine_ndcg10": float(ndcg_cos.mean()),
            "bm25_ndcg10_odd": float(ndcg_bm25[odd].mean()), "bm25_ndcg10_even": float(ndcg_bm25[even].mean()),
            "cosine_ndcg10_odd": float(ndcg_cos[odd].mean()), "cosine_ndcg10_even": float(ndcg_cos[even].mean()),
            "reference_points_dev_split": REFERENCE_DEV,
        },
        "static_curves_all_queries": {
            "note": "information only: every point is scored on the evaluation queries",
            "weights": list(WEIGHTS),
            **{k: {"ndcg10": [float(x) for x in static[k].mean(axis=1)], "best_weight": WEIGHTS[int(np.argmax(static[k].mean(axis=1)))],
                   "best_ndcg10": float(static[k].mean(axis=1).max())} for k in static},
            "rrf_k60_equal_ndcg10": float(static["rrf"][WEIGHTS.index(0.5)].mean()),
        },
        "gate_statistic_diagnostics": diagnostics,
        "fusion_selection": {"primary": primary, "swapped": swapped,
                             "cross_fit": {"best_fusion_eval_ndcg10": float(cross_best), "bm25_ndcg10": cross_bm25, "delta": float(cross_best - cross_bm25)}},
        "bootstrap_primary_best_minus_bm25": boot,
        "roc": roc,
        "recall": recall,
        "oracle": oracle,
        "verdict": verdict,
        "timing_s": {"load": t_load - t0, "bm25": t_bm25 - t_load, "static_fusion": t_static - t_bm25, "gated_fusion_and_folds": t_fusion - t_static,
                     "roc": t_roc - t_fusion, "total": time.perf_counter() - t0},
    }
    doc = rounded(doc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"cannot write {out_path}: {exc}") from exc
    return doc


def fmt(x, nd: int = 4) -> str:
    """Fixed-width number or '  n/a'."""
    return "  n/a" if x is None else f"{x:.{nd}f}"


def print_summary(doc: dict, out_path: Path) -> None:
    """One-screen summary with the verdict."""
    d, b, s = doc["data"], doc["baselines_all_queries"], doc["static_curves_all_queries"]
    p, sw, cf = doc["fusion_selection"]["primary"], doc["fusion_selection"]["swapped"], doc["fusion_selection"]["cross_fit"]
    roc, rc, oc, v, bt = doc["roc"], doc["recall"], doc["oracle"], doc["verdict"], doc["bootstrap_primary_best_minus_bm25"]
    g = doc["gate_statistic_diagnostics"]
    lines = [
        f"=== dense-channel relevance gate: NFCorpus ({d['n_queries_scored']} q / {d['n_corpus']} d, MiniLM-L6 {d['meta']['dim']}-d) ===",
        f"split: {d['split_identification']}; references (dev): bm25 {b['reference_points_dev_split']['bm25_warm']} minilm {b['reference_points_dev_split']['minilm_l6']}",
        f"nDCG@10 all q: BM25 {fmt(b['bm25_ndcg10'])}  cosine {fmt(b['cosine_ndcg10'])}  oracle best-channel {fmt(oc['best_channel'])} "
        f"(cos wins {oc['cosine_wins']}, bm25 wins {oc['bm25_wins']}, ties {oc['ties']})  oracle best-weight {fmt(oc['best_minmax_weight'])}",
        f"static curves all q (info): minmax w={s['minmax']['best_weight']} {fmt(s['minmax']['best_ndcg10'])} | z w={s['z']['best_weight']} {fmt(s['z']['best_ndcg10'])} "
        f"| rrf k60 equal {fmt(s['rrf_k60_equal_ndcg10'])} (best w={s['rrf']['best_weight']} {fmt(s['rrf']['best_ndcg10'])})",
        f"primary fold (tune=odd idx n={p['n_tune']}, eval=even idx n={p['n_eval']}): BM25 eval {fmt(p['bm25']['ndcg10_eval'])}  cosine eval {fmt(p['cosine']['ndcg10_eval'])}",
    ]
    for name, fam in p["families"].items():
        params = {k: val for k, val in fam["params"].items() if k not in ("calibration",)}
        lines.append(f"  {name:<24} tune {fmt(fam['ndcg10_tune'])}  eval {fmt(fam['ndcg10_eval'])}  {params}")
    lines += [
        f"  BEST {p['best']['family']}: eval {fmt(p['best']['ndcg10_eval'])}  delta vs BM25 {p['best']['delta_vs_bm25_eval']:+.4f}  "
        f"bootstrap 95% CI [{bt['ci95'][0]:+.4f}, {bt['ci95'][1]:+.4f}]  P(mean<=0) {bt['p_mean_le_0']:.3f}",
        f"  best gated {p['best_gated']['family']} eval {fmt(p['best_gated']['ndcg10_eval'])} vs best static {p['best_static']['family']} {fmt(p['best_static']['ndcg10_eval'])} "
        f"(gated - static {p['best_gated']['delta_vs_best_static_eval']:+.4f})",
        f"swapped fold: BEST {sw['best']['family']} eval {fmt(sw['best']['ndcg10_eval'])} vs BM25 {fmt(sw['bm25']['ndcg10_eval'])} ({sw['best']['delta_vs_bm25_eval']:+.4f}); "
        f"cross-fit best {fmt(cf['best_fusion_eval_ndcg10'])} vs BM25 {fmt(cf['bm25_ndcg10'])} ({cf['delta']:+.4f})",
        "ROC cosine AUC (judged pairs): within BM25 bands pooled " + fmt(roc["all_bands"]["cos_pooled_auc"], 3) + " macro " + fmt(roc["all_bands"]["cos_macro_auc"], 3)
        + " [" + " ".join(f"b{i + 1} {fmt(roc[f'band{i + 1}']['cos_pooled_auc'], 3)}" for i in range(N_BANDS)) + f"] bm25 control {fmt(roc['all_bands']['bm25_pooled_auc'], 3)}; "
        f"outside top-50 {fmt(roc['outside_top50']['cos_pooled_auc'], 3)} (bm25 {fmt(roc['outside_top50']['bm25_pooled_auc'], 3)}); all docs {fmt(roc['all_docs']['cos_pooled_auc'], 3)} (bm25 {fmt(roc['all_docs']['bm25_pooled_auc'], 3)}); "
        f"pairs within bands {roc['all_bands']['n_pairs']}",
        " ".join(f"recall{k}: BM25 {fmt(rc[k]['bm25'])} cos {fmt(rc[k]['cosine'])} union {fmt(rc[k]['union'])} |" for k in rc),
        "gate stat spearman vs (cos - bm25): " + "  ".join(f"{n} {fmt(g[n]['spearman_vs_cos_minus_bm25'], 3)}" for n in GATE_STATS),
        f"VERDICT: {v['gate']}  (AUC {fmt(v['auc_headline'], 3)} >= {AUC_MIN}: {'yes' if v['auc_ok'] else 'no'}; fusion gain {v['fusion_gain']:+.4f} >= {GAIN_MIN}: {'yes' if v['gain_ok'] else 'no'})",
        f"runtime {doc['timing_s']['total']:.1f} s -> {out_path}",
    ]
    print("\n".join(lines))


def main() -> None:
    """CLI entry: --cache <dir> --out <json>."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "warm-bench" / "nfcorpus", help="NFCorpus cache directory")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out" / "roc_v1.json", help="result JSON path")
    args = ap.parse_args()
    for name in ("meta.json", "corpus.jsonl", "queries.jsonl", "qrels.tsv", "corpus.f32", "queries.f32"):
        if not (args.cache / name).is_file():
            raise SystemExit(f"missing input: {args.cache / name}")
    doc = run(args.cache, args.out)
    print_summary(doc, args.out)


if __name__ == "__main__":
    main()
