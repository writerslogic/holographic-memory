#!/usr/bin/env python3
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Holographic late interaction (HLI) kill test on BEIR NFCorpus and SciFact.

Question: can ColBERT-style token-level retrieval (MaxSim over contextual token embeddings) be
compressed into ONE superposed hypervector per document with VSA algebra (random projection,
permutation binding, bundling, optional binarisation) at single-vector cost, and recover a useful
part of the MaxSim-minus-dense quality gap?

Mechanisms, all computed from the same token embeddings (MiniLM-L6 last hidden state, 384-d,
documents truncated at 256 word pieces, [CLS]/[SEP] dropped, every token l2-normalised):

  dense        cosine of the model's pooled sentence embedding (mean pooling + l2), the baseline.
  maxsim       sum_t max_s cos(q_t, d_s) / n_q, the late-interaction upper bound; also with
               corpus-mean-centred tokens (maxsim_centered) and with IDF-weighted query tokens
               (maxsim_idf).
  hli          token hypervector h(x) = f(R^T x), R a fixed Gaussian 384 x D matrix (entries
               N(0, 1/D), nested across D by slicing one 384 x 16384 draw), f one of
                 real    identity                       real_c  centred: R^T (x - mu) / |x - mu|
                 sign    sign(R^T x) / sqrt(D)          rff     sqrt(2/D) cos(sqrt(2 g D) R^T x + b)
                                                                (random Fourier features of the RBF
                                                                kernel exp(-g |x - y|^2), g = --rff-gamma)
               Grouping of a document's tokens into G groups: `pos` = relative position bucket
               floor(p G / n) (the task's B), or `kmeans` = seeded spherical k-means on the token
               embeddings (addition B, K groups). Storage: `sup` (superposed) binds group g by a
               cyclic permutation (np.roll by g D / G) and sums everything into ONE D-vector;
               `sep` keeps G separate D-vectors (memory G x D). Bundle = weighted sum of the
               token hypervectors (weights 1, or wordpiece IDF with --additions, addition A).
               Addition C (expansion) replaces each document token x by l2(x + lambda sum_i s_i p_i)
               before projection, where p_i are the corpus-mean contextual embeddings (prototypes)
               of the 3 vocabulary items nearest to x's wordpiece in the model's (tied) word
               embedding matrix and s_i their cosines. Addition E (meansub) subtracts
               W_{d,g} h_bar from each stored group, h_bar the (weight-)mean token hypervector over
               the corpus and W_{d,g} the group's weight mass, before quantisation.
               Stored bundle: `int8` (per-row symmetric int8 + fp32 scale, dequantised and
               l2-normalised for scoring; memory G x (D + 4) bytes) or `bin` (sign of the sum,
               memory G x D / 8 bytes; the query token is also binarised and the score is the
               +-1 dot product, i.e. an affine function of the Hamming distance).
               Score(q, d) = sum_t w_t max_g <key_g(h(q_t)), M_d> / (|M_d| sum_t w_t) for `sup`
               (unbind each group key on the query side, G rolled copies), or
               max_g <h(q_t), M_{d,g}> / |M_{d,g}| for `sep` (empty groups masked).
  hli_puresum  <sum_t h(q_t), M_d> cosine, the control that should collapse to dense.

Evaluation modes for every row: first-stage (exhaustive over the corpus), rerank (addition D:
candidates = dense top-100 union positive-BM25 top-100, non-candidates unretrieved) and
hybrid_bm25 (per-query z-scores of the row and of BM25 summed). Metrics: nDCG@10 (linear gain)
and recall@100, on all queries and on the odd (tune) / even (eval) query-index split; paired
bootstrap (2000 resamples, seed 0, shared index matrix) of nDCG@10 against dense (and against
dense+BM25 in hybrid mode). Memory bytes per document, multiply-adds (bit operations for `bin`)
per document per query, the wall time of the full scoring pass at --threads, and a single-thread
micro-benchmark (median of 5 repeats after 1 warm-up on --time-queries queries).

Pre-registered verdict (per dataset and D, first-stage mode, all queries): PASS if an hli row
(not the pure-sum control) with memory <= 4 x dense (4 x 1536 bytes) beats dense on nDCG@10 by
>= 0.02 with the bootstrap 95% CI excluding zero. Overall PASS = some D passes on both datasets.
Best-combination reports pick the row with the highest tune-split nDCG@10 and show its eval split.

Capacity diagnostic: fraction of a document's own tokens whose similarity to the bundle exceeds
the 99th percentile of foreign tokens, by token count and D, on real tokens (200 sampled
documents, 2000 foreign tokens) and on synthetic isotropic unit tokens.

Data layout. Token banks are ragged: `tok` (n_tokens_total, 384) float32 unit rows, `off`
(n_items + 1) int64 offsets (item i owns rows off[i]:off[i+1], always >= 1 row), `ids` wordpiece
ids, `pooled` (n_items, 384). The bank is cached as float16 in out/<dataset>.<docs|queries>.*.npy
with a .cache.json guard (model, revision, max length); delete the cache to re-embed. Bundling
streams document-aligned chunks of <= --token-chunk tokens through one projection per D and
accumulates every (token_map, grouping, store, idf, expand) accumulator of a batch, batches sized
by --acc-budget-gb. Invariants: offsets strictly increase; every (document, group) segment is
non-empty; position-bucket permutations are the identity; all randomness comes from fixed seeds
(projection/RFF phases: --seed; k-means, bootstrap, capacity sampling: constants below).

Runtime: ~10-25 min on a 10-core CPU for both datasets with caches warm; token embedding adds
1-3 min on the first run. `--dims`, `--buckets`, `--kmeans`, `--maps`, `--no-additions`,
`--no-timing` and `--max-docs` trim the grid (see RUN.md).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "roc"))
try:
    import roc  # noqa: E402  benchmarks/research/roc/roc.py: BM25, readers, metrics, bootstrap
except ImportError as exc:  # pragma: no cover - environment problem
    raise SystemExit(f"cannot import benchmarks/research/roc/roc.py ({exc}); it is required for BM25") from exc

SCHEMA = "warm.research.hli/1"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_SEQ = 256
HIDDEN = 384
DIMS = (2048, 4096, 8192, 16384)
BUCKETS = (1, 4, 16)
KMEANS_KS = (2, 4, 8)
TOKEN_MAPS = ("real", "real_c", "sign", "rff")
NDCG_K = 10
RECALL_K = 100
CAND_K = 100
BOOT_N = 2000
BOOT_SEED = 0
KMEANS_SEED = 7
KMEANS_ITERS = 10
CAP_SEED = 11
CAP_DOCS = 200
CAP_FOREIGN = 2000
CAP_TRIALS = 20
CAP_SIZES = (16, 32, 64, 128, 256)
CAP_BINS = ((1, 32), (33, 64), (65, 128), (129, 256))
CAP_QUANTILE = 0.99
PASS_MARGIN = 0.02
MEM_MULT = 4.0
DENSE_BYTES = HIDDEN * 4
EXPAND_TOP = 3
MAXSIM_QUERY_CHUNK = 8
HLI_QUERY_CHUNK = 32
T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def l2n(x: np.ndarray) -> np.ndarray:
    """Rows scaled to unit norm (zero rows stay zero)."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------


@dataclass
class Dataset:
    name: str
    doc_ids: list[str]
    doc_texts: list[str]
    q_ids: list[str]
    q_texts: list[str]
    R: np.ndarray  # (n_q, n_docs) int16 grades, every row has a positive grade
    cached_doc_vecs: np.ndarray | None
    cached_q_vecs: np.ndarray | None
    qrel_counts: dict
    n_queries_dropped: int
    subsample: dict | None


def load_dataset(name: str, cache_dir: Path, max_docs: int, seed: int) -> Dataset:
    meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    doc_ids, doc_texts = roc.read_jsonl(cache_dir / "corpus.jsonl")
    q_ids, q_texts = roc.read_jsonl(cache_dir / "queries.jsonl")
    if len(doc_ids) != int(meta["n_corpus"]) or len(q_ids) != int(meta["n_queries"]):
        raise SystemExit(f"{cache_dir}: jsonl counts {len(doc_ids)}/{len(q_ids)} differ from meta.json")
    dim = int(meta.get("dim", HIDDEN))

    def f32(fname: str, rows: int) -> np.ndarray | None:
        p = cache_dir / fname
        if dim != HIDDEN or not p.exists() or p.stat().st_size != rows * dim * 4 or meta.get("model") != MODEL_ID:
            return None
        return np.fromfile(p, dtype=np.float32).reshape(rows, dim)

    doc_vecs = f32("corpus.f32", len(doc_ids))
    q_vecs = f32("queries.f32", len(q_ids))
    R, counts = roc.read_qrels(cache_dir / "qrels.tsv", {q: i for i, q in enumerate(q_ids)},
                               {d: i for i, d in enumerate(doc_ids)})
    subsample = None
    if max_docs > 0 and len(doc_ids) > max_docs:
        judged = np.flatnonzero((R > 0).any(axis=0))
        rest = np.setdiff1d(np.arange(len(doc_ids)), judged)
        rng = np.random.default_rng(seed)
        extra = rng.choice(rest, size=max(0, min(len(rest), max_docs - len(judged))), replace=False)
        keep_d = np.sort(np.concatenate([judged, extra]))
        subsample = {"n_docs_full": len(doc_ids), "n_judged_kept": int(len(judged)), "n_random_kept": int(len(extra)),
                     "seed": seed}
        doc_ids = [doc_ids[i] for i in keep_d]
        doc_texts = [doc_texts[i] for i in keep_d]
        R = R[:, keep_d]
        doc_vecs = None if doc_vecs is None else doc_vecs[keep_d]
    keep_q = np.flatnonzero((R > 0).any(axis=1))
    dropped = len(q_ids) - len(keep_q)
    return Dataset(name, doc_ids, doc_texts, [q_ids[i] for i in keep_q], [q_texts[i] for i in keep_q],
                   R[keep_q], doc_vecs, None if q_vecs is None else q_vecs[keep_q], counts, dropped, subsample)


# ----------------------------------------------------------------------------------------------
# token embeddings (cached)
# ----------------------------------------------------------------------------------------------


@dataclass
class TokenBank:
    tok: np.ndarray  # (n_tok, 384) float32 unit rows
    off: np.ndarray  # (n_items + 1,) int64
    ids: np.ndarray  # (n_tok,) int32 wordpiece ids
    pooled: np.ndarray  # (n_items, 384) float32 unit rows
    n_at_cap: int  # items whose attention length reached MAX_SEQ (truncated or exactly full)

    @property
    def counts(self) -> np.ndarray:
        return np.diff(self.off)

    @property
    def n_items(self) -> int:
        return len(self.off) - 1


class Embedder:
    """Lazy pinned MiniLM-L6 on CPU; one forward pass yields token states and the pooled vector."""

    def __init__(self, threads: int, batch_size: int):
        self.threads = threads
        self.batch_size = batch_size
        self.model = None

    def load(self):
        if self.model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(self.threads)
            log(f"loading {MODEL_ID}@{MODEL_REVISION[:12]} on cpu")
            m = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu")
            m.max_seq_length = MAX_SEQ
            m.eval()
            self.model = m
        return self.model

    def word_embeddings(self) -> np.ndarray:
        m = self.load()
        w = m[0].auto_model.get_input_embeddings().weight.detach().float().numpy()
        return np.ascontiguousarray(w, dtype=np.float32)

    def embed(self, texts: list[str], label: str) -> TokenBank:
        import torch

        m = self.load()
        special = torch.tensor(sorted(set(int(i) for i in m.tokenizer.all_special_ids)), dtype=torch.long)
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        parts: list[np.ndarray | None] = [None] * len(texts)
        id_parts: list[np.ndarray | None] = [None] * len(texts)
        pooled = np.zeros((len(texts), HIDDEN), dtype=np.float32)
        n_at_cap = 0
        t = time.perf_counter()
        with torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                feats = m.tokenize([texts[i] for i in idx])
                out = m(feats)
                tok = out["token_embeddings"].float()
                mask = feats["attention_mask"].bool()
                ids = feats["input_ids"]
                keep = mask & ~torch.isin(ids, special)
                sent = out["sentence_embedding"].float().numpy()
                for r, i in enumerate(idx):
                    k = keep[r]
                    if not bool(k.any()):
                        k = mask[r]
                    parts[i] = tok[r][k].numpy().astype(np.float16)
                    id_parts[i] = ids[r][k].numpy().astype(np.int32)
                    pooled[i] = sent[r]
                    if int(mask[r].sum()) >= MAX_SEQ:
                        n_at_cap += 1
                if (s // self.batch_size) % 20 == 0:
                    log(f"  {label}: {s + len(idx)}/{len(texts)} embedded")
        counts = np.array([p.shape[0] for p in parts], dtype=np.int64)
        off = np.zeros(len(texts) + 1, dtype=np.int64)
        off[1:] = np.cumsum(counts)
        tok16 = np.concatenate(parts, axis=0)
        ids32 = np.concatenate(id_parts, axis=0)
        log(f"  {label}: {len(texts)} items, {int(off[-1])} tokens in {time.perf_counter() - t:.1f}s")
        return TokenBank(l2n(tok16.astype(np.float32)), off, ids32, l2n(pooled), n_at_cap)


def cache_paths(out_dir: Path, name: str, kind: str) -> dict[str, Path]:
    stem = out_dir / f"{name}.{kind}"
    return {"tok": Path(f"{stem}.tokens.npy"), "off": Path(f"{stem}.offsets.npy"), "ids": Path(f"{stem}.ids.npy"),
            "pooled": Path(f"{stem}.pooled.npy"), "meta": Path(f"{stem}.cache.json")}


def bank_cached(out_dir: Path, name: str, kind: str, texts: list[str], embedder: Embedder) -> tuple[TokenBank, bool]:
    """The token bank for `texts`, from the cache when its guard matches, else embedded and cached."""
    p = cache_paths(out_dir, name, kind)
    guard = {"model": MODEL_ID, "revision": MODEL_REVISION, "max_seq_length": MAX_SEQ, "n_items": len(texts),
             "drop_special_tokens": True, "dtype": "float16"}
    if all(q.exists() for q in p.values()):
        try:
            meta = json.loads(p["meta"].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        if {k: meta.get(k) for k in guard} == guard:
            tok = l2n(np.load(p["tok"]).astype(np.float32))
            off = np.load(p["off"]).astype(np.int64)
            ids = np.load(p["ids"]).astype(np.int32)
            pooled = l2n(np.load(p["pooled"]).astype(np.float32))
            if len(off) == len(texts) + 1 and off[-1] == tok.shape[0] and ids.shape[0] == tok.shape[0]:
                return TokenBank(tok, off, ids, pooled, int(meta.get("n_at_cap", -1))), True
    bank = embedder.embed(texts, f"{name}/{kind}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(p["tok"], bank.tok.astype(np.float16))
    np.save(p["off"], bank.off)
    np.save(p["ids"], bank.ids)
    np.save(p["pooled"], bank.pooled)
    p["meta"].write_text(json.dumps({**guard, "n_at_cap": bank.n_at_cap}, indent=1), encoding="utf-8")
    return bank, False


def word_embeddings_cached(out_dir: Path, embedder: Embedder) -> np.ndarray:
    p = out_dir / "word_embeddings.npy"
    if p.exists():
        w = np.load(p)
        if w.ndim == 2 and w.shape[1] == HIDDEN:
            return w.astype(np.float32)
    w = embedder.word_embeddings()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(p, w)
    return w


# ----------------------------------------------------------------------------------------------
# groupings: position buckets and per-document spherical k-means
# ----------------------------------------------------------------------------------------------


@dataclass
class Grouping:
    name: str
    n_groups: int
    perm: np.ndarray | None  # token permutation sorted by (doc, group, position); None = identity
    starts: np.ndarray  # segment starts in permuted index space, strictly increasing
    seg_doc: np.ndarray
    seg_group: np.ndarray
    seg_ptr: np.ndarray  # (n_docs + 1,) first segment of each document


def make_grouping(name: str, off: np.ndarray, group_of_token: np.ndarray, n_groups: int) -> Grouping:
    n_docs = len(off) - 1
    counts = np.diff(off)
    doc_of_token = np.repeat(np.arange(n_docs, dtype=np.int64), counts)
    pos = np.arange(off[-1], dtype=np.int64) - off[doc_of_token]
    perm = np.lexsort((pos, group_of_token, doc_of_token))
    identity = bool(np.array_equal(perm, np.arange(off[-1])))
    gd = doc_of_token[perm]
    gg = group_of_token[perm].astype(np.int64)
    key = gd * n_groups + gg
    starts = np.concatenate([[0], np.flatnonzero(np.diff(key)) + 1]).astype(np.int64)
    seg_doc = gd[starts]
    seg_group = gg[starts]
    seg_ptr = np.searchsorted(seg_doc, np.arange(n_docs + 1), side="left").astype(np.int64)
    return Grouping(name, n_groups, None if identity else perm, starts, seg_doc, seg_group, seg_ptr)


def position_groups(off: np.ndarray, n_buckets: int) -> np.ndarray:
    counts = np.diff(off)
    doc_of_token = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    pos = np.arange(off[-1], dtype=np.int64) - off[doc_of_token]
    return np.minimum(pos * n_buckets // counts[doc_of_token], n_buckets - 1)


def kmeans_groups(bank: TokenBank, k: int, seed: int) -> np.ndarray:
    """Seeded spherical k-means labels per token, run independently per document."""
    rng = np.random.default_rng(seed + k)
    labels = np.zeros(bank.off[-1], dtype=np.int64)
    for d in range(bank.n_items):
        a, z = int(bank.off[d]), int(bank.off[d + 1])
        n = z - a
        if n <= k:
            labels[a:z] = np.arange(n)
            continue
        X = bank.tok[a:z]
        C = X[rng.choice(n, size=k, replace=False)]
        lab = np.zeros(n, dtype=np.int64)
        for it in range(KMEANS_ITERS):
            new = np.argmax(X @ C.T, axis=1)
            if it > 0 and np.array_equal(new, lab):
                break
            lab = new
            S = np.zeros((k, HIDDEN), dtype=np.float32)
            np.add.at(S, lab, X)
            empty = np.linalg.norm(S, axis=1) < 1e-12
            if empty.any():
                S[empty] = X[rng.integers(0, n, size=int(empty.sum()))]
            C = l2n(S)
        labels[a:z] = lab
    return labels


def doc_chunks(off: np.ndarray, max_tokens: int) -> list[tuple[int, int]]:
    """Document index ranges with <= max_tokens tokens each (at least one document)."""
    n = len(off) - 1
    chunks = []
    c0 = 0
    while c0 < n:
        c1 = c0 + 1
        while c1 < n and off[c1 + 1] - off[c0] <= max_tokens:
            c1 += 1
        chunks.append((c0, c1))
        c0 = c1
    return chunks


# ----------------------------------------------------------------------------------------------
# IDF and expansion tables
# ----------------------------------------------------------------------------------------------


def wordpiece_idf(bank: TokenBank, vocab_size: int) -> np.ndarray:
    """BM25-style idf per wordpiece id from document frequency over the corpus (unseen ids get the max)."""
    df = np.zeros(vocab_size, dtype=np.int64)
    for d in range(bank.n_items):
        df[np.unique(bank.ids[bank.off[d]:bank.off[d + 1]])] += 1
    n = bank.n_items
    return np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)


def expansion_tables(bank: TokenBank, word_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Prototypes (corpus-mean contextual embedding per occurring wordpiece) and each occurring
    wordpiece's EXPAND_TOP nearest occurring wordpieces by cosine of the model's word embeddings."""
    import torch

    vocab = word_emb.shape[0]
    proto = np.zeros((vocab, HIDDEN), dtype=np.float32)
    np.add.at(proto, bank.ids, bank.tok)
    occ = np.flatnonzero(np.linalg.norm(proto, axis=1) > 0)
    proto = l2n(proto)
    E = torch.from_numpy(l2n(word_emb[occ]))
    nn_ids = np.zeros((vocab, EXPAND_TOP), dtype=np.int32)
    nn_sim = np.zeros((vocab, EXPAND_TOP), dtype=np.float32)
    step = 2048
    for s in range(0, len(occ), step):
        sims = torch.matmul(E[s:s + step], E.T).numpy()
        rows = np.arange(sims.shape[0])
        sims[rows, rows + s] = -np.inf
        top = np.argpartition(-sims, EXPAND_TOP - 1, axis=1)[:, :EXPAND_TOP]
        tv = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-tv, axis=1, kind="stable")
        top = np.take_along_axis(top, order, axis=1)
        tv = np.take_along_axis(tv, order, axis=1)
        nn_ids[occ[s:s + step]] = occ[top]
        nn_sim[occ[s:s + step]] = np.maximum(tv, 0.0)
    info = {"vocab_size": int(vocab), "occurring_wordpieces": int(len(occ)), "top": EXPAND_TOP,
            "mean_neighbour_cosine": float(nn_sim[occ].mean())}
    return proto, nn_ids, nn_sim, info


def expand_tokens(X: np.ndarray, ids: np.ndarray, proto: np.ndarray, nn_ids: np.ndarray, nn_sim: np.ndarray,
                  lam: float) -> np.ndarray:
    nb = proto[nn_ids[ids]]  # (n, top, 384)
    add = np.einsum("nk,nkd->nd", nn_sim[ids], nb)
    return l2n(X + lam * add)


# ----------------------------------------------------------------------------------------------
# hypervectors, accumulation, scoring
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AccKey:
    token_map: str
    grouping: str  # pos | kmeans
    n_groups: int
    store: str  # sup | sep
    idf: bool
    expand: bool

    @property
    def label(self) -> str:
        flags = "+".join(f for f, on in (("idf", self.idf), ("exp", self.expand)) if on)
        return f"{self.token_map}|{self.grouping}{self.n_groups}|{self.store}" + (f"|{flags}" if flags else "")

    def grouping_name(self) -> str:
        return f"{self.grouping}{self.n_groups}"


@dataclass(frozen=True)
class RowCfg:
    key: AccKey
    bundle: str  # int8 | bin
    meansub: bool

    @property
    def label(self) -> str:
        return f"{self.key.label}|{self.bundle}" + ("|msub" if self.meansub else "")


@dataclass
class Acc:
    key: AccKey
    acc: np.ndarray  # (n_docs * n_groups, D) float32 bundle sums
    wsum: np.ndarray  # (n_docs, n_groups) float32 weight mass per group
    hsum: np.ndarray  # (D,) float32 weighted sum of token hypervectors over the corpus
    wtot: float


@dataclass
class Proj:
    D: int
    R: np.ndarray  # (384, D) float32, entries N(0, 1/D)
    muR: np.ndarray  # (D,) mu @ R
    rff_b: np.ndarray  # (D,) phases in [0, 2 pi)
    gamma: float


def shift_of(g: int, n_groups: int, D: int) -> int:
    return (g * D) // n_groups


def acc_rows(key: AccKey) -> int:
    """Stored D-vectors per document: one for a superposed bundle, n_groups for separate ones."""
    return key.n_groups if key.store == "sep" else 1


def is_puresum_trigger(cfg: RowCfg) -> bool:
    k = cfg.key
    return (k.grouping, k.n_groups, k.store, k.idf, k.expand, cfg.bundle, cfg.meansub) == (
        "pos", 1, "sup", False, False, "int8", False)


def hv_map(token_map: str, P: np.ndarray, pj: Proj, xcn: np.ndarray | None) -> np.ndarray:
    D = pj.D
    if token_map == "real":
        return P
    if token_map == "real_c":
        if xcn is None:
            raise SystemExit("real_c needs |x - mu| per token")
        return ((P - pj.muR[None, :]) / xcn[:, None]).astype(np.float32)
    if token_map == "sign":
        s = np.float32(1.0 / math.sqrt(D))
        return np.where(P >= 0, s, -s).astype(np.float32)
    if token_map == "rff":
        scale = np.float32(math.sqrt(2.0 * pj.gamma * D))
        return (np.float32(math.sqrt(2.0 / D)) * np.cos(scale * P + pj.rff_b[None, :])).astype(np.float32)
    raise SystemExit(f"unknown token map {token_map!r}")


def build_rows(maps: tuple[str, ...], buckets: tuple[int, ...], ks: tuple[int, ...], additions: bool) -> list[RowCfg]:
    rows: list[RowCfg] = []

    def add(key: AccKey, bundles: tuple[str, ...], meansub: bool = False) -> None:
        for b in bundles:
            rows.append(RowCfg(key, b, meansub))

    # Order = run order within one D: idf weighting and K-bundles first, the plain grid last.
    if additions:
        both = ("int8", "bin")
        add(AccKey("real", "pos", 1, "sup", True, False), both)  # A
        for K in sorted(ks, reverse=True):  # A+B
            add(AccKey("real", "kmeans", K, "sep", True, False), both)
        for K in sorted(ks, reverse=True):  # B
            add(AccKey("real", "kmeans", K, "sep", False, False), both)
        add(AccKey("real", "pos", 1, "sup", True, False), both, meansub=True)  # A+E
        add(AccKey("real", "pos", 1, "sup", True, True), both)  # A+C
        add(AccKey("real", "pos", 1, "sup", True, True), both, meansub=True)  # A+C+E
        add(AccKey("real", "kmeans", 4, "sep", True, True), both, meansub=True)  # A+C+E, K=4 separate
        add(AccKey("real", "pos", 4, "sup", True, True), both, meansub=True)  # A+C+E, 4 position buckets
        add(AccKey("real", "pos", 1, "sup", False, True), both)  # C
        add(AccKey("real", "pos", 1, "sup", False, False), both, meansub=True)  # E
    for m in maps:
        for B in buckets:
            add(AccKey(m, "pos", B, "sup", False, False), ("int8", "bin") if m in ("real", "sign") else ("int8",))
    seen: set[RowCfg] = set()
    uniq = []
    for r in rows:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def accumulate(pj: Proj, keys: list[AccKey], docs: TokenBank, groupings: dict[str, Grouping], idf_w: np.ndarray,
               xcn: np.ndarray, expand_fn, token_chunk: int) -> dict[AccKey, Acc]:
    """One streaming pass over the corpus tokens: project once per chunk, bundle every accumulator."""
    import torch

    N = docs.n_items
    D = pj.D
    R_t = torch.from_numpy(pj.R)
    accs = {k: Acc(k, np.zeros((N * acc_rows(k), D), dtype=np.float32), np.zeros((N, k.n_groups), dtype=np.float32),
                   np.zeros(D, dtype=np.float32), 0.0) for k in keys}
    need_exp = any(k.expand for k in keys)
    for c0, c1 in doc_chunks(docs.off, token_chunk):
        t0, t1 = int(docs.off[c0]), int(docs.off[c1])
        X = docs.tok[t0:t1]
        P = torch.matmul(torch.from_numpy(X), R_t).numpy()
        P_exp = torch.matmul(torch.from_numpy(expand_fn(X, docs.ids[t0:t1])), R_t).numpy() if need_exp else None
        for key in keys:
            H = hv_map(key.token_map, P_exp if key.expand else P, pj, xcn[t0:t1])
            w = idf_w[docs.ids[t0:t1]] if key.idf else np.ones(t1 - t0, dtype=np.float32)
            if key.idf:
                H = H * w[:, None]
            a = accs[key]
            a.hsum += H.sum(axis=0)
            a.wtot += float(w.sum())
            g = groupings[key.grouping_name()]
            sa, sb = int(g.seg_ptr[c0]), int(g.seg_ptr[c1])
            starts = g.starts[sa:sb] - t0
            if g.perm is not None:
                pl = g.perm[t0:t1] - t0
                H = H[pl]
                w = w[pl]
            sums = np.add.reduceat(H, starts, axis=0)
            wseg = np.add.reduceat(w, starts)
            sd = g.seg_doc[sa:sb]
            sg = g.seg_group[sa:sb]
            a.wsum[sd, sg] += wseg
            if key.store == "sup":
                for gi in range(key.n_groups):
                    sel = sg == gi
                    if sel.any():
                        a.acc[sd[sel]] += np.roll(sums[sel], shift_of(gi, key.n_groups, D), axis=1)
            else:
                a.acc[sd * key.n_groups + sg] += sums
    return accs


@dataclass
class DocMatrix:
    M: "object"  # torch tensor (n_docs * n_groups, D) float32 rows ready for the dot product
    valid: np.ndarray | None  # (n_docs, n_groups) bool for `sep`, None for `sup`
    scale: float  # score divisor (D for binary +-1 dots, 1 otherwise)
    int8_bytes: int


def prepare_matrix(a: Acc, cfg: RowCfg, D: int) -> DocMatrix:
    import torch

    key = cfg.key
    src = a.acc
    if cfg.meansub:
        src = a.acc.copy()
        hbar = a.hsum / max(a.wtot, 1e-12)
        if key.store == "sup":
            for gi in range(key.n_groups):
                src -= a.wsum[:, gi:gi + 1] * np.roll(hbar, shift_of(gi, key.n_groups, D))[None, :]
        else:
            src -= a.wsum.reshape(-1, 1) * hbar[None, :]
    valid = (a.wsum > 0) if key.store == "sep" else None
    out = np.empty_like(src)
    step = 4096
    for s in range(0, src.shape[0], step):
        blk = src[s:s + step]
        if cfg.bundle == "bin":
            out[s:s + step] = np.where(blk >= 0, np.float32(1.0), np.float32(-1.0))
        else:
            scale = np.abs(blk).max(axis=1, keepdims=True) / 127.0
            scale[scale == 0] = 1.0
            q = np.rint(blk / scale).astype(np.int8)  # the stored representation: int8 + fp32 scale per row
            out[s:s + step] = l2n(q.astype(np.float32) * scale)
    if valid is not None:
        out[~valid.reshape(-1)] = 0.0
    return DocMatrix(torch.from_numpy(out), valid, float(D) if cfg.bundle == "bin" else 1.0, 0)


def query_hv(token_map: str, Pq: np.ndarray, pj: Proj, qcn: np.ndarray, bundle: str) -> np.ndarray:
    H = hv_map(token_map, Pq, pj, qcn)
    if bundle == "bin":
        return np.where(H >= 0, np.float32(1.0), np.float32(-1.0)).astype(np.float32)
    return np.ascontiguousarray(H, dtype=np.float32)


def hli_scorer(Hq: np.ndarray, qbank: TokenBank, qw: np.ndarray, dm: DocMatrix, key: AccKey, D: int, n_docs: int):
    """Closure scoring queries [lo, hi) -> (hi - lo, n_docs) float32."""
    import torch

    G = key.n_groups
    shifts = [shift_of(g, G, D) for g in range(G)]
    neg = None
    if dm.valid is not None:
        neg = np.where(dm.valid, np.float32(0.0), np.float32(-np.inf)).astype(np.float32)  # (n_docs, G)

    def score(lo: int, hi: int) -> np.ndarray:
        out = np.empty((hi - lo, n_docs), dtype=np.float32)
        for a in range(lo, hi, HLI_QUERY_CHUNK):
            b = min(hi, a + HLI_QUERY_CHUNK)
            ta, tb = int(qbank.off[a]), int(qbank.off[b])
            rows = Hq[ta:tb]
            if key.store == "sup":
                Q = rows if G == 1 else np.concatenate([np.roll(rows, s, axis=1) for s in shifts], axis=0)
                S = torch.matmul(torch.from_numpy(np.ascontiguousarray(Q)), dm.M.T).numpy()
                if G > 1:
                    S = S.reshape(G, tb - ta, n_docs).max(axis=0)
            else:
                S = torch.matmul(torch.from_numpy(np.ascontiguousarray(rows)), dm.M.T).numpy()
                S = (S.reshape(tb - ta, n_docs, G) + neg[None, :, :]).max(axis=2)
            w = qw[ta:tb]
            starts = qbank.off[a:b] - ta
            agg = np.add.reduceat(S * w[:, None], starts, axis=0)
            wq = np.add.reduceat(w, starts)
            out[a - lo:b - lo] = agg / (wq[:, None] * dm.scale)
        return out

    return score


def puresum_scorer(Hq: np.ndarray, qbank: TokenBank, dm: DocMatrix, n_docs: int):
    import torch

    Qs = l2n(np.add.reduceat(Hq, qbank.off[:-1], axis=0))

    def score(lo: int, hi: int) -> np.ndarray:
        return torch.matmul(torch.from_numpy(np.ascontiguousarray(Qs[lo:hi])), dm.M.T).numpy() / dm.scale

    return score


def dense_scorer(qvec: np.ndarray, dvec: np.ndarray):
    import torch

    Dt = torch.from_numpy(np.ascontiguousarray(dvec))

    def score(lo: int, hi: int) -> np.ndarray:
        return torch.matmul(torch.from_numpy(np.ascontiguousarray(qvec[lo:hi])), Dt.T).numpy()

    return score


def maxsim_scorer(qbank: TokenBank, dtok: np.ndarray, doc_off: np.ndarray, qw: np.ndarray | None):
    """MaxSim over all document tokens, mean (or qw-weighted mean) over query tokens."""
    import torch

    Tt = torch.from_numpy(np.ascontiguousarray(dtok))
    n_docs = len(doc_off) - 1
    w_all = np.ones(qbank.off[-1], dtype=np.float32) if qw is None else qw

    def score(lo: int, hi: int) -> np.ndarray:
        out = np.empty((hi - lo, n_docs), dtype=np.float32)
        for a in range(lo, hi, MAXSIM_QUERY_CHUNK):
            b = min(hi, a + MAXSIM_QUERY_CHUNK)
            ta, tb = int(qbank.off[a]), int(qbank.off[b])
            S = torch.matmul(torch.from_numpy(np.ascontiguousarray(qbank.tok[ta:tb])), Tt.T).numpy()
            Mx = np.maximum.reduceat(S, doc_off[:-1], axis=1)
            w = w_all[ta:tb]
            starts = qbank.off[a:b] - ta
            out[a - lo:b - lo] = np.add.reduceat(Mx * w[:, None], starts, axis=0) / np.add.reduceat(w, starts)[:, None]
        return out

    return score


# ----------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------


@dataclass
class Eval:
    R: np.ndarray
    rel: np.ndarray
    idcg: np.ndarray
    tune: np.ndarray  # odd query indices
    evl: np.ndarray  # even query indices
    boot_all: np.ndarray  # (BOOT_N, n_q) resample indices
    boot_eval: np.ndarray  # (BOOT_N, n_eval)
    cand: np.ndarray  # (n_q, n_docs) rerank candidates
    bm25_z: np.ndarray  # (n_q, n_docs) float64 z-scored BM25

    def per_query(self, S: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
        S64 = S.astype(np.float64)
        if mode == "rerank":
            S64 = np.where(self.cand, S64, -np.inf)
        elif mode == "hybrid_bm25":
            S64 = zrows(S64) + self.bm25_z
        top = roc.topk(S64, RECALL_K)
        return roc.ndcg10(top, self.R, self.idcg), roc.recall_at(top, self.rel, RECALL_K)

    def boot(self, diff: np.ndarray, idx: np.ndarray) -> dict:
        means = diff[idx].mean(axis=1)
        return {"mean": float(diff.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
                "p_mean_le_0": float((means <= 0).mean()), "n_resamples": int(idx.shape[0]), "n_queries": int(diff.size),
                "seed": BOOT_SEED}


def zrows(S: np.ndarray) -> np.ndarray:
    m = S.mean(axis=1, keepdims=True)
    s = S.std(axis=1, keepdims=True)
    return (S - m) / np.maximum(s, 1e-9)


def split_means(v: np.ndarray, ev: Eval) -> dict:
    return {"all": float(v.mean()), "tune": float(v[ev.tune].mean()), "eval": float(v[ev.evl].mean())}


MODES = ("first_stage", "rerank", "hybrid_bm25")


def measure(S: np.ndarray, ev: Eval) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for mode in MODES:
        nd, rc = ev.per_query(S, mode)
        out[mode] = {"ndcg10": nd, "recall100": rc}
    return out


def compare(mine: dict, ref: dict, ev: Eval, maxsim: dict | None, dense: dict | None) -> dict:
    """Split means plus bootstrap of nDCG@10 against the reference row, per mode."""
    out = {}
    for mode in MODES:
        nd, rc = mine[mode]["ndcg10"], mine[mode]["recall100"]
        d = {"ndcg10": split_means(nd, ev), "recall100": split_means(rc, ev)}
        if ref is not None:
            diff = nd - ref[mode]["ndcg10"]
            d["vs_ref_ndcg10_all"] = ev.boot(diff, ev.boot_all)
            d["vs_ref_ndcg10_eval"] = ev.boot(diff[ev.evl], ev.boot_eval)
            d["vs_ref_recall100_all"] = float((rc - ref[mode]["recall100"]).mean())
        if maxsim is not None and dense is not None:
            gap = float(maxsim[mode]["ndcg10"].mean() - dense[mode]["ndcg10"].mean())
            d["vs_maxsim_ndcg10_all"] = float(nd.mean() - maxsim[mode]["ndcg10"].mean())
            d["gap_closed_all"] = None if abs(gap) < 1e-9 else float((nd.mean() - dense[mode]["ndcg10"].mean()) / gap)
        out[mode] = d
    return out


def timed(score, n_q: int, threads: int, timing: bool, n_time: int, repeats: int, warmup: int) -> tuple[np.ndarray, dict]:
    import torch

    torch.set_num_threads(threads)
    t = time.perf_counter()
    S = score(0, n_q)
    wall = time.perf_counter() - t
    micro = None
    if timing:
        nt = min(n_time, n_q)
        torch.set_num_threads(1)
        try:
            for _ in range(warmup):
                score(0, nt)
            ts = []
            for _ in range(repeats):
                t = time.perf_counter()
                score(0, nt)
                ts.append((time.perf_counter() - t) * 1e3 / nt)
        finally:
            torch.set_num_threads(threads)
        micro = {"median_ms_per_query": float(np.median(ts)), "min_ms_per_query": float(min(ts)), "repeats": repeats,
                 "warmup": warmup, "n_queries": nt, "threads": 1}
    return S, {"scoring_wall_s": wall, "threads": threads, "ms_per_query_1thread": micro}


# ----------------------------------------------------------------------------------------------
# capacity diagnostic
# ----------------------------------------------------------------------------------------------


def recoverability(own: np.ndarray, foreign: np.ndarray) -> tuple[float, float]:
    """(fraction of own scores above the CAP_QUANTILE of foreign scores, P(own > foreign))."""
    thr = float(np.quantile(foreign, CAP_QUANTILE))
    sf = np.sort(foreign)
    auc = float(np.searchsorted(sf, own, side="left").mean() / len(sf))
    return float((own > thr).mean()), auc


def bundle_scores(H_own: np.ndarray, H_for: np.ndarray, bundle: str) -> tuple[np.ndarray, np.ndarray]:
    m = H_own.sum(axis=0)
    if bundle == "bin":
        mb = np.where(m >= 0, 1.0, -1.0)
        return np.sign(H_own) @ mb / len(m), np.sign(H_for) @ mb / len(m)
    mn = m / max(np.linalg.norm(m), 1e-12)
    return H_own @ mn, H_for @ mn


def capacity_real(docs: TokenBank, pj: Proj, rng: np.random.Generator) -> dict:
    counts = docs.counts
    eligible = np.flatnonzero(counts >= 2)
    sample = rng.choice(eligible, size=min(CAP_DOCS, len(eligible)), replace=False)
    per_bin: dict[str, dict[str, list]] = {f"{lo}-{hi}": {"int8": [], "bin": [], "auc_int8": [], "auc_bin": []}
                                           for lo, hi in CAP_BINS}
    n_tok = int(docs.off[-1])
    for d in sample:
        a, z = int(docs.off[d]), int(docs.off[d + 1])
        fo = rng.integers(0, n_tok, size=CAP_FOREIGN)
        fo = fo[(fo < a) | (fo >= z)]
        H_own = docs.tok[a:z] @ pj.R
        H_for = docs.tok[fo] @ pj.R
        n = z - a
        key = next((f"{lo}-{hi}" for lo, hi in CAP_BINS if lo <= n <= hi), None)
        if key is None:
            continue
        for bundle in ("int8", "bin"):
            so, sf = bundle_scores(H_own, H_for, bundle)
            frac, auc = recoverability(so, sf)
            per_bin[key][bundle].append(frac)
            per_bin[key][f"auc_{bundle}"].append(auc)
    return {"n_docs_sampled": int(len(sample)), "n_foreign_tokens": CAP_FOREIGN, "quantile": CAP_QUANTILE,
            "by_token_count": {k: {"n_docs": len(v["int8"]),
                                   "frac_recoverable_real": float(np.mean(v["int8"])) if v["int8"] else None,
                                   "frac_recoverable_binary": float(np.mean(v["bin"])) if v["bin"] else None,
                                   "auc_real": float(np.mean(v["auc_int8"])) if v["auc_int8"] else None,
                                   "auc_binary": float(np.mean(v["auc_bin"])) if v["auc_bin"] else None}
                               for k, v in per_bin.items()}}


def capacity_synthetic(pj: Proj, rng: np.random.Generator) -> dict:
    out = {}
    for n in CAP_SIZES:
        cells = {"real_tokens_real_bundle": [], "real_tokens_binary_bundle": [], "sign_tokens_binary_bundle": []}
        for _ in range(CAP_TRIALS):
            X = l2n(rng.standard_normal((n, HIDDEN)).astype(np.float32))
            F = l2n(rng.standard_normal((CAP_FOREIGN, HIDDEN)).astype(np.float32))
            H_own, H_for = X @ pj.R, F @ pj.R
            cells["real_tokens_real_bundle"].append(recoverability(*bundle_scores(H_own, H_for, "int8"))[0])
            cells["real_tokens_binary_bundle"].append(recoverability(*bundle_scores(H_own, H_for, "bin"))[0])
            cells["sign_tokens_binary_bundle"].append(
                recoverability(*bundle_scores(np.sign(H_own), np.sign(H_for), "bin"))[0])
        out[str(n)] = {k: float(np.mean(v)) for k, v in cells.items()}
    return {"trials": CAP_TRIALS, "n_foreign_tokens": CAP_FOREIGN, "quantile": CAP_QUANTILE, "by_token_count": out}


# ----------------------------------------------------------------------------------------------
# per-dataset driver
# ----------------------------------------------------------------------------------------------


def memory_bytes(key: AccKey, bundle: str, D: int) -> int:
    per = (D + 4) if bundle == "int8" else D // 8
    return acc_rows(key) * per


def row_record(cfg: RowCfg | None, mechanism: str, D: int | None, mem: int | None, madds: float | None,
               bitops: float | None, metrics: dict, timing: dict, label: str) -> dict:
    rec = {"label": label, "mechanism": mechanism, "dim": D,
           "token_map": cfg.key.token_map if cfg else None, "grouping": cfg.key.grouping if cfg else None,
           "n_groups": cfg.key.n_groups if cfg else None, "store": cfg.key.store if cfg else None,
           "bundle": cfg.bundle if cfg else None, "idf": cfg.key.idf if cfg else None,
           "expand": cfg.key.expand if cfg else None, "meansub": cfg.meansub if cfg else None,
           "memory_bytes_per_doc": mem, "memory_x_dense": None if mem is None else mem / DENSE_BYTES,
           "pass_memory": None if mem is None else bool(mem <= MEM_MULT * DENSE_BYTES),
           "madds_per_doc": madds, "bitops_per_doc": bitops, "metrics": metrics, **timing}
    fs = metrics["first_stage"]
    if mechanism in ("hli",) and "vs_ref_ndcg10_all" in fs:
        b = fs["vs_ref_ndcg10_all"]
        rec["pass_quality"] = bool(b["mean"] >= PASS_MARGIN and b["ci95"][0] > 0)
        rec["pass"] = bool(rec["pass_quality"] and rec["pass_memory"])
    else:
        rec["pass_quality"] = None
        rec["pass"] = None
    return rec


def run_dataset(name: str, cache_dir: Path, out_dir: Path, args, embedder: Embedder, R_full: np.ndarray,
                rff_full: np.ndarray) -> dict:
    log(f"== {name}: loading")
    ds = load_dataset(name, cache_dir, args.max_docs, args.seed)
    docs, docs_hit = bank_cached(out_dir, name, "docs", ds.doc_texts, embedder)
    queries, q_hit = bank_cached(out_dir, name, "queries", ds.q_texts, embedder)
    N, n_q = docs.n_items, queries.n_items
    log(f"{name}: {N} docs ({int(docs.off[-1])} tokens, cache_hit={docs_hit}), {n_q} queries "
        f"({int(queries.off[-1])} tokens, cache_hit={q_hit})")
    pooled_agreement = None
    if ds.cached_doc_vecs is not None and ds.cached_doc_vecs.shape == docs.pooled.shape:
        pooled_agreement = float((l2n(ds.cached_doc_vecs) * docs.pooled).sum(axis=1).mean())

    # evaluation scaffolding
    R = ds.R
    rel = R > 0
    idcg = roc.ideal_dcg(R)
    qidx = np.arange(n_q)
    rng_b = np.random.default_rng(BOOT_SEED)
    tune, evl = qidx[1::2], qidx[0::2]
    boot_all = rng_b.integers(0, n_q, size=(BOOT_N, n_q))
    boot_eval = rng_b.integers(0, len(evl), size=(BOOT_N, len(evl)))
    log(f"{name}: BM25")
    bm = roc.bm25_scores(roc.build_bm25(ds.doc_texts), ds.q_texts)
    bm_top = roc.topk(bm, CAND_K)
    cand = roc.topk_mask(bm_top, N, CAND_K, bm)
    S_dense_raw = docs.pooled @ queries.pooled.T
    dense_top = roc.topk(S_dense_raw.T.astype(np.float64), CAND_K)
    cand |= roc.topk_mask(dense_top, N, CAND_K)
    ev = Eval(R, rel, idcg, tune, evl, boot_all, boot_eval, cand, zrows(bm))
    cand_recall = float(((cand & rel).sum(axis=1) / rel.sum(axis=1)).mean())

    # token statistics, IDF, centring, expansion tables
    counts = docs.counts
    vocab = int(max(int(docs.ids.max()), int(queries.ids.max())) + 1)
    word_emb = None
    if args.additions:
        word_emb = word_embeddings_cached(out_dir, embedder)
        vocab = max(vocab, word_emb.shape[0])
    idf_w = wordpiece_idf(docs, vocab)
    mu = docs.tok.mean(axis=0)
    xcn = np.maximum(np.linalg.norm(docs.tok - mu[None, :], axis=1), 1e-6).astype(np.float32)
    qcn = np.maximum(np.linalg.norm(queries.tok - mu[None, :], axis=1), 1e-6).astype(np.float32)
    q_idf = idf_w[queries.ids]
    q_ones = np.ones(queries.off[-1], dtype=np.float32)
    expand_info = None
    expand_fn = None
    if args.additions:
        proto, nn_ids, nn_sim, expand_info = expansion_tables(docs, word_emb)
        expand_info["lambda"] = args.expand_weight

        def expand_fn(X: np.ndarray, ids: np.ndarray) -> np.ndarray:
            return expand_tokens(X, ids, proto, nn_ids, nn_sim, args.expand_weight)

    stats = {"n_docs": N, "n_queries": n_q, "n_queries_dropped_no_qrels": ds.n_queries_dropped,
             "qrels": ds.qrel_counts, "subsample": ds.subsample,
             "doc_tokens": {"mean": float(counts.mean()), "median": float(np.median(counts)), "max": int(counts.max()),
                            "min": int(counts.min()), "total": int(docs.off[-1]),
                            "frac_at_cap": docs.n_at_cap / N if docs.n_at_cap >= 0 else None},
             "query_tokens": {"mean": float(queries.counts.mean()), "max": int(queries.counts.max()),
                              "total": int(queries.off[-1])},
             "pooled_vs_cached_corpus_f32_cosine_mean": pooled_agreement, "embedding_cache_hit": docs_hit and q_hit,
             "rerank_candidates": {"mean_per_query": float(cand.sum(axis=1).mean()), "recall_ceiling": cand_recall},
             "split": {"tune": "odd query indices", "eval": "even query indices", "n_tune": int(len(tune)),
                       "n_eval": int(len(evl))}, "expansion": expand_info}

    rows: list[dict] = []
    tk = dict(threads=args.threads, timing=not args.no_timing, n_time=args.time_queries, repeats=args.time_repeats,
              warmup=1)
    mean_qt = float(queries.counts.mean())
    mean_dt = float(counts.mean())

    # dense baseline
    S, tm = timed(dense_scorer(queries.pooled, docs.pooled), n_q, **tk)
    dense_m = measure(S, ev)
    rows.append(row_record(None, "dense", None, DENSE_BYTES, float(HIDDEN), None,
                           compare(dense_m, None, ev, None, None), tm, "dense"))
    log(f"{name}: dense nDCG@10 {dense_m['first_stage']['ndcg10'].mean():.4f}")

    # BM25 alone (hybrid mode of this row is BM25 z-summed with itself; reported for the first stage only)
    bm_m = measure(bm.astype(np.float32), ev)
    rows.append(row_record(None, "bm25", None, None, None, None, compare(bm_m, dense_m, ev, None, None),
                           {"scoring_wall_s": None, "threads": None, "ms_per_query_1thread": None}, "bm25"))

    # MaxSim upper bounds
    mx_bytes = {"fp32": mean_dt * HIDDEN * 4, "fp16": mean_dt * HIDDEN * 2}
    mx_madds = mean_qt * mean_dt * HIDDEN
    S, tm = timed(maxsim_scorer(queries, docs.tok, docs.off, None), n_q, **tk)
    maxsim_m = measure(S, ev)
    rows.append(row_record(None, "maxsim", None, int(mx_bytes["fp16"]), mx_madds, None,
                           compare(maxsim_m, dense_m, ev, None, None), tm, "maxsim"))
    rows[-1]["memory_bytes_per_doc_fp32"] = mx_bytes["fp32"]
    log(f"{name}: maxsim nDCG@10 {maxsim_m['first_stage']['ndcg10'].mean():.4f} (wall {tm['scoring_wall_s']:.1f}s)")
    S, tm = timed(maxsim_scorer(queries, docs.tok, docs.off, q_idf), n_q, **tk)
    rows.append(row_record(None, "maxsim_idf", None, int(mx_bytes["fp16"]), mx_madds, None,
                           compare(measure(S, ev), dense_m, ev, maxsim_m, dense_m), tm, "maxsim|idf"))
    dtok_c = l2n(docs.tok - mu[None, :])
    qbank_c = TokenBank(l2n(queries.tok - mu[None, :]), queries.off, queries.ids, queries.pooled, queries.n_at_cap)
    S, tm = timed(maxsim_scorer(qbank_c, dtok_c, docs.off, None), n_q, **tk)
    rows.append(row_record(None, "maxsim_centered", None, int(mx_bytes["fp16"]), mx_madds, None,
                           compare(measure(S, ev), dense_m, ev, maxsim_m, dense_m), tm, "maxsim|centered"))
    del dtok_c, qbank_c, S

    # groupings
    groupings: dict[str, Grouping] = {}
    for B in args.buckets:
        groupings[f"pos{B}"] = make_grouping(f"pos{B}", docs.off, position_groups(docs.off, B), B)
    if args.additions:
        t = time.perf_counter()
        for K in sorted(set(args.kmeans) | {4}):
            groupings[f"kmeans{K}"] = make_grouping(f"kmeans{K}", docs.off, kmeans_groups(docs, K, KMEANS_SEED), K)
        log(f"{name}: k-means groupings in {time.perf_counter() - t:.1f}s")
    if 4 not in args.buckets and args.additions:
        groupings["pos4"] = make_grouping("pos4", docs.off, position_groups(docs.off, 4), 4)
    if 1 not in args.buckets and args.additions:
        groupings["pos1"] = make_grouping("pos1", docs.off, position_groups(docs.off, 1), 1)

    cfgs = build_rows(tuple(args.maps), tuple(args.buckets), tuple(args.kmeans), args.additions)
    capacity = {"real_tokens": {}, "synthetic": {}}
    rng_cap = np.random.default_rng(CAP_SEED)

    # resume: scored hli rows persist per dataset and are skipped when the fingerprint matches
    partial_path = out_dir / f"{name}.rows.partial.json"
    fingerprint = {"schema": SCHEMA, "model_revision": MODEL_REVISION, "seed": args.seed, "rff_gamma": args.rff_gamma,
                   "expand_weight": args.expand_weight, "max_docs": args.max_docs, "n_docs": N, "n_queries": n_q}
    done_labels: set[str] = set()
    if partial_path.exists():
        try:
            pdoc = json.loads(partial_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pdoc = None
        if isinstance(pdoc, dict) and pdoc.get("fingerprint") == fingerprint:
            for r in pdoc.get("rows", []):
                if r.get("mechanism") in ("hli", "hli_puresum") and r.get("label") not in done_labels:
                    rows.append(r)
                    done_labels.add(r["label"])
            log(f"{name}: resumed {len(done_labels)} scored rows from {partial_path.name}")

    def flush() -> None:
        kept = [r for r in rows if r["mechanism"] in ("hli", "hli_puresum")]
        partial_path.write_text(json.dumps({"fingerprint": fingerprint, "rows": roc.rounded(kept)}), encoding="utf-8")

    def cfg_labels(cfg: RowCfg, D: int) -> list[str]:
        labels = [f"{cfg.label}|D{D}"]
        if is_puresum_trigger(cfg):
            labels.append(f"{cfg.key.token_map}|puresum|int8|D{D}")
        return labels

    for D in args.dims:
        t_dim = time.perf_counter()
        R_D = np.ascontiguousarray(R_full[:, :D] / math.sqrt(D), dtype=np.float32)
        pj = Proj(D, R_D, (mu @ R_D).astype(np.float32), np.ascontiguousarray(rff_full[:D]), args.rff_gamma)
        Pq = (queries.tok @ pj.R).astype(np.float32)
        qhv: dict[tuple[str, str], np.ndarray] = {}

        def query_vectors(token_map: str, bundle: str) -> np.ndarray:
            # binarised real and sign queries coincide (sign of the same projection)
            k = ("sign", "bin") if bundle == "bin" and token_map in ("real", "sign") else (token_map, bundle)
            if k not in qhv:
                qhv[k] = query_hv(k[0], Pq, pj, qcn, k[1])
            return qhv[k]

        pending = [c for c in cfgs if any(lb not in done_labels for lb in cfg_labels(c, D))]
        keys: list[AccKey] = []
        for c in pending:
            if c.key not in keys:
                keys.append(c.key)
        log(f"{name} D={D}: {len(pending)} row configs pending, {len(cfgs) - len(pending)} resumed")
        budget = args.acc_budget_gb * (1 << 30)
        batches: list[list[AccKey]] = [[]]
        used = 0
        for k in keys:
            need = N * acc_rows(k) * D * 4
            if batches[-1] and used + need > budget:
                batches.append([])
                used = 0
            batches[-1].append(k)
            used += need
        batches = [b for b in batches if b]
        for bi, batch in enumerate(batches):
            t = time.perf_counter()
            accs = accumulate(pj, batch, docs, groupings, idf_w, xcn, expand_fn, args.token_chunk)
            log(f"{name} D={D}: batch {bi + 1}/{len(batches)} ({len(batch)} accumulators) bundled in "
                f"{time.perf_counter() - t:.1f}s")
            for cfg in pending:
                if cfg.key not in accs:
                    continue
                labels = cfg_labels(cfg, D)
                dm = prepare_matrix(accs[cfg.key], cfg, D)
                Hq = query_vectors(cfg.key.token_map, cfg.bundle)
                mem = memory_bytes(cfg.key, cfg.bundle, D)
                if labels[0] not in done_labels:
                    qw = q_idf if cfg.key.idf else q_ones
                    S, tm = timed(hli_scorer(Hq, queries, qw, dm, cfg.key, D, N), n_q, **tk)
                    ops = mean_qt * cfg.key.n_groups * D
                    rec = row_record(cfg, "hli", D, mem, None if cfg.bundle == "bin" else ops,
                                     ops if cfg.bundle == "bin" else None,
                                     compare(measure(S, ev), dense_m, ev, maxsim_m, dense_m), tm, labels[0])
                    rows.append(rec)
                    done_labels.add(labels[0])
                    flush()
                    fs = rec["metrics"]["first_stage"]
                    log(f"  {rec['label']:<42} nDCG@10 {fs['ndcg10']['all']:.4f} "
                        f"d={fs['vs_ref_ndcg10_all']['mean']:+.4f} "
                        f"ci=[{fs['vs_ref_ndcg10_all']['ci95'][0]:+.4f},{fs['vs_ref_ndcg10_all']['ci95'][1]:+.4f}] "
                        f"R@100 {fs['recall100']['all']:.4f} mem x{mem / DENSE_BYTES:.2f} "
                        f"{'PASS' if rec['pass'] else 'fail'}")
                    del S
                if len(labels) > 1 and labels[1] not in done_labels:
                    S, tm = timed(puresum_scorer(Hq, queries, dm, N), n_q, **tk)
                    rec = row_record(cfg, "hli_puresum", D, mem, float(D), None,
                                     compare(measure(S, ev), dense_m, ev, maxsim_m, dense_m), tm, labels[1])
                    rows.append(rec)
                    done_labels.add(labels[1])
                    flush()
                    fs = rec["metrics"]["first_stage"]
                    log(f"  {rec['label']:<42} nDCG@10 {fs['ndcg10']['all']:.4f} "
                        f"d={fs['vs_ref_ndcg10_all']['mean']:+.4f}")
                    del S
                del dm
            del accs
        capacity["real_tokens"][str(D)] = capacity_real(docs, pj, rng_cap)
        capacity["synthetic"][str(D)] = capacity_synthetic(pj, rng_cap)
        log(f"{name} D={D}: done in {time.perf_counter() - t_dim:.1f}s")

    verdict = dataset_verdict(rows, args.dims)
    for D in args.dims:
        v = verdict["per_dim"][str(D)]
        best = v["best"]
        print(f"{name} D={D}: {'PASS' if v['pass'] else 'FAIL'}  best memory-eligible row "
              f"{best['label'] if best else '-'} nDCG@10 {best['ndcg10'] if best else float('nan'):.4f} "
              f"delta {best['delta_vs_dense'] if best else float('nan'):+.4f} "
              f"gap_closed {best['gap_closed'] if best and best['gap_closed'] is not None else float('nan'):.2f}",
              flush=True)
    return {"stats": stats, "rows": rows, "capacity": capacity, "verdict": verdict}


def dataset_verdict(rows: list[dict], dims: tuple[int, ...]) -> dict:
    def summary(r: dict, mode: str, split: str) -> dict:
        m = r["metrics"][mode]
        boot = m.get("vs_ref_ndcg10_eval" if split == "eval" else "vs_ref_ndcg10_all")
        return {"label": r["label"], "dim": r["dim"], "ndcg10": m["ndcg10"][split], "recall100": m["recall100"][split],
                "delta_vs_dense": None if boot is None else boot["mean"],
                "ci95": None if boot is None else boot["ci95"], "gap_closed": m.get("gap_closed_all"),
                "memory_bytes_per_doc": r["memory_bytes_per_doc"], "memory_x_dense": r["memory_x_dense"]}

    hli = [r for r in rows if r["mechanism"] == "hli"]
    per_dim = {}
    for D in dims:
        cands = [r for r in hli if r["dim"] == D and r["pass_memory"]]
        best = max(cands, key=lambda r: r["metrics"]["first_stage"]["ndcg10"]["all"], default=None)
        per_dim[str(D)] = {"pass": bool(any(r["pass"] for r in cands)), "n_rows": len(cands),
                           "n_passing_rows": int(sum(1 for r in cands if r["pass"])),
                           "best": None if best is None else summary(best, "first_stage", "all"),
                           "rerank_any_pass": bool(any(
                               r["metrics"]["rerank"]["vs_ref_ndcg10_all"]["mean"] >= PASS_MARGIN
                               and r["metrics"]["rerank"]["vs_ref_ndcg10_all"]["ci95"][0] > 0 for r in cands))}
    best_by_tune = {}
    for mode in MODES:
        best_by_tune[mode] = {}
        for cname, pool in (("memory_le_4x_dense", [r for r in hli if r["pass_memory"]]), ("any_memory", hli)):
            b = max(pool, key=lambda r: r["metrics"][mode]["ndcg10"]["tune"], default=None)
            best_by_tune[mode][cname] = None if b is None else {"selected_on": "tune", "tune_ndcg10":
                                                                b["metrics"][mode]["ndcg10"]["tune"],
                                                                "eval": summary(b, mode, "eval")}
    return {"pass_any_dim": bool(any(v["pass"] for v in per_dim.values())), "per_dim": per_dim,
            "best_by_tune_split": best_by_tune}


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------


def protocol_text(args) -> dict:
    return {
        "tokens": f"{MODEL_ID}@{MODEL_REVISION}, last hidden state, max {MAX_SEQ} word pieces, [CLS]/[SEP] dropped "
                  "(kept only when nothing else remains), each token l2-normalised; pooled vector = the model's "
                  "mean pooling + l2 over all word pieces",
        "dense": "cosine of pooled vectors",
        "maxsim": "sum over query tokens of the max cosine over document tokens, divided by the query token count; "
                  "maxsim_idf weights query tokens by wordpiece idf; maxsim_centered subtracts the corpus-mean token "
                  "embedding from every token and renormalises",
        "projection": f"R (384 x D) with entries N(0, 1/D), the first D columns of one 384 x {max(DIMS)} draw "
                      f"(seed {args.seed}); RFF phases U[0, 2 pi) from the same generator; gamma {args.rff_gamma}",
        "token_maps": {"real": "R^T x", "real_c": "R^T (x - mu) / |x - mu|", "sign": "sign(R^T x) / sqrt(D)",
                       "rff": "sqrt(2/D) cos(sqrt(2 gamma D) R^T x + b)"},
        "grouping": "pos<B>: relative position bucket floor(p B / n); kmeans<K>: seeded spherical k-means on the "
                    f"document's token embeddings ({KMEANS_ITERS} iterations, seed {KMEANS_SEED}); sup: one D-vector, "
                    "group g bound by np.roll(., g D / G); sep: G stored D-vectors",
        "bundle": "int8: per-row symmetric int8 (scale = max|x| / 127), dequantised and l2-normalised; bin: sign of "
                  "the sum, query tokens binarised, score = +-1 dot / D (affine in the Hamming distance)",
        "score": "sum_t w_t max_g <key_g(h(q_t)), M_d> / (|M_d| sum_t w_t) for sup (G rolled query copies); "
                 "max over the G stored vectors (cosine each, empty groups masked) for sep; w_t = 1 or idf",
        "additions": {"A_idf": "token hypervectors weighted by BM25-style wordpiece idf before summing; query tokens "
                               "weighted by the same idf in the score",
                      "B_kmeans": "K semantic groups per document stored separately (memory K x D)",
                      "C_expand": f"x <- l2(x + lambda sum_i s_i p_i): p_i = corpus-mean contextual embedding of the "
                                  f"{EXPAND_TOP} occurring vocabulary items nearest to x's wordpiece in the model's "
                                  f"tied word-embedding matrix, s_i = their cosine (clipped at 0), lambda "
                                  f"{args.expand_weight}; documents only",
                      "D_rerank": f"candidates = dense top-{CAND_K} union positive-BM25 top-{CAND_K}; non-candidates "
                                  "unretrieved; reported as metrics.rerank for every row",
                      "E_meansub": "M_{d,g} -= W_{d,g} h_bar, h_bar = weight-mean token hypervector over the corpus"},
        "hybrid": "metrics.hybrid_bm25 = per-query z-score of the row + z-score of BM25 (k1 1.2, b 0.75, WARM tokeniser "
                  "from benchmarks/research/roc/roc.py)",
        "metrics": "nDCG@10 with linear gain over all judged grades, recall@100 over grade > 0, lower document index "
                   "wins ties; all queries plus the odd (tune) / even (eval) split",
        "bootstrap": f"paired percentile bootstrap of the mean nDCG@10 difference, {BOOT_N} resamples, seed {BOOT_SEED}, "
                     "one shared index matrix per dataset (all) and per eval split",
        "memory": f"dense {DENSE_BYTES} B; maxsim tokens x 384 x 2 (fp16; fp32 also recorded); int8 bundle G x (D + 4); "
                  "binary G x D / 8",
        "cost": "madds_per_doc = mean query tokens x G x D (dense 384, maxsim q x d x 384, puresum D); "
                "bitops_per_doc for binary rows (XOR + popcount over G x D bits per query token)",
        "verdict": f"per dataset and D: PASS iff an hli row with memory <= {MEM_MULT:g} x dense beats dense on nDCG@10 "
                   f"by >= {PASS_MARGIN} with 95% CI excluding 0 (first stage, all queries); overall PASS iff some D "
                   "passes on every dataset; best_by_tune_split selects on the tune split and reports the eval split",
        "timing": "scoring_wall_s = one full pass at --threads; ms_per_query_1thread = median of --time-repeats "
                  "repeats after one warm-up on --time-queries queries with torch threads = 1",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache-root", type=Path, default=Path.home() / ".cache/warm-bench")
    ap.add_argument("--datasets", nargs="+", default=["nfcorpus", "scifact"])
    ap.add_argument("--out", type=Path, default=HERE / "out/hli_v1.json")
    ap.add_argument("--dims", nargs="+", type=int, default=list(DIMS))
    ap.add_argument("--buckets", nargs="+", type=int, default=list(BUCKETS))
    ap.add_argument("--kmeans", nargs="+", type=int, default=list(KMEANS_KS))
    ap.add_argument("--maps", nargs="+", default=list(TOKEN_MAPS), choices=TOKEN_MAPS)
    ap.add_argument("--rff-gamma", type=float, default=4.0)
    ap.add_argument("--expand-weight", type=float, default=1.0)
    ap.add_argument("--no-additions", dest="additions", action="store_false", help="skip additions A, B, C, E")
    ap.add_argument("--threads", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--batch-size", type=int, default=64, help="embedding batch size")
    ap.add_argument("--token-chunk", type=int, default=8192)
    ap.add_argument("--acc-budget-gb", type=float, default=6.0)
    ap.add_argument("--no-timing", action="store_true", help="skip the single-thread micro-benchmark")
    ap.add_argument("--time-queries", type=int, default=4)
    ap.add_argument("--time-repeats", type=int, default=5)
    ap.add_argument("--max-docs", type=int, default=0, help="subsample a corpus to judged + random docs (0 = all)")
    ap.add_argument("--seed", type=int, default=20241020)
    args = ap.parse_args()
    args.dims = tuple(sorted(set(args.dims), reverse=True))  # largest D first: the capacity question comes first
    args.buckets = tuple(sorted(set(args.buckets)))
    args.kmeans = tuple(sorted(set(args.kmeans)))
    if max(args.dims) > max(DIMS):
        raise SystemExit(f"--dims above {max(DIMS)} are not supported by the nested projection")
    if any(d % b for d in args.dims for b in args.buckets):
        raise SystemExit("every dim must be a multiple of every bucket count")

    import torch

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    embedder = Embedder(args.threads, args.batch_size)
    rng = np.random.default_rng(args.seed)
    R_full = rng.standard_normal((HIDDEN, max(DIMS))).astype(np.float32)
    rff_full = rng.uniform(0.0, 2.0 * math.pi, size=max(DIMS)).astype(np.float32)

    versions = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}
    try:
        import sentence_transformers

        versions["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        versions["sentence_transformers"] = None
    doc = {"schema": SCHEMA, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "command": " ".join(sys.argv), "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "max_seq_length": MAX_SEQ,
                                                    "hidden": HIDDEN},
           "machine": {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
                       "threads": args.threads, "load_average_1m": os.getloadavg()[0], **versions},
           "config": {"dims": list(args.dims), "buckets": list(args.buckets), "kmeans": list(args.kmeans),
                      "maps": list(args.maps), "rff_gamma": args.rff_gamma, "expand_weight": args.expand_weight,
                      "additions": args.additions, "seed": args.seed, "token_chunk": args.token_chunk,
                      "max_docs": args.max_docs, "timing": not args.no_timing, "time_queries": args.time_queries,
                      "time_repeats": args.time_repeats, "pass_margin": PASS_MARGIN, "memory_multiple": MEM_MULT,
                      "dense_bytes": DENSE_BYTES},
           "protocol": protocol_text(args), "datasets": {}}
    for name in args.datasets:
        cache_dir = args.cache_root / name
        if not (cache_dir / "corpus.jsonl").exists():
            raise SystemExit(f"{cache_dir}: corpus.jsonl missing")
        doc["datasets"][name] = run_dataset(name, cache_dir, out_dir, args, embedder, R_full, rff_full)
        doc["datasets"][name]["wall_s_so_far"] = time.perf_counter() - T0
        args.out.write_text(json.dumps(roc.rounded(doc), indent=1), encoding="utf-8")
    per_dim = {}
    for D in args.dims:
        passes = {n: doc["datasets"][n]["verdict"]["per_dim"][str(D)]["pass"] for n in args.datasets}
        per_dim[str(D)] = {"pass_all_datasets": bool(all(passes.values())), "by_dataset": passes}
    doc["verdict"] = {"per_dim": per_dim, "overall_pass": bool(any(v["pass_all_datasets"] for v in per_dim.values())),
                      "wall_s": time.perf_counter() - T0}
    args.out.write_text(json.dumps(roc.rounded(doc), indent=1), encoding="utf-8")
    for D in args.dims:
        v = per_dim[str(D)]
        print(f"D={D}: {'PASS' if v['pass_all_datasets'] else 'FAIL'} ({v['by_dataset']})")
    print(f"OVERALL: {'PASS' if doc['verdict']['overall_pass'] else 'FAIL'}  ({doc['verdict']['wall_s']:.0f}s) -> {args.out}")


if __name__ == "__main__":
    main()
