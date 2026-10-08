"""BEIR proxies on NFCorpus dev qrels and SciFact train qrels only (the test qrels are never
read here). BM25 reproduces the document API (k1 1.2, b 0.75, idf ln(1 + (n - df + 0.5) /
(df + 0.5)), lowercase split on non-alphanumeric); RRF reproduces its fusion (1 / (60 + rank + 1)).
Usage: uv run --with sentence-transformers --with PyStemmer --with pytrec-eval-terrier python
       beir_proxies.py <stemming|fusion> ..."""
from __future__ import annotations

import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CACHE = Path.home() / ".cache/hms-bench"
OUT = ROOT / "benchmarks/results/proxies_2026-10_beir.json"
MODEL, REV = "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
STOP = set("a an and are as at be by for from has he in is it its of on that the to was were will with this these those or not but if than then so such which who whom whose what when where why how do does did can could should would may might must shall i you we they them their our your his her she him my me us".split())


def write(name, payload):
    doc = json.loads(OUT.read_text()) if OUT.exists() else {"note": "BEIR proxies on NFCorpus dev / SciFact train qrels only; test qrels unread"}
    doc[name] = payload
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print("wrote", name)


def load_split(name, split):
    d = CACHE / "downloads" / name
    qrels = {}
    with (d / "qrels" / f"{split}.tsv").open() as f:
        next(f)
        for line in f:
            qid, did, rel = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[did] = int(rel)
    corpus = []
    with (d / "corpus.jsonl").open() as f:
        for line in f:
            x = json.loads(line)
            corpus.append((x["_id"], f"{x.get('title', '')}\n{x['text']}".strip()))
    queries = []
    with (d / "queries.jsonl").open() as f:
        for line in f:
            x = json.loads(line)
            if x["_id"] in qrels:
                queries.append((x["_id"], x["text"]))
    return corpus, queries, qrels


def tok_hms(s):
    return [t for t in re.split(r"[^0-9A-Za-zÀ-￿]+", s.lower()) if t]


class BM25:
    def __init__(self, docs, tok):
        self.tok = tok
        self.tf = [Counter(tok(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg = self.len.mean()
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.post = {}
        for i, c in enumerate(self.tf):
            for w, t in c.items():
                self.post.setdefault(w, []).append((i, t))

    def score(self, q):
        s = np.zeros(len(self.tf))
        for w in set(self.tok(q)):
            if w not in self.post:
                continue
            idf = self.idf[w]
            for i, tf in self.post[w]:
                s[i] += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * self.len[i] / self.avg))
        return s


def ndcg10(run, qrels):
    import pytrec_eval
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.100"})
    r = ev.evaluate(run)
    return float(np.mean([v["ndcg_cut_10"] for v in r.values()])), float(np.mean([v["recall_100"] for v in r.values()]))


def run_from_scores(qids, dids, S, k=100):
    run = {}
    for qi, qid in enumerate(qids):
        top = np.argsort(-S[qi], kind="stable")[:k]
        run[qid] = {dids[i]: float(S[qi][i]) for i in top}
    return run


def stemming(sets):
    import Stemmer
    st = Stemmer.Stemmer("english")
    res = {}
    for name, split in sets:
        corpus, queries, qrels = load_split(name, split)
        dids = [d for d, _ in corpus]
        qids = [q for q, _ in queries]
        variants = {
            "hms_tokenizer": tok_hms,
            "stop_words": lambda s: [t for t in tok_hms(s) if t not in STOP],
            "porter": lambda s: st.stemWords(tok_hms(s)),
            "porter_stop": lambda s: st.stemWords([t for t in tok_hms(s) if t not in STOP]),
        }
        res[f"{name}_{split}"] = {"n_queries": len(qids), "n_docs": len(dids)}
        for v, tok in variants.items():
            t = time.time()
            bm = BM25([x for _, x in corpus], tok)
            S = np.stack([bm.score(q) for _, q in queries])
            n, r = ndcg10(run_from_scores(qids, dids, S), qrels)
            res[f"{name}_{split}"][v] = {"ndcg@10": round(n, 4), "recall@100": round(r, 4), "secs": round(time.time() - t)}
            print(name, split, v, n, r)
    return {"question": "Does Porter stemming and/or a stop list close BM25's gap to the published BEIR BM25 (which stems)?", "results": res,
            "decision_rule": "adopt stemming in the document API only if nDCG@10 rises >= 0.01 on both dev/train sets; a stop list alone is adopted only if it does not lose on either"}


def fusion(sets):
    from sentence_transformers import SentenceTransformer
    import Stemmer
    st = Stemmer.Stemmer("english")
    model = SentenceTransformer(MODEL, revision=REV, device="cpu")
    res = {}
    for name, split in sets:
        corpus, queries, qrels = load_split(name, split)
        dids = [d for d, _ in corpus]
        qids = [q for q, _ in queries]
        emb_path = CACHE / name / "corpus.f32"
        D = np.fromfile(emb_path, dtype=np.float32).reshape(len(dids), -1)
        Q = model.encode([q for _, q in queries], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        dense = Q @ D.T
        bm = BM25([x for _, x in corpus], lambda s: st.stemWords([t for t in tok_hms(s) if t not in STOP]))
        lex = np.stack([bm.score(q) for _, q in queries])
        out = {"n_queries": len(qids)}
        out["dense_only"] = ndcg10(run_from_scores(qids, dids, dense), qrels)[0]
        out["bm25_porter_stop_only"] = ndcg10(run_from_scores(qids, dids, lex), qrels)[0]
        rk_d = np.argsort(np.argsort(-dense, axis=1, kind="stable"), axis=1)
        rk_l = np.argsort(np.argsort(-lex, axis=1, kind="stable"), axis=1)
        for k0 in (10, 30, 60, 100):
            F = 1 / (k0 + rk_d + 1) + 1 / (k0 + rk_l + 1)
            out[f"rrf_k0_{k0}"] = ndcg10(run_from_scores(qids, dids, F), qrels)[0]
        for w in (0.5, 1.0, 2.0):
            F = 1 / (60 + rk_d + 1) + w / (60 + rk_l + 1)
            out[f"rrf60_lex_weight_{w}"] = ndcg10(run_from_scores(qids, dids, F), qrels)[0]

        def mm(S, top=100):
            # per-query min-max over the top-`top` scores (the rest get 0), as a blend candidate
            idx = np.argsort(-S, axis=1, kind="stable")[:, :top]
            hi = np.take_along_axis(S, idx[:, :1], axis=1)
            lo = np.take_along_axis(S, idx[:, -1:], axis=1)
            N = np.clip((S - lo) / np.maximum(hi - lo, 1e-9), 0, 1)
            return N
        Nd, Nl = mm(dense), mm(lex)
        for a in (0.3, 0.5, 0.7):
            out[f"minmax_blend_dense_{a}"] = ndcg10(run_from_scores(qids, dids, a * Nd + (1 - a) * Nl), qrels)[0]

        def z(S, top=100):
            idx = np.argsort(-S, axis=1, kind="stable")[:, :top]
            T = np.take_along_axis(S, idx, axis=1)
            return (S - T.mean(axis=1, keepdims=True)) / np.maximum(T.std(axis=1, keepdims=True), 1e-9)
        Zd, Zl = z(dense), z(lex)
        for a in (0.3, 0.5, 0.7):
            out[f"zscore_blend_dense_{a}"] = ndcg10(run_from_scores(qids, dids, a * Zd + (1 - a) * Zl), qrels)[0]
        res[f"{name}_{split}"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
        print(json.dumps(res[f"{name}_{split}"], indent=0))
    return {"question": "Is RRF with k0 = 60 and equal weights the best fusion of BM25 and dense, or does a calibrated blend (min-max, z-score) or a re-weighted RRF win on dev/train qrels?", "results": res,
            "decision_rule": "change the fusion only if one setting beats rrf_k0_60 by >= 0.01 nDCG@10 on both sets; the winner is then measured once on the test splits via evaluate.py beir"}


if __name__ == "__main__":
    what = sys.argv[1]
    sets = [("nfcorpus", "dev"), ("scifact", "train")]
    t = time.time()
    r = stemming(sets) if what == "stemming" else fusion(sets)
    r["minutes"] = round((time.time() - t) / 60, 1)
    write(what, r)
