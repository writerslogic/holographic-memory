# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Stage 1 of the local embedding model selection: fp32 retrieval quality on dev splits.

Datasets (train/dev only; the held-out LongMemEval ids and every test split are never read):
  scifact   BEIR SciFact, queries of qrels/train.tsv, nDCG@10 over the full corpus
  nfcorpus  BEIR NFCorpus, queries of qrels/dev.tsv, nDCG@10 over the full corpus
  lme-dev   LongMemEval_S session retrieval on the frozen dev ids of
            benchmarks/public/longmemeval_split.json (non-abstention questions): turn-level
            embeddings of both roles, session score = best turn over the question's own
            haystack; recall_any@5 and nDCG@10

Licence and size come from the Hugging Face API at the pinned revision; the reference model runs
with sentence-transformers on the PyTorch CPU/MPS device in fp32. Stage 2 (int8 engine quality,
latency, RSS) is benchmarks/embedding/run_engine_eval.py through the Rust engine.

Run (outside the repo environment):
  uv run --no-project --with sentence-transformers --with numpy \
    python benchmarks/embedding/select_model.py --out /tmp/stage1.json
"""
import argparse
import hashlib
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DL = Path.home() / ".cache/hms-bench/downloads"
Q_BGE = "Represent this sentence for searching relevant passages: "

# id -> (query prefix, passage prefix, engine-supported architecture)
CANDIDATES = {
    "sentence-transformers/all-MiniLM-L6-v2": ("", "", True),
    "sentence-transformers/all-MiniLM-L12-v2": ("", "", True),
    "BAAI/bge-small-en-v1.5": (Q_BGE, "", True),
    "thenlper/gte-small": ("", "", True),
    "intfloat/e5-small-v2": ("query: ", "passage: ", True),
    "intfloat/multilingual-e5-small": ("query: ", "passage: ", True),
    "Snowflake/snowflake-arctic-embed-s": (Q_BGE, "", True),
    "Snowflake/snowflake-arctic-embed-xs": (Q_BGE, "", True),
    "mixedbread-ai/mxbai-embed-xsmall-v1": (Q_BGE, "", True),
    "avsolatorio/GIST-small-Embedding-v0": (Q_BGE, "", True),
    "ibm-granite/granite-embedding-30m-english": ("", "", True),
    "ibm-granite/granite-embedding-small-english-r2": ("", "", False),
}


def hf_info(model_id):
    url = f"https://huggingface.co/api/models/{model_id}"
    info = json.load(urllib.request.urlopen(url, timeout=60))
    card = dict(info.get("cardData") or {})
    if not card.get("license"):
        tagged = [t.split(":", 1)[1] for t in info.get("tags", []) if t.startswith("license:")]
        card["license"] = tagged[0] if tagged else None
    safetensors = (info.get("safetensors") or {}).get("total")
    return {
        "revision": info["sha"],
        "license": card.get("license"),
        "parameters": safetensors,
        "pipeline_tag": info.get("pipeline_tag"),
    }


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def beir(name, split):
    base = DL / name
    corpus = read_jsonl(base / "corpus.jsonl")
    queries = {q["_id"]: q["text"] for q in read_jsonl(base / "queries.jsonl")}
    qrels = {}
    with open(base / "qrels" / f"{split}.tsv") as f:
        next(f)
        for line in f:
            q, d, s = line.rstrip("\n").split("\t")
            if int(s) > 0:
                qrels.setdefault(q, {})[d] = int(s)
    ids = [d["_id"] for d in corpus]
    texts = [((d.get("title") or "") + " " + (d.get("text") or "")).strip() for d in corpus]
    qids = [q for q in qrels if q in queries]
    return ids, texts, qids, [queries[q] for q in qids], qrels


def ndcg_at(ranked, rel, k=10):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def eval_beir(model, qp, pp, name, split, max_len, cache, batch):
    ids, texts, qids, qtexts, qrels = beir(name, split)
    doc = embed(model, texts, pp, cache / f"{name}-docs.npy", batch)
    qry = embed(model, qtexts, qp, cache / f"{name}-{split}-q.npy", batch)
    scores = qry @ doc.T
    top = np.argsort(-scores, axis=1)[:, :10]
    ndcg = [ndcg_at([ids[j] for j in row], qrels[q]) for q, row in zip(qids, top)]
    return {"queries": len(qids), "corpus": len(ids), "ndcg@10": float(np.mean(ndcg))}


def lme_dev():
    split = json.loads(Path(ROOT / "benchmarks/public/longmemeval_split.json").read_text())
    dev = set(split["dev"])
    data = json.loads(Path(DL / "longmemeval_s_cleaned.json").read_text())
    rows = []
    for item in data:
        qid = item["question_id"]
        if qid not in dev or qid.endswith("_abs"):
            continue
        sessions = []
        for sid, turns in zip(item["haystack_session_ids"], item["haystack_sessions"]):
            sessions.append((sid, [t["content"] for t in turns if t["content"].strip()]))
        rows.append((qid, item["question"], sessions, set(item["answer_session_ids"])))
    return rows


def eval_lme(model, qp, pp, cache, batch):
    """Turn-level retrieval (both roles), session score = best turn, as the repo's tuned
    LongMemEval configuration does; no session is truncated to the model window."""
    rows = lme_dev()
    uniq = {}
    for _, _, sessions, _ in rows:
        for sid, turns in sessions:
            uniq.setdefault(sid, turns)
    sids = list(uniq)
    flat = [t for s in sids for t in uniq[s]]
    owner = np.array([i for i, s in enumerate(sids) for _ in uniq[s]])
    emb = embed(model, flat, pp, cache / "lme-turns.npy", batch)
    pos = {s: i for i, s in enumerate(sids)}
    qry = embed(model, [r[1] for r in rows], qp, cache / "lme-q.npy", batch)
    recall5, ndcg = [], []
    for (qid, _, sessions, answers), qv in zip(rows, qry):
        scores = emb @ qv
        best = {}
        for sid, _ in sessions:
            m = owner == pos[sid]
            best[sid] = float(scores[m].max()) if m.any() else -1.0
        ranked = sorted(best, key=lambda s: -best[s])
        recall5.append(1.0 if answers & set(ranked[:5]) else 0.0)
        ndcg.append(ndcg_at(ranked, {a: 1 for a in answers}))
    return {
        "questions": len(rows),
        "unique_sessions": len(sids),
        "turns": len(flat),
        "recall_any@5": float(np.mean(recall5)),
        "ndcg@10": float(np.mean(ndcg)),
    }


def embed(model, texts, prefix, path, batch):
    key = hashlib.sha256(("\0".join(texts) + prefix + str(model.max_seq_length)).encode()).hexdigest()[:16]
    path = path.with_name(path.stem + "-" + key + ".npy")
    if path.exists():
        return np.load(path)
    out = model.encode([prefix + t for t in texts], batch_size=batch, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    np.save(path, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="*", default=list(CANDIDATES))
    ap.add_argument("--cache", type=Path, default=Path("/tmp/hms-embed-select"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer

    results = json.loads(Path(args.out).read_text()) if args.out.exists() else {}
    for mid in args.models:
        if mid in results:
            continue
        qp, pp, engine = CANDIDATES[mid]
        info = hf_info(mid)
        entry = {"info": info, "engine_architecture": engine, "query_prefix": qp, "passage_prefix": pp}
        lic = (info["license"] or "").lower()
        entry["redistributable"] = lic in ("mit", "apache-2.0")
        if not entry["redistributable"]:
            entry["skipped"] = f"licence {info['license']!r} is not MIT/Apache-2.0"
            results[mid] = entry
            continue
        cache = args.cache / mid.replace("/", "__")
        cache.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            model = SentenceTransformer(mid, revision=info["revision"], device=args.device)
            max_len = min(model.max_seq_length or 512, 512)
            model.max_seq_length = max_len
            entry["max_seq_length"] = max_len
            entry["dimensions"] = model.get_embedding_dimension()
            entry["scifact_train"] = eval_beir(model, qp, pp, "scifact", "train", max_len, cache, args.batch)
            entry["nfcorpus_dev"] = eval_beir(model, qp, pp, "nfcorpus", "dev", max_len, cache, args.batch)
            entry["lme_dev"] = eval_lme(model, qp, pp, cache, args.batch)
        except Exception as e:  # recorded, not hidden
            entry["error"] = f"{type(e).__name__}: {e}"[:500]
        entry["eval_seconds"] = round(time.time() - t0, 1)
        results[mid] = entry
        args.out.write_text(json.dumps(results, indent=1))
        print(mid, {k: entry.get(k) for k in ("scifact_train", "nfcorpus_dev", "lme_dev", "error")}, flush=True)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    sys.exit(main())
