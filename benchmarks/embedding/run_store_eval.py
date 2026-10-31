# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Retrieval gain and cost of `TextEncoder::LocalEmbedV1` against the lexical default, through the
real document API (benchmarks/embedding/embed_store_bench.rs).

For each dataset (SciFact train queries, NFCorpus dev queries, LongMemEval dev sessions) the bench
binary builds the store(s), runs the lexical / semantic / hybrid variants and writes ranked ids;
this script scores them with the same nDCG@10 / recall_any@5 as the model selection and wraps each
process in /usr/bin/time for peak RSS. Cold start is the wall time of a one-document store.

  uv run --no-project --with numpy python benchmarks/embedding/run_store_eval.py \
      --binary $TARGET/release/embed-store-bench --model gte-small --out result.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_model import beir, lme_dev, ndcg_at  # noqa: E402

WORK = "./.warm-embed-work"


def run(binary, task, encoder, model, threads, work):
    out = Path(work) / f"{encoder}.out.json"
    cmd = [binary, "--task", str(task), "--out", str(out), "--encoder", encoder, "--model", model,
           "--threads", str(threads)]
    time_cmd = ["/usr/bin/time", "-l"] if sys.platform == "darwin" else ["/usr/bin/time", "-v"]
    t0 = time.time()
    p = subprocess.run(time_cmd + cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    m = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
    rss = int(m.group(1)) if m else None
    if rss is None:
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.stderr)
        rss = int(m.group(1)) * 1024 if m else None
    res = json.loads(out.read_text())
    res["wall_seconds"] = wall
    res["peak_rss_bytes"] = rss
    return res


def beir_task(name, split, work):
    ids, texts, qids, qtexts, qrels = beir(name, split)
    task = {"stores": [{"docs": [{"id": i, "text": t} for i, t in zip(ids, texts)],
                        "queries": [{"id": q, "text": t} for q, t in zip(qids, qtexts)]}]}
    path = Path(work) / f"{name}-{split}.task.json"
    path.write_text(json.dumps(task))
    return path, qrels


def score_beir(result, qrels):
    out = {}
    runs = result["runs"][0]
    for variant in next(iter(runs.values())):
        nd = [ndcg_at(per[variant], qrels[q]) for q, per in runs.items()]
        out[variant] = {"ndcg@10": float(np.mean(nd)), "queries": len(nd)}
    return out


def lme_task(work):
    rows = lme_dev()
    stores, meta = [], []
    for qid, question, sessions, answers in rows:
        docs, owner = [], {}
        for sid, turns in sessions:
            for i, t in enumerate(turns):
                did = f"{sid}#{i}"
                docs.append({"id": did, "text": t})
                owner[did] = sid
        stores.append({"docs": docs, "queries": [{"id": qid, "text": question}]})
        meta.append((qid, owner, answers, [s for s, _ in sessions]))
    path = Path(work) / "lme-dev.task.json"
    path.write_text(json.dumps({"stores": stores}))
    return path, meta


def score_lme(result, meta):
    out = {}
    variants = next(iter(result["runs"][0].values())).keys()
    for variant in variants:
        r5, nd = [], []
        for run, (qid, owner, answers, sessions) in zip(result["runs"], meta):
            ranked_docs = run[qid][variant]
            ranked, seen = [], set()
            for d in ranked_docs:
                s = owner[d]
                if s not in seen:
                    seen.add(s)
                    ranked.append(s)
            # sessions never retrieved rank after every retrieved one, in haystack order
            ranked += [s for s in sessions if s not in seen]
            r5.append(1.0 if answers & set(ranked[:5]) else 0.0)
            nd.append(ndcg_at(ranked, {a: 1 for a in answers}))
        out[variant] = {"recall_any@5": float(np.mean(r5)), "ndcg@10": float(np.mean(nd)), "questions": len(r5)}
    return out


def summarize(res):
    return {k: res[k] for k in ("documents", "ingest_seconds", "ingest_docs_per_second", "query_latency",
                                "wall_seconds", "peak_rss_bytes")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", default="default")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="*", default=["scifact", "nfcorpus", "lme-dev"])
    a = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    out = {"model": a.model, "threads": a.threads, "load_average_1m": os.getloadavg()[0], "datasets": {}}
    with tempfile.TemporaryDirectory(dir=WORK) as work:
        # one-document store: process start to a working store (cold start with and without the embedder)
        tiny = Path(work) / "tiny.task.json"
        tiny.write_text(json.dumps({"stores": [{"docs": [{"id": "a", "text": "hello world"}],
                                                "queries": [{"id": "q", "text": "hello"}]}]}))
        out["cold_start"] = {enc: min((run(a.binary, tiny, enc, a.model, a.threads, work) for _ in range(3)),
                                      key=lambda r: r["wall_seconds"])
                             for enc in ("default", "local-embed-v1")}
        out["cold_start"] = {k: {"wall_seconds": v["wall_seconds"], "peak_rss_bytes": v["peak_rss_bytes"]}
                             for k, v in out["cold_start"].items()}
        for ds in a.datasets:
            if ds == "lme-dev":
                task, meta = lme_task(work)
                score = lambda r: score_lme(r, meta)  # noqa: E731
            else:
                name, split = {"scifact": ("scifact", "train"), "nfcorpus": ("nfcorpus", "dev")}[ds]
                task, qrels = beir_task(name, split, work)
                score = lambda r: score_beir(r, qrels)  # noqa: E731
            base = run(a.binary, task, "default", a.model, a.threads, work)
            local = run(a.binary, task, "local-embed-v1", a.model, a.threads, work)
            out["datasets"][ds] = {
                "default_encoder": {"scores": score(base), **summarize(base)},
                "local_embed_v1": {"scores": score(local), **summarize(local)},
            }
            print(ds, json.dumps({k: v["scores"] for k, v in out["datasets"][ds].items()}), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
