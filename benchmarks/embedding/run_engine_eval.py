# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Stage 2 of the model selection: the Rust engine on a model pack.

For one pack (f32, f16 or int8) this measures, through `warm-embed embed`:
  quality   SciFact-train and NFCorpus-dev nDCG@10 and LongMemEval-dev session recall/nDCG
            (same data and metrics as select_model.py, from the engine's own vectors)
  latency   ms per passage and per query at 1 thread and at N threads (best of --repeats)
  rss       peak resident set size of the process embedding 2000 passages (/usr/bin/time)
  cold      process start to first embedding of one text (pack hash + parse + tokenizer + pool)

Machine load is recorded with every run; on a shared machine treat absolute times as indicative
and compare models within one run.

  uv run --no-project --with numpy python benchmarks/embedding/run_engine_eval.py \
      --binary $TARGET/release/warm-embed --model-id BAAI/bge-small-en-v1.5 --pack bge-small-int8.hmsemb \
      --query-prefix "Represent this sentence for searching relevant passages: " --out result.json
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


def embed(binary, pack, texts, kind, threads, workdir, max_length=None, timed=False):
    """Returns (matrix, timings dict, peak_rss_bytes)."""
    inp = Path(workdir) / "in.jsonl"
    out = Path(workdir) / "out.f32"
    inp.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in texts))
    cmd = [binary, "embed", "--model", str(pack), "--raw-out", str(out), "--input", str(inp),
           "--json-lines", "--threads", str(threads), "--timings"]
    if kind == "query":
        cmd.append("--query")
    if max_length:
        cmd += ["--max-length", str(max_length)]
    time_cmd = ["/usr/bin/time", "-l"] if sys.platform == "darwin" else ["/usr/bin/time", "-v"]
    p = subprocess.run(time_cmd + cmd, capture_output=True, text=True, check=True)
    timing = json.loads(next(l for l in p.stderr.splitlines() if l.startswith("{")))
    m = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
    if m:
        rss = int(m.group(1))
    else:
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.stderr)
        rss = int(m.group(1)) * 1024 if m else None
    mat = np.fromfile(out, dtype="<f4").reshape(len(texts), -1)
    return mat, timing, rss


def quality(binary, pack, qp, pp, threads, work, max_length, lme=True):
    res = {}
    for name, split in (("scifact_train", ("scifact", "train")), ("nfcorpus_dev", ("nfcorpus", "dev"))):
        ids, texts, qids, qtexts, qrels = beir(*split)
        d, _, _ = embed(binary, pack, texts, "passage", threads, work, max_length)
        q, _, _ = embed(binary, pack, qtexts, "query", threads, work, max_length)
        top = np.argsort(-(q @ d.T), axis=1)[:, :10]
        nd = [ndcg_at([ids[j] for j in row], qrels[qq]) for qq, row in zip(qids, top)]
        res[name] = {"queries": len(qids), "corpus": len(ids), "ndcg@10": float(np.mean(nd)),
                     "per_query_ndcg": [round(float(x), 6) for x in nd]}
    if not lme:
        return res
    rows = lme_dev()
    uniq = {}
    for _, _, sessions, _ in rows:
        for sid, turns in sessions:
            uniq.setdefault(sid, turns)
    sids = list(uniq)
    flat = [t for s in sids for t in uniq[s]]
    owner = np.array([i for i, s in enumerate(sids) for _ in uniq[s]])
    e, _, _ = embed(binary, pack, flat, "passage", threads, work, max_length)
    qv, _, _ = embed(binary, pack, [r[1] for r in rows], "query", threads, work, max_length)
    pos = {s: i for i, s in enumerate(sids)}
    r5, nd = [], []
    for (qid, _, sessions, answers), v in zip(rows, qv):
        sc = e @ v
        best = {sid: float(sc[owner == pos[sid]].max()) if (owner == pos[sid]).any() else -1.0 for sid, _ in sessions}
        ranked = sorted(best, key=lambda s: -best[s])
        r5.append(1.0 if answers & set(ranked[:5]) else 0.0)
        nd.append(ndcg_at(ranked, {a: 1 for a in answers}))
    res["lme_dev"] = {"questions": len(rows), "turns": len(flat), "recall_any@5": float(np.mean(r5)),
                      "ndcg@10": float(np.mean(nd)), "per_question_ndcg": [round(float(x), 6) for x in nd],
                      "per_question_recall5": r5}
    return res


def speed(binary, pack, threads_list, repeats, work, max_length):
    ids, texts, qids, qtexts, _ = beir("scifact", "train")
    passages = texts[:500]
    queries = qtexts[:500]
    res = {"load_average_1m": os.getloadavg()[0]}
    for threads in threads_list:
        best_p = best_q = None
        for _ in range(repeats):
            _, t, _ = embed(binary, pack, passages, "passage", threads, work, max_length)
            best_p = t if best_p is None or t["ms_per_text"] < best_p["ms_per_text"] else best_p
            _, t, _ = embed(binary, pack, queries, "query", threads, work, max_length)
            best_q = t if best_q is None or t["ms_per_text"] < best_q["ms_per_text"] else best_q
        res[f"threads_{threads}"] = {"passage_ms_each": best_p["ms_per_text"], "query_ms_each": best_q["ms_per_text"],
                                     "passages": len(passages), "queries": len(queries)}
    _, t, rss = embed(binary, pack, texts[:2000], "passage", max(threads_list), work, max_length)
    res["peak_rss_bytes_2000_passages"] = rss
    colds = []
    for _ in range(repeats):
        t0 = time.time()
        _, t, rss1 = embed(binary, pack, ["hello world"], "query", max(threads_list), work, max_length)
        colds.append({"wall_ms": (time.time() - t0) * 1e3, "load_ms": t["load_ms"], "rss_bytes": rss1})
    res["cold_start"] = min(colds, key=lambda c: c["wall_ms"])
    res["pack_bytes"] = Path(pack).stat().st_size
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-length", type=int)
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--skip-lme", action="store_true")
    ap.add_argument("--skip-speed", action="store_true")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(dir="./.warm-embed-work") as work:
        out = {"model": a.model_id, "pack": Path(a.pack).name, "threads": a.threads}
        if not a.skip_quality:
            out["quality"] = quality(a.binary, a.pack, "", "", a.threads, work, a.max_length, not a.skip_lme)
        if not a.skip_speed:
            out["speed"] = speed(a.binary, a.pack, [1, a.threads], a.repeats, work, a.max_length)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out)[:600])
