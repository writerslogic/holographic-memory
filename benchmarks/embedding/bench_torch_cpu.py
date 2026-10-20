# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""PyTorch fp32 CPU latency, fp32 size and RSS for each candidate, in one harness with fixed threads.

The reference column of the selection: how the same model costs through the stock stack. Each model
runs in its own subprocess so its peak RSS (ru_maxrss) is its own. Machine load is recorded; on a
shared machine compare models within one run, not against other runs.

  uv run --no-project --with sentence-transformers --with numpy python benchmarks/embedding/bench_torch_cpu.py \
      --out torch_cpu.json
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def child(model_id, revision, threads, n, repeats, max_len):
    import torch
    from sentence_transformers import SentenceTransformer

    from select_model import CANDIDATES, beir

    torch.set_num_threads(threads)
    qp, pp, _ = CANDIDATES[model_id]
    t0 = time.time()
    m = SentenceTransformer(model_id, revision=revision, device="cpu")
    load = time.time() - t0
    m.max_seq_length = min(m.max_seq_length or 512, max_len)
    _, texts, _, qtexts, _ = beir("scifact", "train")
    passages = [pp + t for t in texts[:n]]
    queries = [qp + t for t in qtexts[:n]]
    best = {}
    for name, items in (("passage", passages), ("query", queries)):
        runs = []
        for _ in range(repeats):
            t = time.time()
            m.encode(items, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
            runs.append((time.time() - t) / len(items) * 1e3)
        best[name] = min(runs)
    n_params = sum(p.numel() for p in m.parameters())
    print(json.dumps({"load_seconds": load, "passage_ms_each": best["passage"], "query_ms_each": best["query"],
                      "parameters": n_params, "fp32_bytes": n_params * 4, "threads": threads,
                      "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--stage1", help="stage-1 JSON for the pinned revisions")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--child")
    ap.add_argument("--revision")
    a = ap.parse_args()
    if a.child:
        child(a.child, a.revision, a.threads, a.n, a.repeats, a.max_len)
        sys.exit(0)
    stage1 = json.load(open(a.stage1))
    res = {"load_average_1m": os.getloadavg()[0], "results": {}}
    for mid in a.models or [m for m, v in stage1.items() if v.get("engine_architecture") and "error" not in v
                            and "skipped" not in v]:
        for threads in (1, a.threads):
            p = subprocess.run(
                [sys.executable, __file__, "--child", mid, "--revision", stage1[mid]["info"]["revision"],
                 "--threads", str(threads), "--n", str(a.n), "--repeats", str(a.repeats),
                 "--max-len", str(a.max_len)], capture_output=True, text=True)
            line = next((l for l in p.stdout.splitlines() if l.startswith("{")), None)
            res["results"].setdefault(mid, {})[f"threads_{threads}"] = json.loads(line) if line else {
                "error": p.stderr[-300:]}
            print(mid, threads, line, flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
