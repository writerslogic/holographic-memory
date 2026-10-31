# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Assemble benchmarks/results/public_learned_sparse_v1.json from sparse-bench and Anserini runs.

Usage: learned_sparse_results.py --out FILE --host NAME --warm-dir DIR --anserini-dir DIR
                                 [--host NAME --warm-dir DIR --anserini-dir DIR ...]

For each host, <warm-dir>/warm_<dataset>_<model>.json (sparse-bench serving reports),
<warm-dir>/anserini_<dataset>_<model>_quality.json (the run-file evaluation) and
<anserini-dir>/<dataset>_<model>/{timing.jsonl,index_bytes.txt} are read. Anserini's
single-thread QPS is the throughput Anserini itself reports for the search loop over the test
topics repeated 200 times (median of three runs); wall-clock rounds are kept for reference. Ratios are WARM over Anserini on the same vectors
and host. Targets from the brief are evaluated per row and summarized, never enforced.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

DATASETS = ("nfcorpus", "scifact")
MODELS = ("splade-pp-ed", "os-doc-v2-distill")


def anserini_qps(dir_):
    """Anserini's own report ("N queries processed in HH:MM:SS = ~Q q/s": the search loop
    after the topics are loaded, one thread) over the repeated topics, median of the runs;
    the wall-clock rounds (JVM start, topic parsing and search) are kept for reference."""
    rounds = [json.loads(l.replace(": .", ": 0.")) for l in (dir_ / "timing.jsonl").read_text().splitlines() if l.strip()]
    reported = []
    f = dir_ / "qps_reported.txt"
    if f.exists():
        for line in f.read_text().splitlines():
            if "q/s" in line:
                reported.append(float(line.split("~")[1].split()[0]))
    return (statistics.median(reported) if reported else None), {"reported_q_per_s": reported, "wall_clock_rounds": rounds}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--host", action="append", required=True)
    p.add_argument("--warm-dir", action="append", required=True)
    p.add_argument("--anserini-dir", action="append", required=True)
    a = p.parse_args()
    rows = []
    for host, warm_dir, ans_dir in zip(a.host, a.warm_dir, a.anserini_dir):
        warm_dir, ans_dir = Path(warm_dir), Path(ans_dir)
        for ds in DATASETS:
            for m in MODELS:
                tag = f"{ds}_{m}"
                h = warm_dir / f"warm_{tag}.json"
                if h.exists():
                    d = json.loads(h.read_text())
                    rows.append({"dataset": ds, "model": m, "model_id": d["model"], "license": d["license"],
                                 "system": "warm_impact_index", "host": host, "ndcg_at_10": d["ndcg_at_10"],
                                 "recall_at_100": d["recall_at_100"], "index_bytes": d["index_bytes"],
                                 "postings": d["postings"], "bytes_per_posting": d["bytes_per_posting"],
                                 "qps_single_thread_k10": d["qps_single_thread"]["k10"]["median"],
                                 "qps_single_thread_k100": d["qps_single_thread"]["k100"]["median"],
                                 "rss_bytes_serving": d["rss_bytes"]["serving"], "build_seconds": d["build_seconds"],
                                 "queries": d["queries"], "ratios_to": "anserini_impact"})
                    if "blend" in d:
                        b = d["blend"]
                        rows.append({"dataset": ds, "model": m, "system": "warm_blend_sparse_exact_dense", "host": host,
                                     "ndcg_at_10": b["ndcg_at_10"], "recall_at_100": b["recall_at_100"], "alpha": b["alpha"],
                                     "tuning": b["tuning"], "dense_only_ndcg_at_10": b["dense_only_ndcg_at_10"],
                                     "dense_model": "sentence-transformers/all-MiniLM-L6-v2 (exact dot product over all documents)"})
                q = warm_dir / f"anserini_{tag}_quality.json"
                t = ans_dir / tag / "timing.jsonl"
                if q.exists() and t.exists():
                    d = json.loads(q.read_text())
                    qps, rounds = anserini_qps(ans_dir / tag)
                    rows.append({"dataset": ds, "model": m, "system": "anserini_impact", "host": host,
                                 "ndcg_at_10": d["ndcg_at_10"], "recall_at_100": d["recall_at_100"],
                                 "index_bytes": int((ans_dir / tag / "index_bytes.txt").read_text().strip()),
                                 "qps_single_thread_k10": qps, "timing": rounds, "queries": d["queries"],
                                 "version": "io.anserini:anserini:2.3.0 fatjar, Lucene impact index (-impact -pretokenized, 1 thread, hits 10)"})
    by = {(r["dataset"], r["model"], r["system"], r["host"]): r for r in rows}
    for r in rows:
        if r.get("ratios_to"):
            o = by.get((r["dataset"], r["model"], r["ratios_to"], r["host"]))
            if o and o.get("qps_single_thread_k10"):
                r["qps_ratio"] = r["qps_single_thread_k10"] / o["qps_single_thread_k10"]
                r["bytes_ratio"] = r["index_bytes"] / o["index_bytes"]
                r["ndcg_gap_to_anserini_same_vectors"] = r["ndcg_at_10"] - o["ndcg_at_10"]
    targets = {"ndcg_within_0.005_of_splade_same_vectors": {}, "qps_at_least_2x_anserini": {}, "bytes_at_most_0.7x_anserini": {}, "blend_above_sparse_alone": {}}
    for r in rows:
        key = f"{r['dataset']}/{r['model']}/{r['host']}"
        if r["system"] == "warm_impact_index" and "qps_ratio" in r:
            targets["ndcg_within_0.005_of_splade_same_vectors"][key] = abs(r["ndcg_gap_to_anserini_same_vectors"]) <= 0.005
            targets["qps_at_least_2x_anserini"][key] = r["qps_ratio"] >= 2.0
            targets["bytes_at_most_0.7x_anserini"][key] = r["bytes_ratio"] <= 0.7
        if r["system"] == "warm_blend_sparse_exact_dense":
            s = by.get((r["dataset"], r["model"], "warm_impact_index", r["host"]))
            if s:
                targets["blend_above_sparse_alone"][key] = r["ndcg_at_10"] > s["ndcg_at_10"]
    out = {"schema_version": 1, "comparator_note": "naver/splade-v3 (CC BY-NC-SA 4.0) is a gated repository the account cannot read (HTTP 401 on 2026-10-16); the ungated naver/splade-cocondenser-ensembledistil (SPLADE++ ED, CC BY-NC-SA 4.0) stands in as the comparator vectors; the shippable WARM encoder is opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill (Apache-2.0, document-only, IDF queries)",
           "protocol": "documents encoded once per model (max 256 tokens, log1p(relu) max pooling; Anserini weights quantized by 100), test queries read once per table; WARM serves 8-bit impacts with MaxScore (exact over quantized impacts) at depth 100 for quality and k=10 for QPS; Anserini 2.3.0 impact index, same vectors, single thread; blend alpha tuned on NFCorpus dev / SciFact train only",
           "rows": rows, "targets": targets}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"rows": len(rows), "targets": targets}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
