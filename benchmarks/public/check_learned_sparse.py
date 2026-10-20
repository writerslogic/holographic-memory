# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check a learned-sparse results file (benchmarks/results/public_learned_sparse_*.json).

Usage: check_learned_sparse.py <results.json> [--summary OUT]

Every row must name its dataset, model, system and host; quality values lie in [0, 1]; HMS
rows carry positive index bytes, postings and single-thread QPS; a competitor row that
serves the same vectors on the same host must exist for every HMS row that claims a QPS or
bytes ratio, and the ratios must be recomputed from the rows (a ratio that does not match its
rows within 0.5% fails). The targets are evaluated and reported, never enforced.
"""
from pathlib import Path
import argparse
import json
import sys


def close(a, b, tol=0.005):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def check(doc):
    failures = []
    rows = doc.get("rows", [])
    if not rows:
        failures.append("no rows")
    index = {}
    for i, r in enumerate(rows):
        for key in ("dataset", "model", "system", "host", "ndcg_at_10", "recall_at_100"):
            if key not in r:
                failures.append(f"row {i}: missing {key}")
        for key in ("ndcg_at_10", "recall_at_100"):
            v = r.get(key)
            if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
                failures.append(f"row {i}: {key} out of range: {v}")
        if r.get("system") == "hms_impact_index":
            for key in ("index_bytes", "postings", "qps_single_thread_k10"):
                if not isinstance(r.get(key), (int, float)) or r.get(key, 0) <= 0:
                    failures.append(f"row {i}: {key} must be positive")
        index[(r.get("dataset"), r.get("model"), r.get("system"), r.get("host"))] = r
    for i, r in enumerate(rows):
        comp = r.get("ratios_to")
        if not comp:
            continue
        other = index.get((r["dataset"], r["model"], comp, r["host"]))
        if other is None:
            failures.append(f"row {i}: no {comp} row for the same vectors on the same host")
            continue
        for key, num, den in (("qps_ratio", "qps_single_thread_k10", "qps_single_thread_k10"),
                              ("bytes_ratio", "index_bytes", "index_bytes")):
            if key in r:
                want = r[num] / other[den]
                if not close(r[key], want):
                    failures.append(f"row {i}: {key} {r[key]} differs from {want:.4f}")
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results")
    p.add_argument("--summary")
    a = p.parse_args()
    doc = json.loads(Path(a.results).read_text())
    failures = check(doc)
    out = {"results": a.results, "rows": len(doc.get("rows", [])), "passed": not failures, "failures": failures,
           "targets": doc.get("targets")}
    text = json.dumps(out, indent=1)
    if a.summary:
        Path(a.summary).write_text(text + "\n")
    print(text)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
