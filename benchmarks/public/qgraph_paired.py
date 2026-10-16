# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Summarize and check one paired `public-bench ann-qgraph --sync` run.

Usage: qgraph_paired.py <run dir> <tag> <baseline name> <candidate name>... [--out FILE]
                        [--thresholds 0.90,0.95] [--min-ratio 1.5]

Reads `<run dir>/<tag>_<name>.json` for every process and `<run dir>/<tag>_coord.json` from the
coordinator. Per process and recall threshold it selects the qualifying sweep row (recall@10 at or
above the threshold) with the fewest estimates per query, a deterministic choice that does not
depend on the noisy QPS, and reports the per-round paired QPS ratios of each candidate against the
baseline in the same round. The check fails when a process is missing rounds, when a recall value
is not reproduced across rounds (recall is deterministic), when the coordinator did not complete,
or when no row of a process reaches a threshold.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path


def pick(rows, threshold):
    ok = [r for r in rows if r["recall_at_10"] >= threshold]
    if not ok:
        return None
    return min(ok, key=lambda r: (r["mean_exact_evals_per_query"], r["params"]["ef"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("tag")
    p.add_argument("baseline")
    p.add_argument("candidates", nargs="+")
    p.add_argument("--out")
    p.add_argument("--thresholds", default="0.90,0.95")
    p.add_argument("--min-ratio", type=float, default=1.5)
    a = p.parse_args()
    run = Path(a.run_dir)
    names = [a.baseline] + a.candidates
    data = {n: json.loads((run / f"{a.tag}_{n}.json").read_text()) for n in names}
    coord = json.loads((run / f"{a.tag}_coord.json").read_text())
    failures = []
    if not coord.get("completed"):
        failures.append(f"coordinator did not complete: {coord.get('stopped')}")
    rounds = coord["rounds"] if "rounds" in coord else max(s["round"] for s in coord["steps"]) + 1
    for n, d in data.items():
        for r in d["sweep"]:
            if len(r["qps_runs"]) != rounds:
                failures.append(f"{n} ef={r['params']['ef']}: {len(r['qps_runs'])} of {rounds} rounds")
    out = {
        "schema_version": 1,
        "tag": a.tag,
        "dataset": data[a.baseline].get("dataset"),
        "rounds": rounds,
        "selection_rule": "per process and threshold, the row with recall@10 >= threshold and the fewest estimates per query; paired ratio = candidate QPS / baseline QPS in the same round",
        "load_1m_per_step": [[s["round"], s["proc"], round(s["load_before"][0], 1)] for s in coord["steps"]],
        "index_bytes": {n: d["index_bytes"] for n, d in data.items()},
        "build": {n: d.get("build") for n, d in data.items()},
        "thresholds": {},
    }
    for t in (float(x) for x in a.thresholds.split(",")):
        base = pick(data[a.baseline]["sweep"], t)
        if base is None:
            failures.append(f"{a.baseline}: no row reaches recall {t}")
            continue
        entry = {
            "baseline": {"name": a.baseline, "params": base["params"], "recall_at_10": base["recall_at_10"],
                         "evals_per_query": base["mean_exact_evals_per_query"], "qps_runs": base["qps_runs"],
                         "qps_median": statistics.median(base["qps_runs"])},
            "candidates": {},
        }
        for n in a.candidates:
            cand = pick(data[n]["sweep"], t)
            if cand is None:
                failures.append(f"{n}: no row reaches recall {t}")
                continue
            ratios = [c / b for b, c in zip(base["qps_runs"], cand["qps_runs"])]
            entry["candidates"][n] = {
                "params": cand["params"], "recall_at_10": cand["recall_at_10"],
                "evals_per_query": cand["mean_exact_evals_per_query"], "qps_runs": cand["qps_runs"],
                "qps_median": statistics.median(cand["qps_runs"]),
                "paired_ratios": [round(x, 4) for x in ratios],
                "paired_median_ratio": round(statistics.median(ratios), 4),
                "paired_min_ratio": round(min(ratios), 4),
                "rounds_above_1": sum(x > 1.0 for x in ratios),
                "meets_min_ratio": statistics.median(ratios) >= a.min_ratio,
            }
        out["thresholds"][f"{t:.2f}"] = entry
    out["check"] = {"passed": not failures, "failures": failures, "min_ratio": a.min_ratio}
    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
