"""Builds benchmarks/results/qgraph_stop_screen_heldout.json from the held-out runs of the
patience stop rule (E1) and the 1-bit screen (E2): the kill tests (recall and evaluations at
fixed ef, one run each) and the paired timing (one invocation per set, every arm timed in
every round, QPS at recall 0.90 / 0.95 interpolated in log QPS on each arm's Pareto front per
round, paired ratios against the fixed-ef arm). Usage:
  uv run python stop_screen_report.py <e1 dir> <timing dir> <out.json>"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np


def frontier(rows):
    pts = sorted(((r["recall"], r["qps"]) for r in rows), key=lambda x: (-x[0], -x[1]))
    out, best = [], -1.0
    for rec, qps in pts:
        if qps > best:
            out.append((rec, qps))
            best = qps
    return sorted(out)


def qps_at(front, target):
    lo = [p for p in front if p[0] < target]
    hi = [p for p in front if p[0] >= target]
    if not hi:
        return None
    if not lo:
        # Every row exceeds the target: the slowest such row is a conservative lower bound
        # (the arm was not swept below the target).
        return hi[0][1]
    (r0, q0), (r1, q1) = lo[-1], hi[0]
    if r1 == r0:
        return q1
    t = (target - r0) / (r1 - r0)
    return math.exp(math.log(q0) + t * (math.log(q1) - math.log(q0)))


def arm_of(p, enc=""):
    pat, scr = p.get("patience", 0), p.get("screen", False)
    base = {(False, False): "fixed_ef", (True, False): "patience", (False, True): "screen", (True, True): "patience_and_screen"}[(pat > 0, bool(scr))]
    return f"{enc}{base}"


def kill_tests(e1_dir):
    out = {}
    for f in sorted(glob.glob(f"{e1_dir}/*_p*.json")):
        d = json.loads(Path(f).read_text())
        name = d["dataset"]["name"]
        for r in d["sweep"]:
            p = r["params"]
            out.setdefault(name, []).append({"ef": p["ef"], "rerank": p["rerank"], "patience": p.get("patience", 0), "recall": r["recall_at_10"], "evals_per_query": r["mean_exact_evals_per_query"], "load_1m": r["load_1m_before_runs"]})
    return out


def timing(tdir):
    """Every `<tag>_<proc>.json` of a paired run in `tdir`: the processes of one tag share
    rounds, so their arms are compared round by round; the arm name carries the encoding."""
    res = {}
    by_tag = {}
    for f in sorted(glob.glob(f"{tdir}/e12*_[A-Z].json")):
        d = json.loads(Path(f).read_text())
        by_tag.setdefault(d["dataset"]["name"], []).append(d)
    for name, docs in by_tag.items():
        rows, arms = [], {}
        nrounds = min(len(d["sweep"][0]["qps_runs"]) for d in docs)
        index_bytes = {}
        for d in docs:
            bp = d["build"]["params"]
            enc = f"{bp.get('vertex_bits', 8)}bit_"
            index_bytes[enc] = d["index_bytes"]
            for r in d["sweep"]:
                rows.append(r)
                arms.setdefault(arm_of(r["params"], enc), []).append(r)
        base_arm = "8bit_fixed_ef"
        per_round = {a: {"0.90": [], "0.95": []} for a in arms}
        for k in range(nrounds):
            for a, rs in arms.items():
                front = frontier([{"recall": r["recall_at_10"], "qps": r["qps_runs"][k]} for r in rs if k < len(r["qps_runs"])])
                for t in ("0.90", "0.95"):
                    per_round[a][t].append(qps_at(front, float(t)))
        summary = {}
        for a in arms:
            summary[a] = {}
            for t in ("0.90", "0.95"):
                mine, base = per_round[a][t], per_round[base_arm][t]
                pairs = [(m, b) for m, b in zip(mine, base) if m and b]
                ratios = [m / b for m, b in pairs]
                summary[a][t] = {"qps_by_round": [round(x) if x else None for x in mine],
                                 "median_qps": float(np.median([m for m, _ in pairs])) if pairs else None,
                                 "paired_ratio_median": float(np.median(ratios)) if ratios else None,
                                 "paired_ratio_min_max": [min(ratios), max(ratios)] if ratios else None,
                                 "rounds_faster": sum(r > 1 for r in ratios), "rounds": len(ratios)}
        evals = {a: {f"ef{r['params']['ef']}": {"recall": r["recall_at_10"], "evals": r["mean_exact_evals_per_query"], "screened": r.get("screened_per_query", 0)} for r in rs} for a, rs in arms.items()}
        res[name] = {"rounds": nrounds, "load_gate_met": all(r["load_gate_met"] for r in rows), "index_bytes": index_bytes, "at_recall": summary, "rows": evals,
                     "load_1m_all": sorted({round(x) for r in rows for x in r["load_1m_before_runs"]})}
    return res


if __name__ == "__main__":
    e1, tdir, out = sys.argv[1:4]
    doc = {"workstream": "E1 patience stop rule and E2 1-bit screen (ideas tree 2026-10, docs/research/IDEAS-2026-10.md)",
           "method": "public-bench ann-qgraph --index vertex --residual --holdout 2000 (train vectors held out; test set unread); held-out graphs from the checked cache (degree 64 nytimes / 32 glove, build_ef 200, alpha 1.0); kill tests at --repeats 1 without the load gate (recall and evaluation counts are load-independent); paired timing: one invocation per set, every (ef, patience, screen) configuration timed back to back in each round under timed.sh, 4 rayon threads for the build, single-threaded search; the load gate (< 3) was never met, so absolute QPS are indicative and only the paired ratios decide",
           "decision_rules": "E1 goes if evals at recall within 0.005 of a fixed-ef row fall >= 15% at 0.90 and >= 5% at 0.95 on both sets, then paired QPS; E2 goes if recall at fixed ef is within 0.003 of the unscreened row with >= 60% of the 8-bit estimates skipped, then paired QPS; a ratio below 1 is a loss and is reported",
           "kill_tests": kill_tests(e1), "paired_timing": timing(tdir)}
    Path(out).write_text(json.dumps(doc, indent=1) + "\n")
    for name, t in doc["paired_timing"].items():
        print(name, "rounds", t["rounds"], "gate", t["load_gate_met"], "load", t["load_1m_all"][:3], "..", t["load_1m_all"][-1:])
        for a, v in t["at_recall"].items():
            print("  ", a, {k: (vv["paired_ratio_median"], vv["rounds_faster"], vv["rounds"]) for k, vv in v.items()})
