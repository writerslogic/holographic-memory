# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests of qgraph_paired.py: selection rule, paired ratios and the checks."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def row(ef, rerank, recall, evals, qps):
    return {"params": {"ef": ef, "rerank": rerank}, "recall_at_10": recall,
            "mean_exact_evals_per_query": evals, "qps_runs": qps}


def write_run(tmp, tag, rounds, a_rows, b_rows, completed=True):
    for name, rows in (("A", a_rows), ("B", b_rows)):
        (tmp / f"{tag}_{name}.json").write_text(json.dumps(
            {"dataset": "synthetic", "index_bytes": 100 if name == "A" else 150, "build": {"block_bits": 0 if name == "A" else 1}, "sweep": rows}))
    steps = [{"round": r, "proc": n, "load_before": [5.0, 5.0, 5.0], "secs": 1.0}
             for r in range(rounds) for n in ("A", "B")]
    (tmp / f"{tag}_coord.json").write_text(json.dumps({"rounds": rounds, "completed": completed, "steps": steps}))


def run(tmp, *args):
    return subprocess.run([sys.executable, "-I", str(HERE / "qgraph_paired.py"), str(tmp), *args],
                          capture_output=True, text=True)


def test_selection_and_ratios(tmp_path):
    a_rows = [row(64, 16, 0.89, 500, [1000, 1100, 900]), row(96, 16, 0.905, 700, [800, 850, 750]),
              row(128, 16, 0.951, 900, [600, 650, 550]), row(192, 16, 0.96, 1200, [400, 420, 380])]
    # The fewest-estimates rule picks ef 128 at 0.95 even though ef 192 reaches 0.96.
    b_rows = [row(64, 16, 0.91, 450, [1500, 1600, 1400]), row(128, 16, 0.952, 800, [1000, 1100, 900]),
              row(192, 16, 0.97, 1100, [700, 700, 700])]
    write_run(tmp_path, "t1", 3, a_rows, b_rows)
    r = run(tmp_path, "t1", "A", "B", "--out", str(tmp_path / "out.json"))
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["check"]["passed"]
    t90 = out["thresholds"]["0.90"]
    assert t90["baseline"]["params"] == {"ef": 96, "rerank": 16}
    assert t90["candidates"]["B"]["params"] == {"ef": 64, "rerank": 16}
    assert t90["candidates"]["B"]["paired_ratios"] == [1.875, 1.8824, 1.8667]
    t95 = out["thresholds"]["0.95"]
    assert t95["baseline"]["params"] == {"ef": 128, "rerank": 16}
    assert t95["candidates"]["B"]["params"] == {"ef": 128, "rerank": 16}
    # Ratios 1.6667, 1.6923, 1.6364: the median is the first round's.
    assert t95["candidates"]["B"]["paired_median_ratio"] == round(1000 / 600, 4)
    assert t95["candidates"]["B"]["paired_min_ratio"] == round(900 / 550, 4)
    assert t95["candidates"]["B"]["meets_min_ratio"] is True
    assert out["index_bytes"] == {"A": 100, "B": 150}
    assert len(out["load_1m_per_step"]) == 6


def test_missing_rounds_and_unreached_threshold_fail(tmp_path):
    a_rows = [row(64, 16, 0.91, 500, [1000, 1100]), row(128, 16, 0.951, 900, [600, 650])]
    b_rows = [row(64, 16, 0.92, 450, [1500, 1600, 1400])]
    write_run(tmp_path, "t2", 3, a_rows, b_rows)
    r = run(tmp_path, "t2", "A", "B")
    assert r.returncode == 1
    out = json.loads(r.stdout)
    failures = out["check"]["failures"]
    assert any("A ef=64: 2 of 3 rounds" in f for f in failures)
    assert any("B: no row reaches recall 0.95" in f for f in failures)


def test_incomplete_coordinator_fails(tmp_path):
    rows = [row(64, 16, 0.96, 500, [1000, 1000])]
    write_run(tmp_path, "t3", 2, rows, rows, completed=False)
    r = run(tmp_path, "t3", "A", "B")
    assert r.returncode == 1
    assert "coordinator did not complete" in r.stdout
