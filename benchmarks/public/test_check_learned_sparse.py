# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests of check_learned_sparse.py."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(path):
    return subprocess.run([sys.executable, "-I", str(HERE / "check_learned_sparse.py"), str(path)],
                          capture_output=True, text=True)


def rows():
    hms = {"dataset": "nfcorpus", "model": "m", "system": "hms_impact_index", "host": "h", "ndcg_at_10": 0.33,
           "recall_at_100": 0.27, "index_bytes": 2000, "postings": 1000, "qps_single_thread_k10": 5000.0,
           "ratios_to": "anserini", "qps_ratio": 2.5, "bytes_ratio": 0.5}
    ans = {"dataset": "nfcorpus", "model": "m", "system": "anserini", "host": "h", "ndcg_at_10": 0.33,
           "recall_at_100": 0.27, "index_bytes": 4000, "postings": 1000, "qps_single_thread_k10": 2000.0}
    return [hms, ans]


def test_consistent_file_passes(tmp_path):
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"rows": rows()}))
    r = run(f)
    assert r.returncode == 0, r.stdout
    assert json.loads(r.stdout)["passed"]


def test_wrong_ratio_and_missing_competitor_fail(tmp_path):
    bad = rows()
    bad[0]["qps_ratio"] = 3.0
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"rows": bad}))
    r = run(f)
    assert r.returncode == 1
    assert "qps_ratio 3.0 differs" in r.stdout
    only = rows()[:1]
    f.write_text(json.dumps({"rows": only}))
    r = run(f)
    assert r.returncode == 1
    assert "no anserini row" in r.stdout


def test_out_of_range_quality_fails(tmp_path):
    bad = rows()
    bad[1]["ndcg_at_10"] = 1.2
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"rows": bad}))
    r = run(f)
    assert r.returncode == 1
    assert "ndcg_at_10 out of range" in r.stdout
