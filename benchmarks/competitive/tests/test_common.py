# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Statistics, ground truth and quality metrics that every verdict rests on."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hmsbench import data  # noqa: E402
from hmsbench.common import _Spawner, bootstrap_ci, dir_bytes, ndcg_at, pass_summary, quantile, recall_at, run_measured, timing_lock  # noqa: E402


def test_bootstrap_interval_brackets_the_median_and_narrows_with_more_data():
    rng = np.random.default_rng(1)
    few = list(rng.normal(100, 10, 5))
    many = list(rng.normal(100, 10, 200))
    lo5, hi5 = bootstrap_ci(few)
    lo200, hi200 = bootstrap_ci(many)
    assert lo5 <= float(np.median(few)) <= hi5
    assert (hi200 - lo200) < (hi5 - lo5)
    assert bootstrap_ci([7.0]) == [7.0, 7.0]


def test_higher_level_never_narrows_the_interval():
    values = list(np.random.default_rng(2).normal(0, 1, 20))
    lo95, hi95 = bootstrap_ci(values, level=0.95)
    lo99, hi99 = bootstrap_ci(values, level=0.99)
    assert lo99 <= lo95 and hi99 >= hi95


def test_quantile_matches_numpy_linear_interpolation():
    v = [3, 1, 4, 1, 5, 9, 2, 6]
    for q in (0.0, 0.5, 0.99, 1.0):
        assert quantile(v, q) == pytest.approx(float(np.quantile(v, q)))


def test_ndcg_and_recall_on_a_hand_checked_ranking():
    relevant = {"a": 1, "b": 1, "c": 0}
    assert ndcg_at(["a", "b"], relevant) == pytest.approx(1.0)
    swapped = ndcg_at(["x", "a", "b"], relevant)
    assert swapped == pytest.approx((1 / np.log2(3) + 1 / np.log2(4)) / (1 + 1 / np.log2(3)))
    assert recall_at(["a", "x"], relevant, 100) == 0.5
    assert ndcg_at(["a"], {}) == 0.0


@pytest.mark.parametrize("metric", ["angular", "euclidean"])
def test_recomputed_truth_equals_brute_force(metric):
    rng = np.random.default_rng(3)
    train = rng.normal(size=(500, 16)).astype(np.float32)
    test = rng.normal(size=(20, 16)).astype(np.float32)
    got = data._truth(train, test, metric, 10)
    if metric == "angular":
        a = train / np.linalg.norm(train, axis=1, keepdims=True)
        b = test / np.linalg.norm(test, axis=1, keepdims=True)
        score = -(b @ a.T)
    else:
        score = ((test[:, None, :] - train[None, :, :]) ** 2).sum(-1)
    want = np.argsort(score, axis=1, kind="stable")[:, :10]
    assert (got == want).all()


def test_dir_bytes_counts_allocated_not_apparent_size(tmp_path):
    sparse = tmp_path / "arena"
    with sparse.open("wb") as handle:
        handle.truncate(64 * 2**20)
    (tmp_path / "small").write_bytes(b"x" * 10)
    assert dir_bytes(tmp_path) < 2**20


def test_dir_bytes_ignores_zero_blocks_of_a_file_reported_fully_allocated(tmp_path):
    arena = tmp_path / "arena"
    arena.write_bytes(bytes(4 * 2**20) + b"x" + bytes(4095))
    assert dir_bytes(tmp_path) == 4096


def test_timing_lock_is_a_noop_on_ci(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    with timing_lock():
        pass
    monkeypatch.delenv("GITHUB_ACTIONS")
    monkeypatch.setenv("HMS_BENCH_LOCK", "/nonexistent-dir/timing.lock")
    with timing_lock():
        pass


def test_pass_summary_keeps_what_qps_and_latency_intervals_use():
    ns = [5, 1, 9, 3, 7] * 20
    got = pass_summary(ns)
    srt = sorted(ns)
    assert got == {"queries": 100, "total_ns": sum(ns), "p50_ns": srt[49], "p90_ns": srt[89], "p99_ns": srt[98], "max_ns": 9}


def test_spawner_reports_the_childs_own_peak_rss_and_exit_status(tmp_path):
    spawner = _Spawner()
    out, err = tmp_path / "out", tmp_path / "err"
    out.touch()
    err.touch()
    code = "b = bytearray(80 * 2**20); b[::4096] = b'\\x01' * len(b[::4096]); print('ok')"
    job = {"argv": [sys.executable, "-I", "-c", code], "env": dict(__import__("os").environ), "cwd": "", "stdout": str(out), "stderr": str(err)}
    status, peak = spawner.run(job, 60)
    assert status == 0 and out.read_text().strip() == "ok"
    assert 80 * 2**20 <= peak < 400 * 2**20
    assert spawner.run({**job, "argv": ["false"]}, 60)[0] == 1
    spawner.proc.stdin.close()
    spawner.proc.wait(timeout=10)


def test_run_measured_raises_on_failure_and_on_timeout():
    with pytest.raises(RuntimeError, match="exited 1"):
        run_measured(["false"])
    with pytest.raises(TimeoutError):
        run_measured(["sleep", "30"], timeout=1)
