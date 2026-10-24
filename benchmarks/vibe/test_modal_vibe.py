# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Stall accounting of the Modal VIBE runner: VIBE runs one process per index config, and a config's
process exiting must not look like the job stopping."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.modules.setdefault("modal", mock.MagicMock())
_spec = importlib.util.spec_from_file_location("modal_vibe", Path(__file__).with_name("modal_vibe.py"))
mv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mv)

RUNPY, WORKER, CFG1, CFG2 = (10, 1), (11, 1), (12, 1), (13, 1)


def test_progress_survives_config_process_exit():
    # Config 1 ran 1500 s and exits; config 2 starts at 0 and gains 60 s per beat. The old total
    # (sum of live CPU) dropped to 60 and needed 25 min to pass 1500, which killed healthy builds.
    before = {RUNPY: 2.0, WORKER: 1.0, CFG1: 1500.0}
    after = {RUNPY: 2.0, WORKER: 1.0, CFG2: 60.0}
    assert mv.cpu_progress(before, after) == 60.0
    assert mv.cpu_progress(after, after) == 0.0


def test_progress_counts_only_gains():
    assert mv.cpu_progress({CFG1: 10.0}, {CFG1: 10.5, CFG2: 0.0}) == 0.5


def test_descendants_follow_ppid_and_skip_unrelated():
    table = {
        (1, 0): {"ppid": 0, "pgid": 1, "cpu": 0.0},
        (5, 0): {"ppid": 1, "pgid": 5, "cpu": 0.0},  # supervisor
        RUNPY: {"ppid": 5, "pgid": 10, "cpu": 0.0},
        WORKER: {"ppid": 10, "pgid": 10, "cpu": 0.0},
        CFG1: {"ppid": 11, "pgid": 12, "cpu": 0.0},  # own session, as VIBE starts it
        (20, 3): {"ppid": 1, "pgid": 20, "cpu": 0.0},
    }
    assert mv.descendants(table, 10) == {RUNPY, WORKER, CFG1}
    assert mv.descendants(table, 99) == set()


def test_snapshot_marks_only_same_container_completed_comparisons_fair(tmp_path):
    rev = "a" * 40
    root = "/vibe/run-s-20261023"
    files = {
        f"{root}/plan.json": {"datasets": {
            "a": {"sha256": "a-hash", "size_bytes": 100, "mem_gib": 8, "leaders": {}},
            "b": {"sha256": "b-hash", "size_bytes": 200, "mem_gib": 16, "leaders": {}},
        }},
        f"{root}/ledger/a/hms@{rev[:12]}.json": {"dataset": "a", "job": f"hms@{rev[:12]}",
                                                     "state": "done", "container_id": "c1", "cpu_model": "CPU A"},
        f"{root}/ledger/a/same.json": {"dataset": "a", "job": "same", "state": "done",
                                         "container_id": "c1", "cpu_model": "CPU A"},
        f"{root}/ledger/a/other.json": {"dataset": "a", "job": "other", "state": "done",
                                          "container_id": "c2", "cpu_model": "CPU A"},
        f"{root}/ledger/b/_download.json": {"dataset": "b", "job": "_download", "state": "failed",
                                              "error": "download failed"},
        f"{root}/standings/a@{rev[:12]}.json": {"dataset": "a", "hms_commit": rev,
                                                 "best_qps_at_recall": {"hms": {"0.95": {"qps": 42}}}},
        f"{root}/spend/c1.json": {"dataset": "a", "last": 1, "usd": 1.0},
    }

    def entries(path, recursive=False):
        return [SimpleNamespace(path=p) for p in files if p.startswith(path + "/")
                and (recursive or "/" not in p[len(path) + 1:])]

    with mock.patch.object(mv.VOL, "iterdir", side_effect=entries), \
         mock.patch.object(mv.VOL, "read_file", side_effect=lambda p: [json.dumps(files[p]).encode()]), \
         mock.patch.object(mv.SPEND, "items", return_value=[(f"run-s-20261023:c1", {"dataset": "a", "last": 2, "usd": 2.0})]):
        out = tmp_path / "snapshot.json"
        mv.snapshot_report("run-s-20261023", out, rev)

    report = json.loads(out.read_text())
    assert len(report["rows"]) == 2
    a, b = report["rows"]
    assert a["competitors"]["same"]["fair_comparison"] is True
    assert a["competitors"]["other"]["fair_comparison"] is False
    assert a["comparison_status"] == "comparable"
    assert a["metrics"]["0.95"]["qps"] == 42
    assert a["estimated_cost_usd"] == 2.0
    assert a["peak_rss_bytes"] is None
    assert a["state"] == "complete" and a["hms_state"] == "done"
    assert b["state"] == "failed" and b["download_state"] == "failed"
    assert b["comparison_status"] == "pending"
