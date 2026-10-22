# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from qag_cost_report import profile_summary, timing_summary


def test_profiles_reject_changed_work_and_account_for_timer_overhead():
    work = dict(query_encoding_ns=1, entry_ns=2, adjacency_visited_prefetch_ns=30,
                scoring_pool_ns=40, rerank_sort_ns=20, total_ns=100,
                expansions=12, neighbors_read=600, entry_evaluations=4,
                traversal_evaluations=200, rerank_evaluations=100, retained=256)
    profile = dict(returned_ids_identical=True, persisted_bytes_identical=True,
                   graph_hash=123, settings={"ef": 256, "rerank": 400}, layouts=[
                       dict(compact=flag, recall=.99, index_bytes=size,
                            compaction_seconds=0, persisted={"bytes": 4096},
                            queries=[deepcopy(work) for _ in range(256)])
                       for flag, size in ((False, 2000), (True, 1500))])
    result = profile_summary(profile)
    assert result["encoded_saving_percent"] == 25
    assert result["layouts"][0]["unattributed_percent"] == 7
    profile["layouts"][1]["queries"][128]["neighbors_read"] += 1
    with pytest.raises(ValueError, match="changed production search work"):
        profile_summary(profile)


def test_paired_timing_requires_complete_identical_ids_and_separated_ranges():
    def row(arm, speeds):
        return dict(arm=arm, ef=100, rerank=100, recall_ids_min=.91,
                    qps_median=sorted(speeds)[1], qps_min=min(speeds), qps_max=max(speeds),
                    repetitions=[dict(repetition=i, ids_sha256=str(i), qps=qps)
                                 for i, qps in enumerate(speeds)])
    report = dict(source={"official_test": False}, rows=[row(0, [90, 100, 110]), row(1, [99, 110, 121])])
    gate = dict(identical_returned_ids=True, points=[dict(ef=100, rerank=100)])
    assert timing_summary(report, gate)[0]["status"] == "inconclusive"
    report["rows"][1] = row(1, [120, 130, 140])
    assert "separated ranges" in timing_summary(report, gate)[0]["status"]
    report["rows"][1]["repetitions"][1]["ids_sha256"] = "changed"
    with pytest.raises(ValueError, match="IDs differ"):
        timing_summary(report, gate)
    report["rows"][1]["repetitions"].pop(1)
    with pytest.raises(ValueError, match="three complete paired blocks"):
        timing_summary(report, gate)
