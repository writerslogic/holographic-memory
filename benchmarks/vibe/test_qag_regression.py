# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import copy
import statistics

import pytest

from qag_regression import SETTINGS, matched_gate


def paired_report(candidate_qps, matching=True):
    rows = []
    for arm in (0, 1):
        for ef, rerank in SETTINGS:
            qps = [100., 100., 100.] if arm == 0 else candidate_qps
            rows.append({
                "arm": arm, "ef": ef, "rerank": rerank,
                "recall_ids_min": 0.95,
                "qps_median": statistics.median(qps), "qps_min": min(qps), "qps_max": max(qps),
                "repetitions": [
                    {"repetition": repetition, "qps": value,
                     "ids_sha256": "same" if matching or arm == 0 else "different"}
                    for repetition, value in enumerate(qps)
                ],
            })
    return {"rows": rows}


@pytest.mark.parametrize(("qps", "expected"), [(98., "pass"), (97., "pass"), (96., "fail")])
def test_three_percent_regression_boundary(qps, expected):
    gate = matched_gate(paired_report([qps] * 3))
    assert gate["gate"] == expected
    assert all(point["gate"] == expected for point in gate["points"])


def test_noise_crossing_gate_is_inconclusive():
    gate = matched_gate(paired_report([96., 97., 98.]))
    assert gate["gate"] == "inconclusive"
    assert gate["points"][0]["candidate_qps_range"] == [96., 98.]


def test_different_retrieval_results_do_not_establish_matched_recall():
    assert matched_gate(paired_report([150.] * 3, matching=False))["gate"] == "inconclusive"


def test_missing_or_duplicated_configurations_are_rejected():
    report = paired_report([100.] * 3)
    incomplete = copy.deepcopy(report)
    incomplete["rows"].pop()
    with pytest.raises(ValueError, match="six"):
        matched_gate(incomplete)
    duplicated = copy.deepcopy(report)
    duplicated["rows"].append(duplicated["rows"][0])
    with pytest.raises(ValueError, match="six"):
        matched_gate(duplicated)
