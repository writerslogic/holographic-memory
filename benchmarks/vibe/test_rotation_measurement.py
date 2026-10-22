# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json

import h5py
import numpy as np
import pytest

from check_rotation import ARMS, check, sha


@pytest.fixture
def measured(tmp_path):
    angles = np.linspace(0, 2 * np.pi, 101, endpoint=False)
    train = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
    queries = train[[0, 20, 60, 100]]
    values = np.clip(1 - queries.astype(np.float64) @ train.astype(np.float64).T, 0, 2)
    truth = np.argsort(values, axis=1)[:, :100]
    distances = np.take_along_axis(values, truth, axis=1).astype(np.float32)
    source = tmp_path / "fixture-n101.hdf5"
    with h5py.File(source, "w") as handle:
        handle.attrs["distance"] = "normalized"
        for name, matrix in (("train", train), ("test", queries), ("neighbors", truth), ("distances", distances)):
            handle.create_dataset(name, data=matrix)
    ids = truth.astype("<u4")
    ids[:, -1] = np.iinfo(np.uint32).max
    rows = [{"arm": arm, "ef": 100, "rerank": 100, "recall": .99, "strict_id_recall": .99,
             "missing_results": 4, "prediction_row_offset": i * 400,
             "query_ns": [[10] * 4] * 3, "qps_runs": [1e8] * 3, "median_qps": 1e8}
            for i, arm in enumerate(ARMS)]
    report = {"data": {"input_manifest": {"dataset": "fixture", "rows": 101, "sha256": sha(source)}, "dim": 2},
              "k": 100, "n_queries": 4, "repeats": 3, "graph_hash": "fixed", "rows": rows,
              "arms": [{"name": arm, "index_bytes": 1000} for arm in ARMS]}
    path = tmp_path / "result.json"
    path.write_text(json.dumps(report))
    np.tile(ids, (4, 1)).tofile(path.with_suffix(".ids.u32"))
    return path, report


def test_missing_ids_remain_misses_and_targets_do_not_round_up(measured):
    path, _ = measured
    result = check(path, path.parent)
    assert result["passed"]
    for arm in result["arms"]:
        assert arm["max_recall"] == .99
        assert arm["targets"]["0.99"] is not None
        assert arm["targets"]["1.0"] is None


@pytest.mark.parametrize("corruption", ("duplicate", "outside", "truncated", "offset", "missing", "recall", "timing", "grid", "input"))
def test_corrupt_rotation_evidence_is_rejected(measured, corruption):
    path, report = measured
    raw = path.with_suffix(".ids.u32")
    ids = np.fromfile(raw, dtype="<u4")
    if corruption == "duplicate":
        ids[1] = ids[0]
    elif corruption == "outside":
        ids[0] = 101
    elif corruption == "truncated":
        ids = ids[:-1]
    elif corruption == "offset":
        report["rows"][0]["prediction_row_offset"] = 1
    elif corruption == "missing":
        report["rows"][0]["missing_results"] = 0
    elif corruption == "recall":
        report["rows"][0]["recall"] = 1.0
    elif corruption == "timing":
        report["rows"][0]["query_ns"][0][0] = 0
    elif corruption == "grid":
        report["rows"][0]["ef"] = 101
    else:
        report["data"]["input_manifest"]["sha256"] = "wrong"
    ids.tofile(raw)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        check(path, path.parent)
