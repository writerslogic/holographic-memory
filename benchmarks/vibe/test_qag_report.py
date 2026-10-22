# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("qag_report", Path(__file__).with_name("qag_report.py"))
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def source_file(tmp_path, queries=2, rows=200, metric="ip", scope="validation"):
    path = tmp_path / "data.hdf5"
    values = np.arange(rows, 0, -1, dtype=np.float64)
    base = np.column_stack((values, np.zeros(rows)))
    probes = np.tile([1., 0.], (queries, 1))
    truth = np.tile(np.arange(100), (queries, 1))
    with h5py.File(path, "w") as data:
        data.attrs.update(distance=metric, dimension=2)
        if scope is not None:
            data.attrs["qag_scope"] = scope
        data.create_dataset("train", data=base)
        data.create_dataset("test", data=probes)
        data.create_dataset("neighbors", data=truth)
        data.create_dataset("distances", data=np.full((queries, 100), -999.))
    return path, truth


def result_file(tmp_path, truth, configurations=None):
    path = tmp_path / "raw.hdf5"
    if configurations is None:
        configurations = [(0, 2000, 1000, [truth] * 3, [4., 1., 2.])]
    with h5py.File(path, "w") as output:
        for arm, ef, rerank, predictions, seconds in configurations:
            for repetition, (ids, total) in enumerate(zip(predictions, seconds)):
                group = output.create_group(f"{arm}_{ef}_{rerank}_{repetition}")
                group.attrs.update(count=100, run_count=5, distance="ip", best_qps=1e12,
                                   qag_arm=arm, qag_repetition=repetition,
                                   qag_process_peak_rss_kib=4096, build_time=30., index_size=1024,
                                   qag_resource_scope="all indexes co-resident")
                group.create_dataset("neighbors", data=ids)
                group.create_dataset("distances", data=np.where(np.asarray(ids) < 0, np.inf, 0.))
                group.create_dataset("times", data=np.full(len(truth), total / len(truth)))
    return path


def test_saved_fifth_pass_times_override_best_qps_and_preserve_provenance(tmp_path):
    data, truth = source_file(tmp_path)
    raw = result_file(tmp_path, truth)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"harness_sha256": {"runner.py": "abc"}, "extensions": {"baseline": "123"}}))
    out = tmp_path / "report.json"
    report = reporter.write_report(raw, data, out, receipt)
    row = report["rows"][0]
    assert (row["qps_median"], row["qps_min"], row["qps_max"]) == (1., 0.5, 2.)
    assert row["recall_ids_min"] == row["recall_epsilon_min"] == 1.
    assert report["protocol"]["timed_pass"] == 5
    assert report["source"]["scope"] == "validation"
    assert report["source"]["official_test"] is False
    assert report["provenance"]["raw_results_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert report["provenance"]["source_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert report["provenance"]["harness_sha256"] == {"runner.py": "abc"}
    assert report["provenance"]["execution_receipt"]["metadata"]["extensions"]["baseline"] == "123"
    for repetition in row["repetitions"]:
        assert repetition["ids_sha256"] == hashlib.sha256(truth.astype("<i8").tobytes()).hexdigest()
        assert repetition["resource_attrs"]["qag_process_peak_rss_kib"] == 4096
        assert "best_qps" not in repetition["resource_attrs"]
    original = out.read_bytes()
    with pytest.raises(FileExistsError):
        reporter.write_report(raw, data, out)
    assert out.read_bytes() == original


def test_targets_require_actual_minimum_recall_without_rounding_or_interpolation(tmp_path):
    data, truth = source_file(tmp_path, queries=200)
    deficient = truth.copy()
    deficient[:2] = np.arange(100, 200)
    deficient[2, 0] = 100
    raw = result_file(tmp_path, truth, [
        (1, 100, 100, [truth, deficient, truth], [0.1, 0.1, 0.1]),
        (1, 200, 100, [truth] * 3, [0.2, 0.2, 0.2]),
    ])
    report = reporter.extract(raw, data)
    assert report["rows"][0]["recall_ids_min"] == 0.98995
    assert len({run["ids_sha256"] for run in report["rows"][0]["repetitions"]}) == 2
    assert next(point for point in report["targets"] if point["target"] == 0.95)["ef"] == 100
    assert next(point for point in report["targets"] if point["target"] == 0.99)["ef"] == 200


@pytest.mark.parametrize("metric,query,rows,expected", [
    ("ip", [1., 0.], [[2., 0.], [40., 30.]], [-2., -40.]),
    ("euclidean", [0., 0.], [[3., 4.], [6., 8.]], [5., 10.]),
    ("cosine", [1., 0.], [[2., 0.], [40., 30.]], [0., 0.2]),
    ("angular", [1., 0.], [[2., 0.], [40., 30.]], [0., 0.2]),
    ("normalized", [1., 0.], [[0.8, 0.6], [0.6, 0.8], [2., 0.], [-2., 0.]], [0.2, 0.4, 0., 2.]),
])
def test_original_metric_direction_and_scaling(metric, query, rows, expected):
    np.testing.assert_allclose(reporter.metric_distances(metric, query, rows), expected, atol=1e-15)


def test_epsilon_recall_recomputes_float64_original_scores_instead_of_saved_distances(tmp_path):
    data, truth = source_file(tmp_path, queries=1)
    with h5py.File(data, "r+") as source:
        source["train"][100, 0] = source["train"][99, 0] - 0.0005
    ids = truth.copy()
    ids[0, -1] = 100
    raw = result_file(tmp_path, truth, [(0, 100, 100, [ids] * 3, [1., 1., 1.])])
    row = reporter.extract(raw, data)["rows"][0]
    assert row["recall_ids_min"] == 0.99
    assert row["recall_epsilon_min"] == 1.


def test_missing_ids_do_not_wrap_to_the_last_base_row_or_count_as_duplicates(tmp_path):
    data, truth = source_file(tmp_path)
    missing = truth.copy()
    missing[:, :3] = -1
    raw = result_file(tmp_path, truth, [(4, 2, 0, [missing] * 3, [1., 1., 1.])])
    report = reporter.extract(raw, data)
    assert report["rows"][0]["recall_ids_min"] == 0.97
    assert report["rows"][0]["recall_epsilon_min"] == 0.97
    assert report["rows"][0]["method"] == "mlann"
    assert next(point for point in report["targets"] if point["target"] == 0.99)["reached"] is False


@pytest.mark.parametrize("malformed", ["duplicate", "out_of_range", "negative", "float", "shape"])
def test_invalid_returned_ids_are_rejected(tmp_path, malformed):
    data, truth = source_file(tmp_path)
    raw = result_file(tmp_path, truth)
    with h5py.File(raw, "r+") as output:
        group = output["0_2000_1000_0"]
        if malformed == "duplicate":
            group["neighbors"][0, 0] = 1
        elif malformed == "out_of_range":
            group["neighbors"][0, 0] = 200
        elif malformed == "negative":
            group["neighbors"][0, 0] = -2
        else:
            del group["neighbors"]
            group.create_dataset("neighbors", data=truth.astype(float) if malformed == "float" else truth[:, :99])
    with pytest.raises(ValueError, match="IDs|range"):
        reporter.extract(raw, data)


@pytest.mark.parametrize("malformed", ["count", "passes", "missing_repetition", "zero_time", "nan_time",
                                      "infinite_time", "times_shape", "arguments", "metric", "distances", "missing_times"])
def test_malformed_measurement_protocol_is_rejected(tmp_path, malformed):
    data, truth = source_file(tmp_path)
    raw = result_file(tmp_path, truth)
    with h5py.File(raw, "r+") as output:
        group = output["0_2000_1000_0"]
        if malformed == "count":
            group.attrs["count"] = 99
        elif malformed == "passes":
            group.attrs["run_count"] = 4
        elif malformed == "missing_repetition":
            del output["0_2000_1000_2"]
        elif malformed.endswith("time"):
            group["times"][0] = {"zero_time": 0., "nan_time": np.nan, "infinite_time": np.inf}[malformed]
        elif malformed == "times_shape":
            del group["times"]
            group.create_dataset("times", data=[1.])
        elif malformed == "arguments":
            group.attrs["qag_repetition"] = 1
        elif malformed == "metric":
            group.attrs["distance"] = "euclidean"
        elif malformed == "distances":
            group["distances"][0, 0] = np.nan
        elif malformed == "missing_times":
            del group["times"]
    with pytest.raises(ValueError):
        reporter.extract(raw, data)


def test_source_truth_duplicates_and_nonfinite_vectors_are_rejected(tmp_path):
    data, truth = source_file(tmp_path)
    raw = result_file(tmp_path, truth)
    with h5py.File(data, "r+") as source:
        source["neighbors"][0, 0] = 1
    with pytest.raises(ValueError, match="duplicate"):
        reporter.extract(raw, data)
    with h5py.File(data, "r+") as source:
        source["neighbors"][0, 0] = 0
        source["train"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        reporter.extract(raw, data)


def test_undeclared_source_scope_is_official_and_unrecorded_receipt_stays_explicit(tmp_path):
    data, truth = source_file(tmp_path, scope=None)
    raw = result_file(tmp_path, truth)
    report = reporter.extract(raw, data)
    assert report["source"]["official_test"] is True
    assert report["source"]["scope"] == "official_test"
    assert report["provenance"]["harness_sha256"] is None
    assert report["provenance"]["execution_receipt"] is None
