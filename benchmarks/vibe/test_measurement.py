# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measurement = load("run")
preparation = load("prepare")


def test_tolerance_recall_is_distinct_from_exact_id_recall():
    train = np.array([[1.0, 0], [0.9995, 0], [0.9989, 0]], dtype=np.float32)
    queries = np.array([[1.0, 0]], dtype=np.float32)
    truth, distances = np.array([[0]]), np.array([[0.0]])
    admitted = measurement.score(np.array([[1]]), train, queries, truth, distances, "normalized")
    excluded = measurement.score(np.array([[2]]), train, queries, truth, distances, "normalized")
    assert admitted["recall"] == 1 and admitted["strict_id_recall"] == 0
    assert excluded["recall"] == 0


def test_missing_results_keep_the_denominator():
    train = np.eye(3, dtype=np.float32)
    result = measurement.score(np.array([[0, -1]]), train, train[:1], np.array([[0, 1]]),
                               np.array([[0, 1]]), "normalized")
    assert result["recall"] == result["strict_id_recall"] == 0.5
    assert result["missing_results"] == 1


@pytest.mark.parametrize("ids", [[[0, 0]], [[0, 3]], [[0, -2]], [[0.0, 1.0]]])
def test_corrupt_ids_are_rejected(ids):
    with pytest.raises(ValueError):
        measurement.score(np.array(ids), np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)[:1],
                          np.array([[0, 1]]), np.array([[0, 1]]), "normalized")


def test_recall_selection_never_interpolates_or_drops_unreachable_targets():
    rows = [{"recall": 0.95 - 1e-8, "median_qps": 100},
            {"recall": 0.95, "median_qps": 50}, {"recall": 0.95 + 1e-8, "median_qps": 40}]
    assert measurement.select(rows, 0.95) is rows[1]
    assert measurement.select(rows, 1.0) is None


def test_subset_truth_is_recomputed_and_euclidean_norms_are_preserved():
    train = np.zeros((100, 2), dtype=np.float32)
    train[:, 0] = np.arange(1, 101)
    queries = np.array([[99.9, 0]], dtype=np.float32)
    ids, distances = preparation.exact_truth(train, queries, "euclidean")
    assert ids[0, 0] == 99
    assert distances[0, 0] == pytest.approx(0.1, abs=1e-3)
    assert np.array_equal(np.sort(ids[0]), np.arange(100))
    cosine_ids, cosine_distances = preparation.exact_truth(train, queries, "cosine")
    assert np.array_equal(np.sort(cosine_ids[0]), np.arange(100))
    assert np.max(np.abs(cosine_distances)) < 1e-6


def test_leaderboard_keeps_missing_datasets_in_denominator(tmp_path):
    import duckdb

    leaderboard = load("leaderboard")
    (tmp_path / "dataset_basics.csv").write_text("dataset,type\na,in-distribution\nb,out-of-distribution\nc,in-distribution-additional\n")
    (tmp_path / "algorithms_basics.csv").write_text("algorithm,is_gpu\nx,false\ny,false\ngpu,true\n")
    connection = duckdb.connect()
    connection.execute("create table source(k integer,dataset varchar,algorithm varchar,qps double,recall double)")
    connection.executemany("insert into source values (100,?,?,?,?)", [
        ("a", "x", 100, .95), ("a", "y", 100, .95), ("b", "y", 100, .95),
        ("a", "gpu", 10000, 1.), ("c", "x", 10000, 1.),
    ])
    connection.execute("copy source to ? (format parquet)", [str(tmp_path / "summary.parquet")])
    connection.close()
    result = leaderboard.summarize(tmp_path)
    rows = result["thresholds"]["0.95"]
    assert [row["algorithm"] for row in rows] == ["y", "x"]
    assert rows[0]["mean_normalized_qps"] == 1
    assert rows[1]["mean_normalized_qps"] == .5
    assert rows[1]["datasets_reached"] == 1 and rows[1]["datasets"] == 2
    assert all(row["mean_normalized_qps"] == 0 for row in result["thresholds"]["1.0"])
    with pytest.raises(ValueError, match="no k=10"):
        leaderboard.summarize(tmp_path, k=10)
