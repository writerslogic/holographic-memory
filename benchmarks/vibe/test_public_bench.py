# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run with PUBLIC_BENCH set to the compiled public-bench executable."""

import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest


BINARY = os.environ.get("PUBLIC_BENCH")
pytestmark = pytest.mark.skipif(not BINARY, reason="PUBLIC_BENCH executable is required")


def fixture(path, n, width=100):
    path.mkdir()
    (path / "meta.json").write_text(json.dumps({"metric": "angular", "dim": 2, "n_neighbors": width}))
    train = [coordinate for i in range(n) for coordinate in (math.cos(i / n), math.sin(i / n))]
    (path / "train.f32").write_bytes(struct.pack(f"<{len(train)}f", *train))
    (path / "test.f32").write_bytes(struct.pack("<2f", 1, 0))
    (path / "neighbors.i32").write_bytes(struct.pack(f"<{width}i", *range(width)))
    return path


def execute(data, *extra):
    return subprocess.run([BINARY, "ann-qgraph", "--data", str(data), "--out", str(data / "result.json"),
                           "--index", "vertex", "--vertex-bits", "4", "--residual", "--degree", "8",
                           "--build-ef", "32", "--ef", "128", "--rerank", "128", "--repeats", "1",
                           "--max-wait-secs", "0", *extra], text=True, capture_output=True,
                          env=os.environ | {"RAYON_NUM_THREADS": "1"}, timeout=30)


@pytest.mark.parametrize("n", [99, 100, 101])
def test_count100_corpus_boundary(tmp_path, n):
    data = fixture(tmp_path / "data", n)
    result = execute(data, "--count", "100")
    if n < 100:
        assert result.returncode != 0 and "at least 100" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        report = json.loads((data / "result.json").read_text())
        assert report["k"] == 100
        recall = report["sweep"][0]["recall_at_100"]
        assert 0 <= recall <= 1 and recall * 100 == round(recall * 100)
        if n == 100:
            assert recall == 1
        assert "recall_at_10" not in report["sweep"][0]


@pytest.mark.parametrize("width", [99, 100, 101])
def test_count100_ground_truth_boundary(tmp_path, width):
    data = fixture(tmp_path / "data", 101, width)
    result = execute(data, "--count", "100")
    assert (result.returncode == 0) == (width >= 100), result.stderr


def test_default_recall10_and_empty_query_rejection(tmp_path):
    data = fixture(tmp_path / "data", 100)
    result = execute(data)
    assert result.returncode == 0, result.stderr
    report = json.loads((data / "result.json").read_text())
    assert report["k"] == 10 and report["sweep"][0]["recall_at_10"] == 1
    if sys.platform == "linux":
        assert all(value is not None and value >= 0 for value in report["sweep"][0]["load_1m_before_runs"])
    result = execute(data, "--queries", "0")
    assert result.returncode != 0 and "at least one query" in result.stderr


@pytest.mark.parametrize("n", [99, 100, 101])
def test_count100_saved_index_boundary_without_resident_training_vectors(tmp_path, n):
    data = fixture(tmp_path / "data", n)
    index = data / "index.hms"
    built = execute(data, "--save-index", str(index))
    assert built.returncode == 0, built.stderr
    (data / "train.f32").unlink()
    result = execute(data, "--count", "100", "--load-index", str(index), "--residency", "heap")
    if n < 100:
        assert result.returncode != 0 and "at least 100" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads((data / "result.json").read_text())["k"] == 100


def test_zero_dimension_is_rejected_before_loading_vectors(tmp_path):
    data = fixture(tmp_path / "data", 100)
    meta = json.loads((data / "meta.json").read_text())
    meta["dim"] = 0
    (data / "meta.json").write_text(json.dumps(meta))
    result = execute(data, "--count", "100")
    assert result.returncode != 0 and "dimension must be positive" in result.stderr
