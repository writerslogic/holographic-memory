# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("qag_prepare", Path(__file__).with_name("qag_prepare.py"))
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "variable-norm-ip.hdf5"
    row = np.arange(101, dtype=np.float32)
    base = np.column_stack((-(row + 1), row % 7, (row % 11) - 5))
    base[0] = [2, 0, 0]
    base[1] = [40, 30, 0]
    learn = np.array([[1, 0, 0], [2, 0, 0], [0.5, 0, 0], [3, 1, 0],
                      [-1, 0.5, 2], [-2, -1, 1], [1, -3, 0.5], [0.25, 0, 0]], dtype=np.float32)
    with h5py.File(path, "w") as data:
        data.attrs["distance"] = "ip"
        data.attrs["dimension"] = 3
        data.create_dataset("train", data=base)
        data.create_dataset("learn", data=learn)
        data.create_dataset("test", data=np.full((2, 3), np.nan, dtype=np.float32))
        data.create_dataset("neighbors", data=np.full((2, 100), 999999, dtype=np.uint32))
    return path, base, learn


def read_prepared(path, rows, queries, dim=3):
    return {
        "base": np.fromfile(path / "base.f32", dtype="<f4").reshape(rows, dim),
        "validation": np.fromfile(path / "validation.f32", dtype="<f4").reshape(queries, dim),
        "truth": np.fromfile(path / "truth.u32", dtype="<u4").reshape(queries, 100),
        "scores": np.fromfile(path / "scores.f64", dtype="<f8").reshape(queries, 100),
        "validation_ids": np.fromfile(path / "validation_ids.u32", dtype="<u4"),
        "construction_ids": np.fromfile(path / "construction_ids.u32", dtype="<u4"),
        "meta": json.loads((path / "meta.json").read_text()),
    }


@pytest.mark.parametrize("rows", [99, 100, 101])
def test_top100_prefix_boundary_uses_original_inner_products(source, tmp_path, rows):
    path, base, learn = source
    out = tmp_path / "prepared"
    if rows == 99:
        with pytest.raises(ValueError, match="bounds"):
            preparation.prepare(path, out, rows, 7, 24301)
        assert not out.exists()
        return

    preparation.prepare(path, out, rows, 7, 24301)
    result = read_prepared(out, rows, 7)
    np.testing.assert_array_equal(result["base"], base[:rows])
    np.testing.assert_array_equal(result["validation"], learn[result["validation_ids"]])
    assert set(result["truth"].ravel()) <= set(range(rows))

    for query, ids, scores in zip(result["validation"], result["truth"], result["scores"]):
        products = [sum(float(x) * float(y) for x, y in zip(vector, query)) for vector in base[:rows]]
        expected = sorted(range(rows), key=lambda index: (-products[index], index))[:100]
        assert ids.tolist() == expected
        np.testing.assert_array_equal(scores, [products[index] for index in expected])
        assert len(set(ids.tolist())) == 100
        if np.array_equal(query, [1, 0, 0]) or np.array_equal(query, [2, 0, 0]):
            assert ids[0] == 1
            assert products[1] > products[0]
            assert base[0, 0] / np.linalg.norm(base[0]) > base[1, 0] / np.linalg.norm(base[1])

    assert result["meta"]["normalization"] == "none"
    assert result["meta"]["source_metric"] == "ip"
    assert result["meta"]["official_test_read"] is False


def test_split_is_deterministic_disjoint_and_preserves_source(source, tmp_path):
    path, _, learn = source
    original = path.read_bytes()
    expected_hash = hashlib.sha256(original).hexdigest()
    first, second = tmp_path / "first", tmp_path / "second"
    preparation.prepare(path, first, 101, 4, 24301)
    preparation.prepare(path, second, 101, 4, 24301)
    result = read_prepared(first, 101, 4)

    validation_ids = result["validation_ids"].tolist()
    construction_ids = result["construction_ids"].tolist()
    assert len(validation_ids) == len(set(validation_ids)) == 4
    assert len(construction_ids) == len(set(construction_ids)) == len(learn) - 4
    assert not set(validation_ids) & set(construction_ids)
    assert set(validation_ids) | set(construction_ids) == set(range(len(learn)))
    assert construction_ids == sorted(construction_ids)
    np.testing.assert_array_equal(result["validation"], learn[validation_ids])
    assert {file.name: file.read_bytes() for file in first.iterdir()} == {
        file.name: file.read_bytes() for file in second.iterdir()
    }
    assert result["meta"]["source_sha256"] == expected_hash
    assert path.read_bytes() == original
    for name, checksum in result["meta"]["files"].items():
        assert hashlib.sha256((first / name).read_bytes()).hexdigest() == checksum


@pytest.mark.parametrize("rows,queries", [(1_000_001, 1), (100, 0), (100, 1001), (102, 1), (100, 8), (100, 9)])
def test_invalid_prefix_and_query_bounds_are_rejected(source, tmp_path, rows, queries):
    path, _, _ = source
    out = tmp_path / "prepared"
    with pytest.raises(ValueError):
        preparation.prepare(path, out, rows, queries, 24301)
    assert not out.exists()


@pytest.mark.parametrize("metric", ["cosine", "normalized", "euclidean"])
def test_non_ip_metric_is_rejected(source, tmp_path, metric):
    path, _, _ = source
    with h5py.File(path, "r+") as data:
        data.attrs["distance"] = metric
    out = tmp_path / "prepared"
    with pytest.raises(ValueError, match="inner-product"):
        preparation.prepare(path, out, 101, 4, 24301)
    assert not out.exists()


def test_missing_learning_pool_is_rejected_without_falling_back_to_test(source, tmp_path):
    path, _, _ = source
    with h5py.File(path, "r+") as data:
        del data["learn"]
    out = tmp_path / "prepared"
    with pytest.raises(ValueError, match="separate learning queries"):
        preparation.prepare(path, out, 101, 4, 24301)
    assert not out.exists()


@pytest.mark.parametrize("dimension", [0, 2, 8193])
def test_invalid_dimension_or_shape_is_rejected(source, tmp_path, dimension):
    path, _, _ = source
    with h5py.File(path, "r+") as data:
        data.attrs["dimension"] = dimension
    out = tmp_path / "prepared"
    with pytest.raises(ValueError, match="dimensions|shape"):
        preparation.prepare(path, out, 101, 4, 24301)
    assert not out.exists()


@pytest.mark.parametrize("dataset,value", [("train", np.inf), ("train", np.nan), ("learn", np.inf), ("learn", np.nan)])
def test_nonfinite_construction_or_validation_input_is_rejected(source, tmp_path, dataset, value):
    path, _, _ = source
    with h5py.File(path, "r+") as data:
        data[dataset][:] = value
    out = tmp_path / "prepared"
    with pytest.raises(ValueError, match="nonfinite"):
        preparation.prepare(path, out, 101, 4, 24301)
    assert not out.exists()


def test_existing_evidence_is_not_overwritten(source, tmp_path):
    path, _, _ = source
    out = tmp_path / "prepared"
    out.mkdir()
    evidence = out / "existing"
    evidence.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        preparation.prepare(path, out, 101, 4, 24301)
    assert evidence.read_bytes() == b"preserve"
    assert list(out.iterdir()) == [evidence]
