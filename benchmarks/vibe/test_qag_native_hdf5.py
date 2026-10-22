# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json

import h5py
import numpy as np
import pytest

from qag_native_hdf5 import export
from qag_report import digest


def native_data(tmp_path, metric="euclidean"):
    folder = tmp_path / "native"
    folder.mkdir()
    rng = np.random.default_rng(24301)
    train = rng.normal(size=(8, 3)).astype("<f4")
    test = np.stack((train[0], train[-1]))
    if metric in ("angular", "cosine", "normalized"):
        if metric == "normalized":
            train /= np.linalg.norm(train, axis=1, keepdims=True)
            test /= np.linalg.norm(test, axis=1, keepdims=True)
        score = 1 - (test.astype(np.float64) @ train.astype(np.float64).T) / (
            np.linalg.norm(test.astype(np.float64), axis=1)[:, None]
            * np.linalg.norm(train.astype(np.float64), axis=1)[None, :]
        )
        score = np.clip(score, 0, 2)
    elif metric == "ip":
        score = -(test.astype(np.float64) @ train.astype(np.float64).T)
    else:
        score = np.linalg.norm(test.astype(np.float64)[:, None] - train.astype(np.float64)[None], axis=2)
    neighbors = np.argsort(score, axis=1, kind="stable")[:, :4].astype("<i4")
    expected = np.take_along_axis(score, neighbors, axis=1)
    for name, values in (("train.f32", train), ("test.f32", test), ("neighbors.i32", neighbors)):
        values.tofile(folder / name)
    (folder / "meta.json").write_text(json.dumps({
        "name": f"synthetic-3-{metric}", "metric": metric, "dim": 3,
        "n_train": 8, "n_test": 2, "n_neighbors": 4,
        "source": "synthetic exact brute force", "sha256": "0" * 64,
    }))
    return folder, train, test, neighbors, expected


@pytest.mark.parametrize("metric", ["euclidean", "angular", "normalized", "ip"])
@pytest.mark.parametrize("copy_vectors", [False, True])
def test_saved_truth_and_original_coordinates_survive_export(tmp_path, metric, copy_vectors):
    folder, train, test, neighbors, expected = native_data(tmp_path, metric)
    before = {path.name: digest(path) for path in folder.iterdir()}
    out = tmp_path / "fixture.hdf5"
    report = export(folder, out, copy_vectors)
    assert {path.name: digest(path) for path in folder.iterdir()} == before
    with h5py.File(out, "r") as handle:
        assert handle.attrs["dimension"] == 3
        assert handle.attrs["distance"] == ("cosine" if metric == "angular" else metric)
        np.testing.assert_array_equal(handle["train"][:], train)
        np.testing.assert_array_equal(handle["test"][:], test)
        np.testing.assert_array_equal(handle["neighbors"][:], neighbors)
        np.testing.assert_allclose(handle["distances"][:], expected, atol=2e-7, rtol=0)
        assert (handle["train"].external is None) == copy_vectors
    assert report["rows"] == 8 and report["queries"] == 2
    assert report["sha256"] == digest(out)
    assert report["inputs"]["neighbors.i32"]["sha256"] == before["neighbors.i32"]
    assert json.loads(out.with_suffix(".json").read_text()) == report
    with pytest.raises(FileExistsError):
        export(folder, out, copy_vectors)


@pytest.mark.parametrize("invalid", [-1, 8])
def test_ground_truth_ids_reject_outside_boundaries(tmp_path, invalid):
    folder, _, _, neighbors, _ = native_data(tmp_path)
    neighbors[0, 0] = invalid
    neighbors.tofile(folder / "neighbors.i32")
    with pytest.raises(ValueError, match="outside"):
        export(folder, tmp_path / "invalid.hdf5")


def test_ground_truth_accepts_first_and_last_base_ids(tmp_path):
    folder, _, _, neighbors, _ = native_data(tmp_path)
    assert neighbors[0, 0] == 0 and neighbors[1, 0] == 7
    export(folder, tmp_path / "valid.hdf5")


def test_duplicate_truth_is_rejected(tmp_path):
    folder, _, _, neighbors, _ = native_data(tmp_path)
    neighbors[0, 1] = neighbors[0, 0]
    neighbors.tofile(folder / "neighbors.i32")
    with pytest.raises(ValueError, match="duplicate"):
        export(folder, tmp_path / "invalid.hdf5")


def test_truncated_native_matrix_is_rejected(tmp_path):
    folder, _, _, _, _ = native_data(tmp_path)
    source = folder / "train.f32"
    source.write_bytes(source.read_bytes()[:-4])
    with pytest.raises(ValueError, match="byte length"):
        export(folder, tmp_path / "invalid.hdf5")


@pytest.mark.parametrize("name", ["train.f32", "test.f32"])
def test_nonfinite_native_coordinates_are_rejected(tmp_path, name):
    folder, train, test, _, _ = native_data(tmp_path)
    values = train if name == "train.f32" else test
    values[0, 0] = np.nan
    values.tofile(folder / name)
    with pytest.raises(ValueError, match="nonfinite"):
        export(folder, tmp_path / "invalid.hdf5")


def test_misordered_saved_truth_is_rejected(tmp_path):
    folder, _, _, neighbors, _ = native_data(tmp_path)
    neighbors[:, ::-1].tofile(folder / "neighbors.i32")
    with pytest.raises(ValueError, match="ordered"):
        export(folder, tmp_path / "invalid.hdf5")
