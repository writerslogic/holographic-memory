# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import h5py
import numpy as np
import pytest

from qag_hdf5 import export
from qag_prepare import prepare as prepare_validation
from qag_training import prepare as prepare_training


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "source.hdf5"
    row = np.arange(101, dtype=np.float32)
    base = np.column_stack([((row * (axis + 3) + axis) % 29 - 14) * (1 + row % 5)
                            for axis in range(8)]).astype(np.float32)
    learn = ((np.arange(96).reshape(12, 8) % 13 - 6) / 4).astype(np.float32)
    with h5py.File(source, "w") as handle:
        handle.attrs.update(distance="ip", dimension=8)
        handle.create_dataset("train", data=base)
        handle.create_dataset("learn", data=learn)
        handle.create_dataset("test", data=np.full((2, 8), np.nan, dtype=np.float32))
        handle.create_dataset("neighbors", data=np.full((2, 100), 999999, dtype=np.uint32))
    data = tmp_path / "validation"
    prepare_validation(source, data, 101, 3, 24301)
    return source, data, base, learn


def scalar_truth(base, queries):
    all_ids, all_scores = [], []
    for query in queries:
        products = [sum(float(x) * float(y) for x, y in zip(vector, query)) for vector in base]
        ids = sorted(range(len(base)), key=lambda index: (-products[index], index))[:100]
        all_ids.append(ids)
        all_scores.append([products[index] for index in ids])
    return np.asarray(all_ids, dtype=np.uint32), np.asarray(all_scores, dtype=np.float64)


def corrupt(path):
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)


def test_original_ip_training_and_validation_export_preserves_split(prepared, tmp_path):
    source, data, base, learn = prepared
    original = source.read_bytes()
    training, repeated = tmp_path / "training", tmp_path / "repeated"
    prepare_training(data, source, training, 4)
    prepare_training(data, source, repeated, 4)
    assert {path.name: path.read_bytes() for path in training.iterdir()} == {
        path.name: path.read_bytes() for path in repeated.iterdir()
    }
    training_ids = np.fromfile(training / "learn_ids.u32", dtype="<u4")
    validation_ids = np.fromfile(data / "validation_ids.u32", dtype="<u4")
    permitted_ids = np.fromfile(data / "construction_ids.u32", dtype="<u4")
    assert len(training_ids) == len(set(training_ids.tolist())) == 4
    assert set(training_ids) <= set(permitted_ids)
    assert not set(training_ids) & set(validation_ids)
    assert set(permitted_ids) | set(validation_ids) == set(range(len(learn)))
    training_queries = np.fromfile(training / "queries.f32", dtype="<f4").reshape(4, 8)
    training_neighbors = np.fromfile(training / "neighbors.u32", dtype="<u4").reshape(4, 100)
    np.testing.assert_array_equal(training_queries, learn[training_ids])
    expected_training, _ = scalar_truth(base, learn[training_ids])
    expected_validation, expected_scores = scalar_truth(base, learn[validation_ids])
    np.testing.assert_array_equal(training_neighbors, expected_training)
    assert all(len(set(ids)) == 100 for ids in training_neighbors.tolist())

    out = tmp_path / "validation-only.hdf5"
    export(data, training, out)
    with h5py.File(out, "r") as handle:
        assert set(handle) == {"train", "test", "neighbors", "distances", "learn", "learn_neighbors"}
        assert handle.attrs["distance"] == "ip"
        assert handle.attrs["dimension"] == 8
        assert handle.attrs["point_type"] == "float"
        np.testing.assert_array_equal(handle["train"][:], base)
        np.testing.assert_array_equal(handle["test"][:], learn[validation_ids])
        np.testing.assert_array_equal(handle["learn"][:], learn[training_ids])
        np.testing.assert_array_equal(handle["neighbors"][:], expected_validation)
        np.testing.assert_array_equal(handle["distances"][:], -expected_scores)
        np.testing.assert_array_equal(handle["learn_neighbors"][:], expected_training)
        assert all(np.isfinite(handle[name][:]).all() for name in handle)
    report = json.loads(out.with_suffix(".json").read_text())
    assert report["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert report["base_rows"] == 101
    assert report["validation_queries"] == 3
    assert report["construction_queries"] == 4
    assert source.read_bytes() == original


@pytest.mark.parametrize("count,error", [(0, "invalid training count"), (1, None),
                                        (32768, "insufficient construction rows"),
                                        (32769, "invalid training count")])
def test_training_count_boundaries(prepared, tmp_path, count, error):
    source, data, _, _ = prepared
    out = tmp_path / "training"
    if error is None:
        prepare_training(data, source, out, count)
        assert np.fromfile(out / "learn_ids.u32", dtype="<u4").size == 1
    else:
        with pytest.raises(ValueError, match=error):
            prepare_training(data, source, out, count)
        assert not out.exists()


@pytest.mark.parametrize("name", ["base.f32", "validation_ids.u32", "construction_ids.u32"])
def test_training_rejects_corrupted_prepared_matrices_before_output(prepared, tmp_path, name):
    source, data, _, _ = prepared
    corrupt(data / name)
    out = tmp_path / "training"
    with pytest.raises(ValueError, match="digest"):
        prepare_training(data, source, out, 4)
    assert not out.exists()


def test_validation_id_drift_is_rejected_even_with_updated_file_digest(prepared, tmp_path):
    source, data, _, _ = prepared
    validation = data / "validation_ids.u32"
    ids = np.fromfile(validation, dtype="<u4")
    construction = np.fromfile(data / "construction_ids.u32", dtype="<u4")
    ids[0] = construction[0]
    ids.tofile(validation)
    manifest = data / "meta.json"
    meta = json.loads(manifest.read_text())
    meta["files"][validation.name] = hashlib.sha256(validation.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(meta))
    out = tmp_path / "training"
    with pytest.raises(ValueError):
        prepare_training(data, source, out, 4)
    assert not out.exists()


def test_changed_source_is_rejected_before_output(prepared, tmp_path):
    source, data, _, _ = prepared
    with h5py.File(source, "r+") as handle:
        handle["train"][0, 0] += 1
    out = tmp_path / "training"
    with pytest.raises(ValueError, match="source digest"):
        prepare_training(data, source, out, 4)
    assert not out.exists()


@pytest.mark.parametrize("field", ["source_sha256", "data_manifest_sha256"])
def test_export_rejects_inconsistent_source_provenance(prepared, tmp_path, field):
    source, data, _, _ = prepared
    training = tmp_path / "training"
    prepare_training(data, source, training, 4)
    manifest = training / "meta.json"
    meta = json.loads(manifest.read_text())
    meta[field] = "0" * 64
    manifest.write_text(json.dumps(meta))
    out = tmp_path / "validation-only.hdf5"
    with pytest.raises(ValueError, match="provenance"):
        export(data, training, out)
    assert not out.exists()


def test_export_rejects_overlap_even_with_updated_matrix_digest(prepared, tmp_path):
    source, data, _, _ = prepared
    training = tmp_path / "training"
    prepare_training(data, source, training, 4)
    path = training / "learn_ids.u32"
    ids = np.fromfile(path, dtype="<u4")
    ids[0] = np.fromfile(data / "validation_ids.u32", dtype="<u4")[0]
    ids.tofile(path)
    manifest = training / "meta.json"
    meta = json.loads(manifest.read_text())
    meta["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(meta))
    out = tmp_path / "validation-only.hdf5"
    with pytest.raises(ValueError, match="overlap"):
        export(data, training, out)
    assert not out.exists()


@pytest.mark.parametrize("name", ["../outside.f32", "/outside.f32"])
def test_training_rejects_manifest_paths_outside_data_directory(prepared, tmp_path, name):
    source, data, _, _ = prepared
    manifest = data / "meta.json"
    meta = json.loads(manifest.read_text())
    meta["files"][name] = "0" * 64
    manifest.write_text(json.dumps(meta))
    out = tmp_path / "training"
    with pytest.raises(ValueError, match="digest"):
        prepare_training(data, source, out, 4)
    assert not out.exists()


def test_training_rejects_nonfinite_base_even_with_updated_digest(prepared, tmp_path):
    source, data, _, _ = prepared
    path = data / "base.f32"
    base = np.fromfile(path, dtype="<f4")
    base[0] = np.nan
    base.tofile(path)
    manifest = data / "meta.json"
    meta = json.loads(manifest.read_text())
    meta["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(meta))
    out = tmp_path / "training"
    with pytest.raises(ValueError, match="nonfinite"):
        prepare_training(data, source, out, 4)
    assert not out.exists()


@pytest.mark.parametrize("folder,name", [("validation", "base.f32"), ("validation", "scores.f64"),
                                        ("training", "queries.f32"), ("training", "neighbors.u32"),
                                        ("training", "learn_ids.u32")])
def test_export_rejects_corrupted_matrices_before_output(prepared, tmp_path, folder, name):
    source, data, _, _ = prepared
    training = tmp_path / "training"
    prepare_training(data, source, training, 4)
    corrupt((data if folder == "validation" else training) / name)
    out = tmp_path / "validation-only.hdf5"
    with pytest.raises(ValueError, match="digest"):
        export(data, training, out)
    assert not out.exists()
    assert not out.with_suffix(".json").exists()
