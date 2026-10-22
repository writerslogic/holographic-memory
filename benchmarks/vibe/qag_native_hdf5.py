# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Expose full native ANN data to VIBE while retaining its saved ground-truth IDs.

Only distances to saved neighbors are computed; nearest neighbors are not recomputed.
By default train/test/neighbors use HDF5 external raw storage, so the native files
must remain at their recorded absolute paths. --copy-vectors makes a standalone file.
Run with uv run --no-project --with h5py --with numpy qag_native_hdf5.py ...
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from qag_report import digest, metric_distances, validate_ids


METRICS = {"angular": "cosine", "cosine": "cosine", "normalized": "normalized",
           "euclidean": "euclidean", "ip": "ip"}


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def native_matrix(folder, name, shape, dtype):
    path = (folder / name).resolve(strict=True)
    size = int(np.prod(shape, dtype=object)) * np.dtype(dtype).itemsize
    if size > np.iinfo(np.intp).max or not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"{name} byte length does not match its metadata")
    return path, np.memmap(path, dtype=dtype, mode="r", shape=shape)


def check_vectors(rows, metric, name):
    for offset in range(0, len(rows), 4096):
        block = rows[offset:offset + 4096]
        if not np.isfinite(block).all():
            raise ValueError(f"{name} contains nonfinite coordinates")
        if metric in ("angular", "cosine", "normalized"):
            norms = np.linalg.norm(block.astype(np.float64), axis=1)
            if np.any(norms == 0):
                raise ValueError(f"{name} contains zero cosine vectors")
            if metric == "normalized" and not np.allclose(norms, 1, atol=2e-4, rtol=0):
                raise ValueError(f"{name} violates the normalized metric")


def export(native, out, copy_vectors=False):
    native = Path(native).resolve(strict=True)
    out = Path(out)
    receipt = out.with_suffix(".json")
    partial = out.with_name(out.name + ".part")
    receipt_partial = receipt.with_name(receipt.name + ".part")
    if any(path.exists() for path in (out, receipt, partial, receipt_partial)):
        raise FileExistsError("export output or an unfinished export already exists")
    meta_path = native / "meta.json"
    meta = json.loads(meta_path.read_text())
    rows = positive_integer(meta["n_train"], "n_train")
    queries = positive_integer(meta["n_test"], "n_test")
    dim = positive_integer(meta["dim"], "dim")
    width = positive_integer(meta["n_neighbors"], "n_neighbors")
    if rows > np.iinfo(np.int32).max or width > rows:
        raise ValueError("saved int32 neighbor IDs cannot represent this dataset")
    metric = meta["metric"]
    if metric not in METRICS:
        raise ValueError(f"unsupported original metric {metric!r}")
    specs = (
        ("train", "train.f32", (rows, dim), "<f4"),
        ("test", "test.f32", (queries, dim), "<f4"),
        ("neighbors", "neighbors.i32", (queries, width), "<i4"),
    )
    matrices = {}
    paths = {}
    for name, filename, shape, dtype in specs:
        paths[name], matrices[name] = native_matrix(native, filename, shape, dtype)
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns)
              for path in (meta_path, *paths.values())}
    check_vectors(matrices["train"], metric, "train")
    check_vectors(matrices["test"], metric, "test")
    distances = np.empty((queries, width), dtype="<f8")
    for offset in range(0, queries, 256):
        ids = validate_ids(matrices["neighbors"][offset:offset + 256],
                           (min(256, queries - offset), width), rows, "saved ground truth")
        for relative, row in enumerate(ids):
            query = offset + relative
            distances[query] = metric_distances(
                metric, matrices["test"][query], matrices["train"][row]
            )
    if not np.isfinite(distances).all():
        raise ValueError("saved-neighbor distances are nonfinite")
    if np.any(distances[:, 1:] < distances[:, :-1] - 1e-5):
        raise ValueError("saved ground-truth IDs are not ordered by the original metric")
    inputs = {
        path.name: {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
        for path in (meta_path, *paths.values())
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(partial, "x") as handle:
        handle.attrs.update(
            dimension=dim, distance=METRICS[metric], point_type="float",
            original_dataset=meta["name"], original_metric=metric,
            source=meta.get("source", ""), source_sha256=meta.get("sha256", ""),
            external_raw_storage=not copy_vectors,
        )
        for name, filename, shape, dtype in specs:
            if copy_vectors:
                dataset = handle.create_dataset(name, shape=shape, dtype=dtype)
                for offset in range(0, shape[0], 4096):
                    dataset[offset:offset + 4096] = matrices[name][offset:offset + 4096]
            else:
                # External raw storage exposes existing bytes without allocating or writing them.
                # https://docs.h5py.org/en/stable/high/group.html#h5py.Group.create_dataset
                handle.create_dataset(
                    name, shape=shape, dtype=dtype,
                    external=[(str(paths[name]), 0, paths[name].stat().st_size)],
                )
        handle.create_dataset("distances", data=distances)
    if any((path.stat().st_size, path.stat().st_mtime_ns) != snapshot
           for path, snapshot in before.items()):
        raise ValueError("native input changed during export")
    with h5py.File(partial, "r") as handle:
        for name, _, shape, _ in specs:
            for offset in sorted({0, shape[0] - 1}):
                if not np.array_equal(handle[name][offset], matrices[name][offset]):
                    raise ValueError("exported dataset does not preserve native row coordinates")
    report = {
        "schema_version": 1, "scope": "full original base and official test; regression only",
        "dataset": meta["name"], "vibe_dataset": out.stem,
        "original_metric": metric, "metric": METRICS[metric],
        "dimension": dim, "rows": rows, "queries": queries, "neighbors": width,
        "ground_truth": "saved full-dataset neighbor IDs, unchanged; only their original-metric distances recomputed in float64",
        "ground_truth_limit": "saved IDs trusted from the recorded source; this export does not recompute full-corpus nearest neighbors",
        "storage": "standalone" if copy_vectors else "external native raw files at immutable absolute paths",
        "inputs": inputs, "source": meta.get("source"), "source_sha256": meta.get("sha256"),
        "sha256": digest(partial), "bytes": partial.stat().st_size,
        "exporter_sha256": digest(Path(__file__)),
    }
    with receipt_partial.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    partial.rename(out)
    receipt_partial.rename(receipt)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--copy-vectors", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args.native, args.out, args.copy_vectors)), flush=True)
