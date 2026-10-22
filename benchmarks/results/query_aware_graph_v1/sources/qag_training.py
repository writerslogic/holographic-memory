# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Recompute construction-query neighborhoods; exclude reserved validation rows."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from qag_prepare import digest


def prepare(data, source, out, count):
    if out.exists():
        raise FileExistsError(out)
    meta = json.loads((data / "meta.json").read_text())
    if not 1 <= count <= 32768 or digest(source) != meta["source_sha256"]:
        raise ValueError("invalid training count or source digest")
    for name, checksum in meta["files"].items():
        if Path(name).name != name or digest(data / name) != checksum:
            raise ValueError("prepared matrix digest mismatch")
    dim, rows = meta["dimension"], meta["rows"]
    if not 1 <= dim <= 8192 or not 100 <= rows <= 1_000_000 or rows * dim > (1 << 29):
        raise ValueError("invalid prepared matrix dimensions")
    base = np.fromfile(data / "base.f32", dtype="<f4").reshape(rows, dim)
    if not np.isfinite(base).all():
        raise ValueError("nonfinite prepared base")
    validation_ids = np.fromfile(data / "validation_ids.u32", dtype="<u4")
    selected = np.random.default_rng(meta["seed"]).permutation(meta["learn_rows"])
    if not np.array_equal(validation_ids, selected[:len(validation_ids)]):
        raise ValueError("reserved validation IDs differ from seeded split")
    training_ids = selected[len(validation_ids):len(validation_ids) + count]
    if len(training_ids) != count or np.intersect1d(training_ids, validation_ids).size:
        raise ValueError("training/validation overlap or insufficient construction rows")
    with h5py.File(source, "r") as handle:
        order = np.argsort(training_ids)
        queries = np.asarray(handle["learn"][np.sort(training_ids)][np.argsort(order)], dtype="<f4")
    if queries.shape != (count, dim) or not np.isfinite(queries).all():
        raise ValueError("invalid construction queries")
    base64 = base.astype(np.float64)
    neighbors = np.empty((count, 100), dtype=np.uint32)
    for offset in range(0, count, 32):
        products = base64 @ queries[offset:offset + 32].astype(np.float64).T
        thresholds = np.partition(products, -100, axis=0)[-100]
        for column in range(products.shape[1]):
            candidates = np.flatnonzero(products[:, column] >= thresholds[column])
            ranked = candidates[np.lexsort((candidates, -products[candidates, column]))[:100]]
            neighbors[offset + column] = ranked
    checks = []
    for i in range(min(4, count)):
        exact = base64 @ queries[i].astype(np.float64)
        candidates = np.flatnonzero(exact >= np.partition(exact, -100)[-100])
        truth = candidates[np.lexsort((candidates, -exact[candidates]))[:100]]
        overlap = len(set(truth.tolist()) & set(neighbors[i].tolist())) / 100
        if overlap != 1:
            raise ValueError("construction neighborhood failed original-IP spot check")
        checks.append({"construction_row": i, "float64_top100_overlap": overlap})
    if np.any(neighbors < 0) or np.any(neighbors >= rows):
        raise ValueError("construction neighbor outside base")
    out.mkdir(parents=True)
    for name, array, dtype in (("queries.f32", queries, "<f4"),
                               ("neighbors.u32", neighbors, "<u4"),
                               ("learn_ids.u32", training_ids, "<u4")):
        np.asarray(array, dtype=dtype).tofile(out / name)
    report = {"schema_version": 1, "data_manifest_sha256": digest(data / "meta.json"),
              "source_sha256": meta["source_sha256"], "queries": count, "dimension": dim, "rows": rows,
              "neighbors": 100, "selection": "seeded learn permutation immediately after validation",
              "truth": "float64 original-IP block matrix products; descending score then base ID; vector-product spot checks",
              "official_test_read": False, "checks": checks,
              "files": {p.name: digest(p) for p in sorted(out.iterdir())},
              "preparer_sha256": digest(Path(__file__))}
    (out / "meta.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32768)
    args = parser.parse_args()
    prepare(args.data, args.source, args.out, args.count)
