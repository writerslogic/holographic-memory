# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Export prepared validation/construction splits for the unmodified VIBE runner."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from qag_prepare import digest


def export(data, training, out):
    if out.exists():
        raise FileExistsError(out)
    meta = json.loads((data / "meta.json").read_text())
    learn = json.loads((training / "meta.json").read_text())
    if (learn["data_manifest_sha256"] != digest(data / "meta.json")
            or learn["source_sha256"] != meta["source_sha256"]):
        raise ValueError("construction/validation provenance mismatch")
    for folder, manifest in ((data, meta), (training, learn)):
        for name, checksum in manifest["files"].items():
            if Path(name).name != name or digest(folder / name) != checksum:
                raise ValueError("matrix digest mismatch")
    dim, rows, nq, nl = meta["dimension"], meta["rows"], meta["validation_queries"], learn["queries"]
    base = np.fromfile(data / "base.f32", dtype="<f4").reshape(rows, dim)
    queries = np.fromfile(data / "validation.f32", dtype="<f4").reshape(nq, dim)
    truth = np.fromfile(data / "truth.u32", dtype="<u4").reshape(nq, 100)
    scores = np.fromfile(data / "scores.f64", dtype="<f8").reshape(nq, 100)
    construction = np.fromfile(training / "queries.f32", dtype="<f4").reshape(nl, dim)
    neighbors = np.fromfile(training / "neighbors.u32", dtype="<u4").reshape(nl, 100)
    construction_ids = np.fromfile(training / "learn_ids.u32", dtype="<u4")
    validation_ids = np.fromfile(data / "validation_ids.u32", dtype="<u4")
    if np.intersect1d(construction_ids, validation_ids).size:
        raise ValueError("construction/validation overlap")
    with h5py.File(out, "x") as handle:
        handle.attrs.update(dimension=dim, distance="ip", point_type="float",
                            source="permitted learn split, validation only; no official test")
        for name, matrix in (("train", base), ("test", queries), ("neighbors", truth),
                             ("distances", -scores), ("learn", construction), ("learn_neighbors", neighbors)):
            handle.create_dataset(name, data=matrix)
    report = {"schema_version": 1, "scope": "VIBE validation-only export, no official test",
              "native_manifest": digest(data / "meta.json"),
              "construction_manifest": digest(training / "meta.json"),
              "sha256": digest(out), "base_rows": rows, "validation_queries": nq,
              "construction_queries": nl, "dimension": dim, "metric": "ip",
              "exporter_sha256": digest(Path(__file__))}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export(args.data, args.training, args.out)
