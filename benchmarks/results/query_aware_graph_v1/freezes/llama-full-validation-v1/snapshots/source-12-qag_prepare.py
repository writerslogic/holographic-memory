# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepare query-graph diagnostics exclusively from VIBE's learning-query split."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def prepare(source, out, rows, queries, seed):
    if out.exists():
        raise FileExistsError(out)
    if not 100 <= rows <= 1_000_000 or not 1 <= queries <= 1000:
        raise ValueError("invalid corpus/query bounds")
    with h5py.File(source, "r") as data:
        if data.attrs["distance"] != "ip" or "learn" not in data:
            raise ValueError("requires inner-product data and separate learning queries")
        dim = int(data.attrs["dimension"])
        if not 1 <= dim <= 8192 or rows > len(data["train"]) or rows * dim > (1 << 29):
            raise ValueError("invalid dimensions or corpus prefix")
        if queries >= len(data["learn"]):
            raise ValueError("validation must leave construction queries")
        selected = np.random.default_rng(seed).permutation(len(data["learn"]))
        validation_ids = selected[:queries]
        # HDF5 fancy indexing requires sorted row IDs; retain the seeded order explicitly.
        ordered = np.argsort(validation_ids)
        validation = data["learn"][np.sort(validation_ids)][np.argsort(ordered)]
        base = np.asarray(data["train"][:rows], dtype="<f4")
        original_rows = len(data["train"])
        official_test_shape = list(data["test"].shape)
        learn_rows = len(data["learn"])
    if base.shape != (rows, dim) or validation.shape != (queries, dim):
        raise ValueError("invalid source shape")
    if not np.isfinite(base).all() or not np.isfinite(validation).all():
        raise ValueError("nonfinite input")
    base64 = base.astype(np.float64)
    norms2 = np.einsum("ij,ij->i", base64, base64)
    extra = np.sqrt(np.maximum(norms2.max() - norms2, 0)).astype(np.float32)
    augmented = np.column_stack((base, extra)).astype(np.float64)
    truth, scores = [], []
    transform_checks = []
    for i, query in enumerate(validation.astype(np.float64)):
        products = base64 @ query
        candidates = np.flatnonzero(products >= np.partition(products, -100)[-100])
        ids = candidates[np.lexsort((candidates, -products[candidates]))[:100]]
        truth.append(ids)
        scores.append(products[ids])
        if i < 4:
            # Independent scalar sums verify direction, IDs, and the asymmetric transform.
            spot = np.array([sum(float(a) * float(b) for a, b in zip(base[j], query)) for j in ids])
            np.testing.assert_allclose(spot, products[ids], rtol=1e-12, atol=1e-9)
            distances = np.sum((augmented - np.append(query, 0)) ** 2, axis=1)
            transformed = np.argsort(distances, kind="stable")[:100]
            overlap = len(set(ids.tolist()) & set(transformed.tolist())) / 100
            transform_checks.append({"query": i, "top100_overlap_after_f32_augmentation": overlap,
                                     "max_distance_identity_error": float(np.max(np.abs(
                                         distances - (query @ query + norms2.max() - 2 * products))))})
    out.mkdir(parents=True)
    for name, array, dtype in (("base.f32", base, "<f4"), ("validation.f32", validation, "<f4"),
                               ("truth.u32", truth, "<u4"), ("scores.f64", scores, "<f8"),
                               ("validation_ids.u32", validation_ids, "<u4")):
        np.asarray(array, dtype=dtype).tofile(out / name)
    construction_ids = np.sort(selected[queries:])
    np.asarray(construction_ids, dtype="<u4").tofile(out / "construction_ids.u32")
    manifest = {
        "schema_version": 1, "source": str(source), "source_sha256": digest(source),
        "source_metric": "ip", "dimension": dim, "rows": rows, "original_rows": original_rows,
        "validation_queries": queries, "seed": seed, "learn_rows": learn_rows,
        "official_test_shape": official_test_shape, "official_test_read": False,
        "validation_selection": "numpy default_rng(seed).permutation(learn_rows)[:queries]",
        "construction_selection": "remaining learn row IDs, disjoint from validation",
        "truth": "float64 original inner product, descending score then ascending base ID",
        "normalization": "none", "transform_checks": transform_checks,
        "files": {p.name: digest(p) for p in sorted(out.iterdir())},
        "preparer_sha256": digest(Path(__file__)),
    }
    (out / "meta.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=200000)
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--seed", type=int, default=24301)
    args = parser.parse_args()
    prepare(args.source, args.out, args.rows, args.queries, args.seed)
