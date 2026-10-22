# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Check paired encoding diagnostics against original vectors and raw predictions."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from check import sha


TARGETS = (0.73, 0.77, 0.90, 0.94, 0.95, 0.99, 1.0)
ARMS = ("plain4", "rotated4", "plain8", "rotated8")


def select(rows, target, metric):
    eligible = [row for row in rows if row[metric] >= target]
    if not eligible:
        return None
    row = max(eligible, key=lambda row: row["median_qps"])
    return {key: row[key] for key in ("ef", "rerank", "recall", "strict_id_recall", "median_qps", "qps_runs")}


def check(path, directory):
    result = json.loads(path.read_text())
    meta = result["data"]
    manifest = meta["input_manifest"]
    source = directory / f"{manifest['dataset']}-n{manifest['rows']}.hdf5"
    if sha(source) != manifest["sha256"] or result["k"] != 100:
        raise ValueError("input hash or k differs")
    with h5py.File(source, "r") as handle:
        train = handle["train"][:].astype(np.float64)
        queries = handle["test"][:].astype(np.float64)
        truth, distances = handle["neighbors"][:], handle["distances"][:]
        metric = str(handle.attrs["distance"])
    if metric == "cosine":
        train /= np.linalg.norm(train, axis=1)[:, None]
        queries /= np.linalg.norm(queries, axis=1)[:, None]
    elif metric != "normalized":
        raise ValueError("rotation diagnostic only supports angular data")
    nq, k = truth.shape
    if nq != result["n_queries"] or train.shape[1] != meta["dim"]:
        raise ValueError("matrix shape differs")
    spot_ids = sorted(set(np.linspace(0, nq - 1, 4, dtype=int).tolist()))
    for qi in spot_ids:
        values = np.clip(1 - train @ queries[qi], 0, 2)
        threshold = np.partition(values, k - 1)[k - 1]
        if not np.isclose(threshold, distances[qi, -1], atol=2e-6, rtol=0):
            raise ValueError("oracle distance differs from original vectors")
        if np.any(values[truth[qi]] > threshold + 2e-6):
            raise ValueError("oracle IDs exceed exact neighbor boundary")
    prediction_path = path.with_suffix(".ids.u32")
    predictions = np.fromfile(prediction_path, dtype="<u4")
    rows = result["rows"]
    if len(predictions) != len(rows) * nq * k:
        raise ValueError("raw prediction length differs")
    if set(arm["name"] for arm in result["arms"]) != set(ARMS):
        raise ValueError("missing experimental arm")
    grids = {arm: [] for arm in ARMS}
    for i, row in enumerate(rows):
        if row["prediction_row_offset"] != i * nq * k:
            raise ValueError("prediction offset differs")
        grids[row["arm"]].append((row["ef"], row["rerank"]))
        ids = predictions[i * nq * k:(i + 1) * nq * k].reshape(nq, k)
        valid = ids != np.iinfo(np.uint32).max
        if np.any(ids[valid] >= len(train)):
            raise ValueError("prediction outside corpus")
        hits, strict = 0, 0
        for qi, (found, mask) in enumerate(zip(ids, valid)):
            found = found[mask]
            if len(np.unique(found)) != len(found):
                raise ValueError("duplicate result ID")
            values = np.clip(1 - train[found] @ queries[qi], 0, 2)
            hits += np.count_nonzero(values <= distances[qi, -1] + 1e-3)
            strict += np.count_nonzero(np.isin(found, truth[qi]))
        for key, value in (("recall", hits / (nq * k)), ("strict_id_recall", strict / (nq * k))):
            if not np.isclose(value, row[key], atol=1e-12, rtol=0):
                raise ValueError(f"reported {key} differs from predictions")
        if int((~valid).sum()) != row["missing_results"]:
            raise ValueError("reported missing count differs from predictions")
        ns = np.asarray(row["query_ns"])
        if ns.shape != (result["repeats"], nq) or ns.dtype.kind not in "iu" or np.any(ns <= 0):
            raise ValueError("invalid raw timing matrix")
        qps = nq * 1e9 / ns.sum(axis=1)
        if not np.allclose(qps, row["qps_runs"], rtol=1e-12):
            raise ValueError("QPS differs from raw timings")
        if not np.isclose(np.median(qps), row["median_qps"], rtol=1e-12):
            raise ValueError("QPS median differs from raw timings")
    if any(grid != grids["plain4"] for grid in grids.values()):
        raise ValueError("experimental search grids differ")
    summary = []
    for arm in result["arms"]:
        arm_rows = [row for row in rows if row["arm"] == arm["name"]]
        summary.append({**arm, "max_recall": max(row["recall"] for row in arm_rows),
                        "max_strict_id_recall": max(row["strict_id_recall"] for row in arm_rows),
                        "targets": {str(target): select(arm_rows, target, "recall") for target in TARGETS},
                        "strict_targets": {str(target): select(arm_rows, target, "strict_id_recall") for target in TARGETS}})
    return {"passed": True, "file": path.name, "sha256": sha(path), "predictions_sha256": sha(prediction_path),
            "input_sha256": manifest["sha256"], "graph_hash": result["graph_hash"], "configurations": len(rows),
            "scope": "100k corpus prefix, 256 original queries; encoding-only comparison, no leaderboard claim",
            "oracle_spot_query_ids": spot_ids, "arms": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checks = [check(path, args.data) for path in args.results]
    args.out.write_text(json.dumps({"passed": True, "checker_sha256": sha(Path(__file__)), "results": checks}, indent=2) + "\n")
