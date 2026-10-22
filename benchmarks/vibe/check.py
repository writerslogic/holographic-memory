# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independently check saved predictions, timings and a sample of the exact oracle."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(result_path, data_directory):
    result = json.loads(result_path.read_text())
    manifest = result["input"]
    source = data_directory / f"{manifest['dataset']}-n{manifest['rows']}.hdf5"
    if sha(source) != manifest["sha256"] or sha(result_path.with_suffix(".npz")) != result["predictions_sha256"]:
        raise ValueError("input or predictions hash mismatch")
    k = result["k"]
    with h5py.File(source, "r") as handle:
        train = handle["train"][:].astype(np.float64)
        queries = handle["test"][:].astype(np.float64)
        truth = handle["neighbors"][:, :k]
        distances = handle["distances"][:, :k]
        metric = str(handle.attrs["distance"])
    if metric == "cosine":
        train /= np.linalg.norm(train, axis=1)[:, None]
        queries /= np.linalg.norm(queries, axis=1)[:, None]
    spot_ids = sorted(set(np.linspace(0, len(queries) - 1, 4, dtype=int).tolist()))
    for query_id in spot_ids:
        query = queries[query_id]
        values = np.linalg.norm(train - query, axis=1) if metric == "euclidean" else np.clip(1 - train @ query, 0, 2)
        threshold = np.partition(values, k - 1)[k - 1]
        if not np.isclose(threshold, distances[query_id, -1], atol=2e-6, rtol=0):
            raise ValueError("exact oracle distance disagrees with original vectors")
        if np.any(values[truth[query_id]] > threshold + 2e-6):
            raise ValueError("oracle IDs exceed exact neighbor boundary")
    with np.load(result_path.with_suffix(".npz"), allow_pickle=False) as raw:
        if len(raw.files) != 2 * len(result["rows"]):
            raise ValueError("missing or extra raw configurations")
        for index, row in enumerate(result["rows"]):
            predictions, ns = raw[f"ids_{index}"], raw[f"ns_{index}"]
            if predictions.shape != (len(queries), k) or predictions.dtype.kind not in "iu":
                raise ValueError("invalid prediction shape or type")
            if np.any(predictions < -1) or np.any(predictions >= len(train)):
                raise ValueError("prediction outside corpus")
            if ns.shape != (result["protocol"]["repeats"], len(queries)) or np.any(ns <= 0):
                raise ValueError("invalid raw timings")
            per_query, strict = [], []
            for query_id, ids in enumerate(predictions):
                found = ids[ids >= 0]
                if len(np.unique(found)) != len(found):
                    raise ValueError("duplicate result ID")
                candidates, query = train[found], queries[query_id]
                values = (np.linalg.norm(candidates - query, axis=1) if metric == "euclidean"
                          else np.clip(1 - candidates @ query, 0, 2))
                per_query.append(float(np.count_nonzero(values <= distances[query_id, -1] + 1e-3) / k))
                strict.append(float(np.count_nonzero(np.isin(found, truth[query_id])) / k))
            for field, values in (("recall", per_query), ("strict_id_recall", strict)):
                if not np.isclose(np.mean(values), row[field], atol=1e-12, rtol=0):
                    raise ValueError(f"reported {field} differs from raw predictions")
            qps = len(queries) * 1e9 / ns.sum(axis=1)
            if not np.allclose(qps, row["qps_runs"], rtol=1e-12):
                raise ValueError("reported QPS differs from raw timings")
            if not np.isclose(np.median(qps), row["median_qps"], rtol=1e-12):
                raise ValueError("reported QPS median differs from raw timings")
    for threshold, selected in result["targets"].items():
        eligible = [row for row in result["rows"] if row["recall"] >= float(threshold)]
        if bool(eligible) != (selected is not None):
            raise ValueError("unreached target was dropped or fabricated")
        if eligible and selected["median_qps"] != max(row["median_qps"] for row in eligible):
            raise ValueError("selected row is not the fastest eligible row")
    return {"file": result_path.name, "sha256": sha(result_path), "passed": True,
            "configurations": len(result["rows"]), "oracle_spot_query_ids": spot_ids,
            "oracle_scope": "four queries against all original prefix vectors; every saved prediction and timing checked"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(args.results.glob("*.json")):
        if "method" in json.loads(path.read_text()):
            records.append(check(path, args.data))
    if not records:
        raise ValueError("no completed results")
    args.out.write_text(json.dumps({"passed": True, "results": records}, indent=2) + "\n")
