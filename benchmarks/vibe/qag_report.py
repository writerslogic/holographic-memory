# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Extract conservative paired results from VIBE's saved fifth query pass."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics

import h5py
import numpy as np


COUNT = 100
EPSILON = 1e-3
ARMS = {0: "baseline4", 1: "candidate8", 2: "roargraph48", 3: "roargraph96", 4: "mlann"}


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def plain(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def validate_ids(values, shape, rows, name, missing=False):
    ids = np.asarray(values)
    if ids.shape != shape or ids.dtype.kind not in "iu":
        raise ValueError(f"{name} must have integer IDs with shape {shape}")
    lower = -1 if missing else 0
    if (ids.dtype.kind == "i" and np.any(ids < lower)) or np.any(ids >= rows):
        raise ValueError(f"{name} contains an ID outside the base range")
    ids = ids.astype(np.int64)
    for row in ids:
        present = row[row >= 0]
        if len(np.unique(present)) != len(present):
            raise ValueError(f"{name} contains duplicate IDs")
    return ids


def metric_distances(metric, query, rows):
    query = np.asarray(query, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(query).all() or not np.isfinite(rows).all():
        raise ValueError("nonfinite source vectors")
    products = np.einsum("ij,j->i", rows, query)
    if metric == "ip":
        return -products
    if metric == "euclidean":
        return np.sqrt(np.sum((rows - query) ** 2, axis=1, dtype=np.float64))
    if metric in ("angular", "cosine"):
        denominator = np.linalg.norm(rows, axis=1) * np.linalg.norm(query)
        if np.any(denominator == 0):
            raise ValueError("cosine distance is undefined for zero source vectors")
        products = products / denominator
    elif metric != "normalized":
        raise ValueError(f"unsupported source metric {metric}")
    return np.clip(1 - products, 0, 2)


def _read_runs(path, queries, rows, metric):
    runs = []
    seen = set()
    with h5py.File(path, "r") as results:
        for name in sorted(results):
            match = re.fullmatch(r"(\d+)_(\d+)_(\d+)_(\d+)", name)
            if match is None or not isinstance(results[name], h5py.Group):
                raise ValueError(f"unrecognized paired result group {name}")
            arm, ef, rerank, repetition = map(int, match.groups())
            if arm not in ARMS or ef <= 0 or repetition not in (0, 1, 2):
                raise ValueError(f"invalid paired query arguments in {name}")
            key = (arm, ef, rerank, repetition)
            if key in seen:
                raise ValueError("duplicate repetition for a paired configuration")
            seen.add(key)
            group = results[name]
            if integer(group.attrs.get("count"), "count") != COUNT:
                raise ValueError("paired results require count=100")
            if integer(group.attrs.get("run_count"), "run_count") != 5:
                raise ValueError("paired results require run_count=5")
            for attr, expected in (("qag_arm", arm), ("qag_repetition", repetition)):
                if attr in group.attrs and integer(group.attrs[attr], attr) != expected:
                    raise ValueError(f"{attr} differs from the group arguments")
            if "distance" in group.attrs:
                recorded = plain(group.attrs["distance"])
                if not isinstance(recorded, str) or recorded.replace("angular", "cosine") != metric.replace("angular", "cosine"):
                    raise ValueError("result and source metrics differ")
            if any(field not in group for field in ("neighbors", "distances", "times")):
                raise ValueError("result group lacks neighbors, distances or times")
            ids = validate_ids(group["neighbors"][:], (queries, COUNT), rows, name, missing=True)
            distances = group["distances"][:]
            if distances.shape != ids.shape or distances.dtype.kind not in "fiu":
                raise ValueError("raw result distances have the wrong shape or type")
            if np.isnan(distances).any() or not np.isfinite(distances[ids >= 0]).all():
                raise ValueError("raw result distances are nonfinite for a returned ID")
            times = np.asarray(group["times"][:], dtype=np.float64)
            if times.shape != (queries,) or not np.isfinite(times).all() or np.any(times <= 0):
                raise ValueError("raw query times must be finite positive values, one per query")
            resources = {
                key: plain(value) for key, value in group.attrs.items()
                if key.startswith("qag_") or key in ("build_time", "index_size", "algo", "dataset", "gpu_mode")
            }
            runs.append({
                "arm": arm, "ef": ef, "rerank": rerank, "repetition": repetition,
                "group": name, "ids": ids, "seconds": float(times.sum(dtype=np.float64)),
                "strict_hits": 0, "epsilon_hits": 0, "resource_attrs": resources,
                "ids_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
            })
    if not runs:
        raise ValueError("no paired result groups")
    configurations = {}
    for run in runs:
        configurations.setdefault((run["arm"], run["ef"], run["rerank"]), []).append(run)
    if any(sorted(run["repetition"] for run in group) != [0, 1, 2] for group in configurations.values()):
        raise ValueError("every configuration requires exactly repetitions 0, 1 and 2")
    return runs, configurations


def extract(results, data, receipt=None):
    results, data = Path(results), Path(data)
    with h5py.File(data, "r") as source:
        if any(field not in source for field in ("train", "test", "neighbors")):
            raise ValueError("source requires train, test and neighbors")
        base, queries = source["train"], np.asarray(source["test"][:], dtype=np.float64)
        if (base.ndim != 2 or queries.ndim != 2 or base.shape[1] == 0
                or base.shape[1] != queries.shape[1] or base.dtype.kind not in "fiu"):
            raise ValueError("source base and query dimensions differ")
        if len(base) < COUNT or not len(queries) or not np.isfinite(queries).all():
            raise ValueError("source requires at least 100 base vectors and finite queries")
        metric = plain(source.attrs.get("distance"))
        if metric not in ("ip", "euclidean", "angular", "cosine", "normalized"):
            raise ValueError("unsupported source metric")
        if "dimension" in source.attrs and integer(source.attrs["dimension"], "dimension") != base.shape[1]:
            raise ValueError("source dimension attribute differs from its matrices")
        truth_data = source["neighbors"]
        if truth_data.ndim != 2 or truth_data.shape[0] != len(queries) or truth_data.shape[1] < COUNT:
            raise ValueError("source truth has the wrong shape")
        truth = validate_ids(truth_data[:, :COUNT], (len(queries), COUNT), len(base), "source truth")
        runs, configurations = _read_runs(results, len(queries), len(base), metric)
        for qi, query in enumerate(queries):
            # Read each needed base row once per query across all configurations/repetitions.
            union = np.unique(np.concatenate([truth[qi]] + [run["ids"][qi] for run in runs]))
            union = union[union >= 0]
            distances = metric_distances(metric, query, base[union])
            if not np.isfinite(distances).all():
                raise ValueError("original-metric distances overflowed")
            threshold = distances[np.searchsorted(union, truth[qi])].max() + EPSILON
            for run in runs:
                present = run["ids"][qi]
                present = present[present >= 0]
                run["strict_hits"] += int(np.isin(present, truth[qi]).sum())
                run["epsilon_hits"] += int((distances[np.searchsorted(union, present)] <= threshold).sum())
        attributes = {key: plain(value) for key, value in source.attrs.items()}
        scope = attributes.get("qag_scope", attributes.get("scope"))
        source_info = {
            "metric": metric, "base_rows": len(base), "queries": len(queries),
            "dimension": base.shape[1], "attributes": attributes,
            "scope": scope if scope is not None else "official_test",
            "official_test": scope is None or str(scope).lower() in ("official", "official_test"),
        }
    rows = []
    for (arm, ef, rerank), group in sorted(configurations.items()):
        repetitions = []
        for run in sorted(group, key=lambda item: item["repetition"]):
            repetitions.append({
                "repetition": run["repetition"], "raw_group": run["group"],
                "recall_ids": run["strict_hits"] / (len(queries) * COUNT),
                "recall_epsilon": run["epsilon_hits"] / (len(queries) * COUNT),
                "qps": len(queries) / run["seconds"], "total_seconds": run["seconds"],
                "ids_sha256": run["ids_sha256"], "resource_attrs": run["resource_attrs"],
            })
        qps = [run["qps"] for run in repetitions]
        rows.append({
            "arm": arm, "method": ARMS[arm], "ef": ef, "rerank": rerank,
            "recall_ids_min": min(run["recall_ids"] for run in repetitions),
            "recall_epsilon_min": min(run["recall_epsilon"] for run in repetitions),
            "qps_median": statistics.median(qps), "qps_min": min(qps), "qps_max": max(qps),
            "repetitions": repetitions,
        })
    targets = []
    for arm in sorted({row["arm"] for row in rows}):
        for target in (0.90, 0.95, 0.99):
            eligible = [row for row in rows if row["arm"] == arm and row["recall_ids_min"] >= target]
            selected = max(eligible, key=lambda row: (row["qps_median"], -row["ef"], -row["rerank"])) if eligible else None
            point = {"arm": arm, "method": ARMS[arm], "target": target, "reached": selected is not None}
            if selected is not None:
                point.update({key: value for key, value in selected.items() if key != "repetitions"})
            targets.append(point)
    provenance = {
        "raw_results": str(results.resolve()), "raw_results_sha256": digest(results),
        "source": str(data.resolve()), "source_sha256": digest(data),
        "extractor_sha256": digest(Path(__file__)),
        "harness_sha256": None, "execution_receipt": None,
    }
    if receipt is not None:
        receipt = Path(receipt)
        metadata = json.loads(receipt.read_text())
        if not isinstance(metadata, dict):
            raise ValueError("execution receipt must contain a JSON object")
        provenance["execution_receipt"] = {"path": str(receipt.resolve()), "sha256": digest(receipt), "metadata": metadata}
        provenance["harness_sha256"] = metadata.get("harness_sha256")
    return {
        "schema_version": 1, "source": source_info, "provenance": provenance,
        "protocol": {
            "count": COUNT, "repetitions": [0, 1, 2], "passes_per_repetition": 5,
            "timed_pass": 5, "qps": "query_count / sum(saved raw per-query fifth-pass times); best_qps ignored",
            "selection": "fastest measured median with minimum strict ID recall across repetitions >= target; no interpolation or rounding",
            "epsilon_recall": "float64 original-metric distance <= recomputed true kth distance + 0.001",
            "resources": "per-arm build seconds/encoded bytes where recorded; process RSS and VIBE USS delta aggregate all co-resident arms",
        },
        "rows": rows, "targets": targets,
    }


def write_report(results, data, out, receipt=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    report = extract(results, data, receipt)
    with out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, default=os.environ.get("HMS_QAG_RUN_RECEIPT"))
    args = parser.parse_args()
    write_report(args.results, args.data, args.out, args.receipt)
