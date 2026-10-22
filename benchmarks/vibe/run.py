# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Measure an actual VIBE adapter on a prepared dataset, one CPU and one query at a time.

This is a diagnostic with one fixed build per method, not VIBE's full parameter search.
Both VIBE's distance-tolerant recall and strict neighbor-ID recall are retained.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import h5py
import numpy as np


EF = (100, 128, 160, 200, 256, 320, 400, 512, 640, 800, 1000)
TARGETS = (0.73, 0.77, 0.90, 0.94, 0.95, 0.99, 1.0)
METHODS = ("hms", "pag", "glass", "glass-sq8", "symphonyqg", "ngt-qg", "rabitq-hnsw", "lorann", "hnswlib", "pynndescent")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def specification(method, metric):
    configurations = [[ef] for ef in EF]
    if method == "hms":
        return "hms", "HMS", [metric, 64, 500], [[ef, rerank] for ef in EF for rerank in (100, 200, 400)]
    if method == "hnswlib":
        return "hnswlib", "HnswLib", [metric, 32, 500], configurations
    if method == "symphonyqg":
        return "rabitq", "SymphonyQG", [metric, 64, 400, 0], configurations
    if method == "rabitq-hnsw":
        return "rabitq", "RabitqHNSW", [metric, 32, 32, 400, 5, False], configurations
    if method == "pynndescent":
        return "pynndescent", "PyNNDescent", [metric, 80, 1.0, 1.0], [[v] for v in (0.0, 0.04, 0.1, 0.2, 0.36)]
    if method == "pag":
        return "pag", "PAG", [metric, 500, 64, 32, 100], configurations
    if method in ("glass", "glass-sq8"):
        quantizer = "SQ8U" if method == "glass-sq8" else "SQ4U"
        return "glass", "Glass", [metric, 400, 64, "SQ8U", quantizer, "FP32"], configurations
    if method == "lorann":
        queries = [[[nprobe, rerank]] for nprobe in (8, 32, 128, 256) for rerank in (400, 1600, 4800)]
        return "lorann", "Lorann", [metric, 8, 1024, 128, 32, 5], queries
    raise NotImplementedError(f"{method}: use its official VIBE container and configuration grid")


def score(ids, train, queries, truth, distances, metric, epsilon=1e-3):
    ids = np.asarray(ids)
    nq, k = truth.shape
    if ids.shape != (nq, k) or ids.dtype.kind not in "iu":
        raise ValueError("invalid result shape or type")
    if np.any(ids < -1) or np.any(ids >= len(train)):
        raise ValueError("result ID outside corpus")
    valid = ids >= 0
    if any(len(set(row[row >= 0].tolist())) != int(present.sum()) for row, present in zip(ids, valid)):
        raise ValueError("duplicate result IDs")
    strict = [len(set(row.tolist()) & set(expected.tolist())) / k for row, expected in zip(ids, truth)]
    vectors = train[np.maximum(ids, 0)].astype(np.float64)
    q = queries.astype(np.float64)
    if metric == "euclidean":
        measured = np.linalg.norm(vectors - q[:, None, :], axis=2)
    elif metric == "cosine":
        measured = 1 - np.einsum("qkd,qd->qk", vectors, q) / (
            np.linalg.norm(vectors, axis=2) * np.linalg.norm(q, axis=1)[:, None])
    else:
        measured = 1 - np.einsum("qkd,qd->qk", vectors, q)
    recalls = np.sum(valid & (measured <= distances[:, -1, None] + epsilon), axis=1) / k
    return {"recall": float(recalls.mean()), "strict_id_recall": float(np.mean(strict)),
            "query_recalls": recalls.tolist(), "strict_query_recalls": strict,
            "missing_results": int((~valid).sum())}


def select(rows, target):
    eligible = [row for row in rows if row["recall"] >= target]
    return max(eligible, key=lambda row: row["median_qps"]) if eligible else None


def run(args):
    if not 1 <= args.k <= 100 or not 1 <= args.repeats <= 20:
        raise ValueError("k must be in 1..100 and repeats in 1..20")
    if args.out.exists() or args.out.with_suffix(".npz").exists():
        raise FileExistsError(args.out)
    if hasattr(os, "sched_setaffinity"):
        available = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(available)})
    else:
        raise RuntimeError("use a Linux host so the process can be pinned to one CPU")
    import faiss
    import psutil

    faiss.omp_set_num_threads(1)
    sys.path.insert(0, str(args.vibe.resolve()))
    manifest = json.loads(args.data.with_suffix(".json").read_text())
    if sha(args.data) != manifest["sha256"]:
        raise ValueError("input hash mismatch")
    with h5py.File(args.data, "r") as handle:
        train, queries = handle["train"][:], handle["test"][:]
        truth = handle["neighbors"][:, :args.k]
        distances = handle["distances"][:, :args.k]
        metric = str(handle.attrs["distance"])
        learn = handle["learn"][:] if "learn" in handle else None
        learn_truth = handle["learn_neighbors"][:] if learn is not None else None
    if not np.isfinite(train).all() or not np.isfinite(queries).all():
        raise ValueError("nonfinite input")
    module, constructor, build, configurations = specification(args.method, metric)
    source = args.vibe / "vibe" / "algorithms" / module / "module.py"
    result = {
        "schema_version": 1, "method": args.method, "input": manifest, "k": args.k,
        "scope": "exploratory corpus prefix; one fixed build per method; no official leaderboard claim",
        "protocol": {"build": build, "query_grid": configurations, "repeats": args.repeats,
                     "recall_epsilon": 1e-3, "selection": "fastest measured median QPS reaching target; diagnostic only"},
        "environment": {"host": platform.node(), "platform": platform.platform(),
                        "cpu_affinity": sorted(os.sched_getaffinity(0)), "load": os.getloadavg(),
                        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}},
        "adapter_sha256": sha(source), "harness_sha256": sha(Path(__file__)),
        "vibe_revision": subprocess.check_output(["git", "-C", str(args.vibe), "rev-parse", "HEAD"], text=True).strip(),
        "rows": [],
    }
    cls = getattr(importlib.import_module(f"vibe.algorithms.{module}.module"), constructor)
    adapter = cls(*build)
    rss_before = psutil.Process().memory_full_info().uss
    original_hash = hashlib.sha256(train.tobytes()).hexdigest()
    started = time.perf_counter()
    fitting = train.copy()
    if args.method == "lorann" and learn is not None:
        adapter.fit_ood(fitting, learn, learn_truth)
    else:
        adapter.fit(fitting)
    result["build_seconds"] = time.perf_counter() - started
    del fitting
    result["index_uss_delta_bytes"] = psutil.Process().memory_full_info().uss - rss_before
    if hashlib.sha256(train.tobytes()).hexdigest() != original_hash:
        raise ValueError("adapter changed shared corpus")
    print(f"built {args.method} in {result['build_seconds']:.1f}s", flush=True)
    predictions = {}
    for configuration_index, configuration in enumerate(configurations):
        adapter.set_query_arguments(*configuration)
        adapter.query(queries[0], args.k)
        timings, predicted = [], None
        loads = []
        for _ in range(args.repeats):
            loads.append(os.getloadavg()[0])
            elapsed, answers = [], []
            for query in queries:
                t0 = time.perf_counter_ns()
                answer = adapter.query(query, args.k)
                elapsed.append(time.perf_counter_ns() - t0)
                found = np.asarray(answer)
                if found.ndim != 1 or found.dtype.kind not in "iu" or len(found) > args.k:
                    raise ValueError("adapter returned invalid IDs")
                answers.append(np.pad(found.astype(np.int64), (0, args.k - len(found)), constant_values=-1))
            current = np.asarray(answers)
            if predicted is not None and not np.array_equal(current, predicted):
                raise ValueError("search results changed across timing repeats")
            predicted = current
            timings.append(elapsed)
            loads.append(os.getloadavg()[0])
        values = np.asarray(timings)
        qps = (len(queries) * 1e9 / values.sum(axis=1)).tolist()
        row = score(predicted, train, queries, truth, distances, metric)
        row.update(params=configuration, qps_runs=qps, median_qps=float(np.median(qps)),
                   best_qps=max(qps), p50_us=float(np.percentile(values, 50) / 1000),
                   p99_us=float(np.percentile(values, 99) / 1000), load_1m=loads,
                   load_gate_met=all(load < 3 for load in loads), name=str(adapter))
        result["rows"].append(row)
        predictions[f"ids_{configuration_index}"] = predicted.astype(np.int32)
        predictions[f"ns_{configuration_index}"] = values
        print(f"{configuration}: recall={row['recall']:.5f}, strict={row['strict_id_recall']:.5f}, QPS={row['median_qps']:.1f}", flush=True)
    result["targets"] = {
        str(target): ({"params": row["params"], "recall": row["recall"], "strict_id_recall": row["strict_id_recall"],
                       "median_qps": row["median_qps"]} if (row := select(result["rows"], target)) else None)
        for target in TARGETS
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.with_suffix(".npz"), **predictions)
    result["predictions_sha256"] = sha(args.out.with_suffix(".npz"))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vibe", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    run(parser.parse_args())
