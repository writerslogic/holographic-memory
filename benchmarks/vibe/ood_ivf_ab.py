# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Learn-only development grid, followed by one sealed OOD comparison; never standings."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import signal
import statistics
import time
from pathlib import Path

import h5py
import numpy as np

LEARN_QUERIES = 32768
VALIDATION = 1000
SEED = 24301
K = 100
GRAPH_EF = {
    "ip": [64, 80, 100, 128, 160, 200, 256, 320, 400, 512, 640, 800, 1000, 1500, 2000],
    "angular": [40, 50, 64, 80, 100, 128, 160, 200, 256, 320, 400, 512, 640, 800, 1000],
}
GRAPH_RERANK = {"ip": [200, 400, 1000], "angular": [100, 200, 400]}
LOWRANK_NPROBE = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
LOWRANK_RERANK = [100, 200, 400, 800]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def exact_top(base, queries, k):
    if not 0 < k <= len(base):
        raise ValueError("invalid k")
    base64 = base.astype(np.float64)
    out = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), 32):
        scores = queries[start:start + 32].astype(np.float64) @ base64.T
        for j, row in enumerate(scores):
            threshold = np.partition(row, -k)[-k]
            candidates = np.flatnonzero(row >= threshold)
            out[start + j] = candidates[np.lexsort((candidates, -row[candidates]))[:k]]
    return out


def recall(found, truth):
    if len(found) != len(truth) or not len(truth):
        raise ValueError("missing query results")
    if any(len(f) != K or len(set(f)) != K for f in found):
        raise ValueError("result count or uniqueness violation")
    return float(np.mean([len(set(f) & set(t)) / K for f, t in zip(found, truth)]))


def normalized(rows):
    norm = np.linalg.norm(rows.astype(np.float64), axis=1, keepdims=True)
    return np.asarray(np.divide(rows, norm, where=norm != 0, out=np.zeros_like(rows, dtype=np.float64)), dtype=np.float32)


def choose(rows, arm):
    eligible = [(i, r) for i, r in enumerate(rows) if r["arm"] == arm and r["val_recall"] >= .95]
    if not eligible:
        return None
    return min(eligible, key=lambda pair: (-statistics.median(pair[1]["qps"]), pair[1]["build"], pair[1]["args"]))[0]


@contextlib.contextmanager
def timing_lock(path, max_load, wait_seconds):
    deadline = time.monotonic() + wait_seconds
    with open(path, "a") as lock:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("timing lock/load gate was not available")
            if os.getloadavg()[0] < max_load:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    # Another timing window owns the lock; retry after the shared delay.
                    pass
            time.sleep(5)
        def expired(_signum, _frame):
            raise TimeoutError("bounded timing window exceeded 720 seconds")
        previous = signal.signal(signal.SIGALRM, expired)
        signal.alarm(720)
        try:
            yield
            if os.getloadavg()[0] >= max_load:
                raise RuntimeError("load gate exceeded during timing; measurements rejected")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
            fcntl.flock(lock, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--slice", type=int, default=0)
    parser.add_argument("--degree", type=int, default=64)
    parser.add_argument("--build-threads", type=int, default=2)
    parser.add_argument("--lock", default=os.path.expanduser("~/hms-timing.lock"))
    parser.add_argument("--max-load", type=float, default=3.0)
    parser.add_argument("--wait-seconds", type=float, default=600)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    if args.out.exists() or args.slice < 0 or args.build_threads < 1:
        raise ValueError("output exists or invalid dimensions")
    import holographic_memory as hm
    with h5py.File(args.hdf5, "r") as handle:
        metric = str(handle.attrs["distance"])
        if metric not in ("ip", "angular", "normalized"):
            raise ValueError("unsupported metric")
        base = np.asarray(handle["train"][:args.slice or None], dtype=np.float32)
        learn = np.asarray(handle["learn"], dtype=np.float32)
    if len(learn) < LEARN_QUERIES + VALIDATION or len(base) < K:
        raise ValueError("insufficient rows for preregistered split")
    kind = "ip" if metric == "ip" else "angular"
    if kind == "angular":
        base, learn = normalized(base), normalized(learn)
    if not np.isfinite(base).all() or not np.isfinite(learn).all():
        raise ValueError("nonfinite input")
    order = np.random.default_rng(SEED).permutation(len(learn))
    train_q = np.ascontiguousarray(learn[order[:LEARN_QUERIES]])
    val_q = np.ascontiguousarray(learn[order[LEARN_QUERIES:LEARN_QUERIES + VALIDATION]])
    val_truth = exact_top(base, val_q, K)
    module_file = Path(hm.__file__).parent / "holographic_memory.abi3.so"
    result = {"schema_version": 1, "measurement": "development", "dataset": args.hdf5.name,
              "dataset_sha256": digest(args.hdf5), "source": args.source,
              "script_sha256": digest(__file__), "module_sha256": digest(module_file),
              "machine": platform.machine(), "cpu": platform.processor(), "affinity": sorted(os.sched_getaffinity(0)),
              "metric": kind, "base_rows": len(base), "dimension": base.shape[1], "k": K,
              "training_rows": len(train_q), "validation_rows": len(val_q), "seed": SEED,
              "training_ids_sha256": hashlib.sha256(order[:LEARN_QUERIES].astype('<u4').tobytes()).hexdigest(),
              "validation_ids_sha256": hashlib.sha256(order[LEARN_QUERIES:LEARN_QUERIES + VALIDATION].astype('<u4').tobytes()).hexdigest(),
              "build_threads": args.build_threads, "query_threads": 1, "rounds": 3, "official_test_read": False,
              "builds_s": {}, "rows": [], "loads": [], "stage": "build"}
    def save():
        temp = args.out.with_suffix(args.out.suffix + ".partial")
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(args.out)
    save()
    configs = []
    def add(arm, label, ix, query_args):
        for query in query_args:
            configs.append((arm, label, ix, query))
            result["rows"].append({"arm": arm, "build": label, "args": list(query), "qps": []})
    def build(label, make):
        start = time.monotonic()
        ix = make()
        ix.fit(base, training_queries=train_q)
        result["builds_s"][label] = time.monotonic() - start
        save()
        print("built", label, result["builds_s"][label], flush=True)
        return ix
    graph = build("graph-R64", lambda: hm.VectorIndex(kind, degree=args.degree, build_ef=500,
                   threads=args.build_threads, code_bits=8, direct_ip=kind == "ip"))
    add("graph", "graph-R64", graph, [(ef, rr) for rr in GRAPH_RERANK[kind] for ef in GRAPH_EF[kind]])
    for clusters in (256, 1024):
        for rank in (16, 32, 64):
            label = f"lowrank-C{clusters}-r{rank}"
            ix = build(label, lambda clusters=clusters, rank=rank: hm.LowRankIvf(kind, clusters, rank))
            add("lowrank", label, ix, [(p, rr) for p in LOWRANK_NPROBE for rr in LOWRANK_RERANK])
    def run(config, queries):
        _, _, ix, query_args = config
        ix.set_query_arguments(*query_args)
        return [ix.query(q, K) for q in queries]
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    result["query_affinity"] = sorted(os.sched_getaffinity(0))
    result["stage"] = "validation"
    for row, config in zip(result["rows"], configs):
        row["val_recall"] = recall(run(config, val_q), val_truth)
    save()
    print("validation complete", flush=True)
    labels = list(result["builds_s"])
    for rnd in range(3):
        for label in labels[rnd:] + labels[:rnd]:
            with timing_lock(args.lock, args.max_load, args.wait_seconds):
                before = os.getloadavg()[0]
                for row, config in zip(result["rows"], configs):
                    if config[1] == label:
                        run(config, val_q[:5])
                        start = time.perf_counter()
                        run(config, val_q)
                        row["qps"].append(len(val_q) / (time.perf_counter() - start))
                result["loads"].append({"round": rnd, "build": label, "before": before, "after": os.getloadavg()[0]})
            save()
        print("tuning round", rnd, flush=True)
    result["selected"] = {arm: choose(result["rows"], arm) for arm in ("graph", "lowrank")}
    result["stage"] = "selection_sealed"
    save()
    if any(i is None for i in result["selected"].values()):
        result.update(stage="complete", verdict="FAIL: no eligible development configuration", official_test_read=False)
        save()
        return
    with h5py.File(args.hdf5, "r") as handle:
        test = np.asarray(handle["test"], dtype=np.float32)
    if kind == "angular":
        test = normalized(test)
    truth = exact_top(base, test, K)
    result["official_test_read"] = True
    result["confirmation"] = {}
    for arm, i in result["selected"].items():
        found = run(configs[i], test)
        result["confirmation"][arm] = {"recall": recall(found, truth), "qps": [],
               "answers_sha256": hashlib.sha256(np.asarray(found, dtype='<i8').tobytes()).hexdigest()}
    for rnd in range(3):
        with timing_lock(args.lock, args.max_load, args.wait_seconds):
            before = os.getloadavg()[0]
            for arm in (("graph", "lowrank") if rnd % 2 == 0 else ("lowrank", "graph")):
                config = configs[result["selected"][arm]]
                run(config, test[:5])
                start = time.perf_counter()
                run(config, test)
                result["confirmation"][arm]["qps"].append(len(test) / (time.perf_counter() - start))
            result["loads"].append({"confirmation_round": rnd, "before": before, "after": os.getloadavg()[0]})
        save()
    g, lr = (result["confirmation"][arm] for arm in ("graph", "lowrank"))
    result["paired_speedups"] = [b / a for a, b in zip(g["qps"], lr["qps"])]
    result["verdict"] = "PASS" if min(g["recall"], lr["recall"]) >= .95 and min(result["paired_speedups"]) >= 1 else "FAIL"
    result["stage"] = "complete"
    save()
    print(result["verdict"], result["paired_speedups"], flush=True)


if __name__ == "__main__":
    main()
