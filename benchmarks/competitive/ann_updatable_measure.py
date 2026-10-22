# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""One Linux SIFT-1M comparison; all measured children hold the host timing lock."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def hnsw_ops(a):
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent))
    from hmsbench.ann_worker import Hnswlib
    meta = json.loads((a.data / "meta.json").read_text())
    n, dim = meta["n_train"], meta["dim"]
    count = n // 100
    rows = np.memmap(a.data / "train.f32", dtype="<f4", mode="r", shape=(n, dim))
    fresh = np.array(rows[2 * count:3 * count], copy=True)
    index = Hnswlib(dim, a.work, 1, "euclidean")
    index.open({}, n + count)
    report = {"count": count}
    for operation, ids in (("replace", np.arange(count)), ("delete", np.arange(count, 2 * count))):
        begin = time.perf_counter()
        if operation == "replace":
            index.replace(ids, fresh)
        else:
            index.delete(ids)
        apply = time.perf_counter() - begin
        begin = time.perf_counter()
        index.flush()
        flush = time.perf_counter() - begin
        report[operation] = {"apply_s": apply, "flush_s": flush, "per_op_ms_durable": (apply + flush) * 1000 / count}
    report["flush_semantics"] = "hnswlib adapter save_index; same save boundary as the competitive harness"
    report["smaps_rollup"] = Path("/proc/self/smaps_rollup").read_text()
    print("RESULT " + json.dumps(report))


def best(configs, target, old=False):
    if old:
        eligible = [{"recall": c["recall_at_10"], "qps": c["qps_single_thread"], "params": c["params"]}
                    for c in configs if c["recall_at_10"] >= target]
    else:
        import statistics
        eligible = [{"recall": c["recall_at_10"], "qps": statistics.median(r["qps"] for r in c["repeats"]), "params": c["params"]}
                    for c in configs if c["recall_at_10"] >= target]
    return max(eligible, key=lambda c: c["qps"]) if eligible else None


def main(a):
    a.work.mkdir(parents=True, exist_ok=False)
    commands = []
    env = {**os.environ, "RAYON_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}

    def run(label, arguments, cores="2,3", threads="1", output=None):
        command = ["flock", "-F", str(Path.home() / "hms-timing.lock"), "taskset", "-c", cores, *map(str, arguments)]
        commands.append({"label": label, "command": command, "environment": {"RAYON_NUM_THREADS": threads, "OMP_NUM_THREADS": threads}})
        with (a.work / f"{label}.stdout").open("w") as stdout, (a.work / f"{label}.stderr").open("w") as stderr:
            with subprocess.Popen(command, env={**env, "RAYON_NUM_THREADS": threads, "OMP_NUM_THREADS": threads}, stdout=stdout, stderr=stderr) as process:
                while True:
                    try:
                        commands[-1]["smaps_rollup_last"] = Path(f"/proc/{process.pid}/smaps_rollup").read_text()
                    except (FileNotFoundError, PermissionError, ProcessLookupError):
                        pass  # the process exited between polls; keep the last sample
                    try:
                        code = process.wait(timeout=1)
                        if code:
                            raise subprocess.CalledProcessError(code, command)
                        break
                    except subprocess.TimeoutExpired:
                        continue
        if output:
            return json.loads(Path(output).read_text())
        for line in reversed((a.work / f"{label}.stdout").read_text().splitlines()):
            if line.startswith("RESULT "):
                return json.loads(line[7:])
        raise RuntimeError(f"{label}: missing result")

    vertex = a.work / "index.vgraph"
    updated = a.work / "index.updates"
    shared = ["--data", a.data, "--metric", "euclidean", "--index", "vertex", "--degree", "64", "--build-ef", "500",
              "--quantized-search", "--residual", "--residual-bits", "8", "--vertex-bits", "8", "--id-bytes", "3", "--align-rows", "--reorder",
              "--ef", "10,16,24,32,48,64,96,128,192,256", "--rerank", "32,64", "--max-load", "1000000", "--max-wait-secs", "0"]
    build_path = a.work / "build.json"
    build = run("hms-build", [a.after, "ann-qgraph", *shared, "--save-index", vertex, "--save-updates", updated, "--out", build_path], "12-21", "10", build_path)
    before_path = a.work / "before.json"
    before = run("before", [a.before, "ann-qgraph", *shared, "--load-index", vertex, "--residency", "mapped", "--latency", "--repeats", "2", "--out", before_path], output=before_path)
    after_path = a.work / "after.json"
    after = run("after", [a.after, "ann-updates", "--action", "serve", "--index", updated, "--data", a.data, "--out", after_path], output=after_path)
    cold_path = a.work / "cold.json"
    cold = run("cold", [a.after, "ann-updates", "--action", "cold-start", "--cold", "--index", updated, "--data", a.data, "--ef", "64", "--rerank", "64", "--out", cold_path], output=cold_path)
    ops = {}
    for operation in ("replace", "delete"):
        copied = a.work / f"{operation}.updates"
        shutil.copytree(updated, copied)
        output = a.work / f"{operation}.json"
        ops[operation] = run(operation, [a.after, "ann-updates", "--action", operation, "--index", copied, "--data", a.data, "--out", output], output=output)
    worker = Path(__file__).parent / "hmsbench/ann_worker.py"
    hnsw_path = a.work / "hnsw"
    hnsw_build = run("hnsw-build", [a.python_hnsw, worker, "build", "--system", "hnswlib", "--build", "m32e500", "--threads", "10", "--data", a.data, "--workdir", hnsw_path], "12-21", "10")
    with (hnsw_path / "index.bin").open("rb") as handle:
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    hnsw = run("hnsw-serve", [a.python_hnsw, worker, "serve", "--system", "hnswlib", "--build", "m32e500", "--threads", "1", "--repeats", "2", "--data", a.data, "--workdir", hnsw_path])
    hnsw_ops_report = run("hnsw-ops", [a.python_hnsw, Path(__file__), "--hnsw-ops", "--data", a.data, "--work", hnsw_path])
    report = {"commit": a.commit, "before_commit": a.before_commit, "command": sys.argv, "commands": commands,
              "hardware": subprocess.check_output(["lscpu"], text=True), "dataset": json.loads((a.data / "meta.json").read_text()),
              "summary": {str(t): {"before": best(before["sweep"], t, True), "after": best(after["configs"], t), "hnswlib": best(hnsw["configs"], t)} for t in (0.90, 0.95, 0.99)},
              "hms": {"build": build, "before": before, "after": after, "cold": cold, "operations": ops},
              "hnswlib": {"build": hnsw_build, "serve": hnsw, "operations": hnsw_ops_report},
              "notes": ["One run per arm; timing noise is not estimated by independent repetitions.",
                        "Before is the immutable mapped base; after opens the updatable directory with an empty delta.",
                        "Recall is measured before updates; randomized and 50k tests cover updates and consolidation.",
                        "HMS operation batches contain 1% of ids and include commit; hnswlib uses its harness save_index flush."]}
    a.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--python-hnsw", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--before-commit", default="6685568")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--hnsw-ops", action="store_true")
    args = parser.parse_args()
    hnsw_ops(args) if args.hnsw_ops else main(args)
