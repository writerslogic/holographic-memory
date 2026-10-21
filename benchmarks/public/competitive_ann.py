# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = "==3.13.9"
# dependencies = ["numpy==2.2.6"]
# ///
"""Local, train-only ANN comparisons with isolated native-library workers."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import re
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/superiority_v3/ann_competition"
SEED = 24731
K = 10
VERSIONS = {"numpy": "2.2.6", "faiss-cpu": "1.15.1", "hnswlib": "0.8.0", "rabitqlib": "0.5.2"}
ENV = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
DATASETS = {"glove-100-angular": 100, "nytimes-256-angular": 256}
SCALES = {10000: 128, 100000: 512, 250000: 1024}
SYSTEMS = ("faiss_hnsw", "hnswlib", "rabitq_ivf", "symphonyqg", "faiss_ivfsq", "rabitq_ivf_current")
LOCK_PATH = Path("./.hms-target/timing.lock")
_CAMPAIGN = {"lock": None}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def write(path, value, exclusive=True):
    with Path(path).open("x" if exclusive else "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def normalize(rows):
    value = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(value, axis=1)
    return np.ascontiguousarray(value / np.where(norms > 0, norms, 1)[:, None], dtype="<f4")


def load_input(path):
    path = Path(path)
    info = json.loads((path / "inputs.json").read_text())
    if info["dim"] not in DATASETS.values() or info["database_rows"] not in SCALES:
        raise ValueError("input shape exceeds registered campaign")
    for name, expected in info["files"].items():
        if sha(path / name) != expected:
            raise ValueError(f"input changed: {name}")
    return info


def read_rows(path, dim):
    rows = np.fromfile(path, dtype="<f4")
    if rows.size % dim or rows.size > 250000 * 256 or not np.isfinite(rows).all():
        raise ValueError("invalid or oversized vector file")
    return rows.reshape(-1, dim)


def worker_command(system, *arguments):
    package = "faiss-cpu" if system in ("prepare", "faiss_hnsw", "faiss_ivfsq") else "hnswlib" if system == "hnswlib" else "rabitqlib"
    version = "0.7.0" if system == "rabitq_ivf_current" else VERSIONS[package]
    return ["uv", "run", "--no-project", "--python", "3.13.9", "--with", "numpy==2.2.6", "--with", f"{package}=={version}", "python", str(Path(__file__).resolve()), *map(str, arguments)]


@contextmanager
def timing_lock(path=LOCK_PATH):
    if _CAMPAIGN["lock"] is not None:
        yield _CAMPAIGN["lock"]
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _CAMPAIGN["lock"] = lock
        try:
            yield lock
        finally:
            _CAMPAIGN["lock"] = None
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def execute(command, receipt_path, timeout=1800):
    receipt_path = Path(receipt_path)
    with timing_lock() as lock:
        before = os.getloadavg()
        start = time.monotonic()
        actual_command = ["/usr/bin/time", "-l", *command] if sys.platform == "darwin" else command
        with receipt_path.with_suffix(".log").open("x") as log:
            try:
                result = subprocess.run(actual_command, env=os.environ | ENV, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
                exit_code = result.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
        after = os.getloadavg()
        rss_match = re.search(r"(\d+)\s+maximum resident set size", receipt_path.with_suffix(".log").read_text())
        peak_rss = int(rss_match[1]) if rss_match else None
        if peak_rss is not None and peak_rss > 8 * (1 << 30):
            exit_code = 70
        receipt = {"command": command, "exit_code": exit_code, "elapsed_seconds": time.monotonic() - start,
                   "actual_command": actual_command, "external_peak_rss_bytes": peak_rss,
                   "load_before": before, "load_after": after, "exclusive_timing_lock": str(LOCK_PATH),
                   "lock_device": os.fstat(lock.fileno()).st_dev, "lock_inode": os.fstat(lock.fileno()).st_ino,
                   "load_gate_met": before[0] < 3 and after[0] < 3,
                   "timing_scope": "descriptive when host load gate fails; single-thread batch-one calls include Python binding overhead for competitors",
                   "platform": platform.platform(), "machine": platform.machine(), "environment": ENV,
                   "harness_sha256": sha(Path(__file__))}
        write(receipt_path, receipt)
        if exit_code:
            raise subprocess.CalledProcessError(exit_code, command)
    return receipt


def prepare(args):
    destination = Path(args.out) / args.dataset / str(args.rows)
    destination.mkdir(parents=True, exist_ok=False)
    source = Path.home() / ".cache/hms-bench" / args.dataset / "train.f32"
    dim = DATASETS[args.dataset]
    data = np.memmap(source, dtype="<f4", mode="r").reshape(-1, dim)
    chosen = np.random.default_rng(SEED).choice(len(data), size=250200, replace=False)
    query_ids = chosen[-200:]
    if args.rows != 10000:
        eligible = np.ones(len(data), dtype=bool)
        eligible[chosen] = False
        for start in range(0, len(data), 8192):
            eligible[start:start + 8192] &= np.any(data[start:start + 8192] != 0, axis=1)
        fresh = np.random.default_rng(SEED + 1).choice(np.flatnonzero(eligible), size=240, replace=False)
        offset = 0 if args.rows == 100000 else 120
        query_ids = np.concatenate((query_ids[:80], fresh[offset:offset + 120]))
    if np.any(np.linalg.norm(data[query_ids].astype(np.float64), axis=1) == 0):
        raise ValueError("registered query split contains zero vector")
    train_ids = chosen[:args.rows]
    for name, ids in (("train", train_ids), ("calibration", query_ids[:80]), ("validation", query_ids[80:])):
        raw = np.asarray(data[ids], dtype="<f4")
        raw.tofile(destination / f"{name}_raw.f32")
        normalize(raw).tofile(destination / f"{name}.f32")
        np.save(destination / f"{name}_original_ids.npy", ids, allow_pickle=False)
    files = sorted(p.name for p in destination.iterdir())
    write(destination / "inputs.json", {"schema_version": 1, "dataset": args.dataset, "dim": dim,
          "database_rows": args.rows, "calibration_rows": 80, "validation_rows": 120, "seed": SEED,
          "source_train_path": str(source), "source_train_sha256": sha(source), "benchmark_test_opened": False,
          "normalization": "float64 L2 division, f32 storage; zero database vectors preserved",
          "files": {name: sha(destination / name) for name in files}})
    if args.inputs_only:
        print(destination, flush=True)
        return
    command = worker_command("prepare", "cluster", "--input", destination)
    execute(command, destination / "cluster-execution.json")
    print(destination, flush=True)


def cluster(args):
    import faiss

    faiss.omp_set_num_threads(1)
    info = load_input(args.input)
    path = Path(args.input)
    train = read_rows(path / "train.f32", info["dim"])
    nlist = SCALES[len(train)]
    start = time.perf_counter_ns()
    kmeans = faiss.Kmeans(info["dim"], nlist, niter=20, seed=SEED, spherical=True,
                          min_points_per_centroid=1, max_points_per_centroid=256)
    kmeans.train(train)
    centroids = np.asarray(kmeans.centroids, dtype="<f4")
    quantizer = faiss.IndexFlatIP(info["dim"])
    quantizer.add(centroids)
    centroids.tofile(path / "centroids.f32")
    quantizer.search(train, 1)[1].astype("<u4").tofile(path / "assignments.u32")
    training_ns = time.perf_counter_ns() - start
    exact = faiss.IndexFlatIP(info["dim"])
    exact.add(train)
    # Validation queries are used only to construct an oracle, never for tuning.
    for phase in ("calibration", "validation"):
        queries = read_rows(path / f"{phase}.f32", info["dim"])
        distances, ids = exact.search(queries, K + 1)
        np.savez(path / f"{phase}_truth.npz", ids=ids, similarities=distances)
    names = ("centroids.f32", "assignments.u32", "calibration_truth.npz", "validation_truth.npz")
    write(path / "clusters.json", {"schema_version": 1, "nlist": nlist, "niter": 20, "seed": SEED,
          "spherical": True, "max_points_per_centroid": 256, "training_ns": training_ns,
          "faiss_version": importlib.metadata.version("faiss-cpu"), "faiss_compile_options": faiss.get_compile_options(),
          "files": {name: sha(path / name) for name in names}})


def builds(system):
    if system in ("faiss_hnsw", "hnswlib"):
        return [{"M": m, "ef_construction": 200, "seed": SEED} for m in (32, 16)]
    if system == "rabitq_ivf":
        return [{"bits": bits} for bits in (4, 7, 32, 8, 9, 5, 6)]
    if system == "rabitq_ivf_current":
        return [{"bits": bits} for bits in (5, 6, 7, 8, 9, 32)]
    if system == "faiss_ivfsq":
        return [{"bits": bits, "by_residual": True} for bits in (4, 8)]
    return [{"M": 32, "ef_construction": 200, "bits": bits, "init": "pipnn"} for bits in (4, 8)]


def searches(system, nlist):
    if system in ("faiss_hnsw", "hnswlib", "symphonyqg"):
        return [{"ef": ef} for ef in (32, 64, 128, 256, 512, 1024)]
    return [{"nprobe": nprobe} for nprobe in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) if nprobe <= nlist]


def worker(args):
    info = load_input(args.input)
    path = Path(args.input)
    train = read_rows(path / "train.f32", info["dim"])
    queries = read_rows(path / f"{args.phase}.f32", info["dim"])
    params = json.loads(args.build)
    search_params = json.loads(args.search)
    package = "faiss-cpu" if args.system in ("faiss_hnsw", "faiss_ivfsq") else "hnswlib" if args.system == "hnswlib" else "rabitqlib"
    expected_version = "0.7.0" if args.system == "rabitq_ivf_current" else VERSIONS[package]
    if importlib.metadata.version(package) != expected_version:
        raise ValueError("competitor version differs")
    index_path = Path(args.index)
    if args.system in ("faiss_hnsw", "faiss_ivfsq"):
        import faiss

        faiss.omp_set_num_threads(1)
        start = time.perf_counter_ns()
        if args.phase == "calibration" and args.system == "faiss_hnsw":
            index = faiss.IndexHNSWFlat(info["dim"], params["M"], faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = params["ef_construction"]
            index.hnsw.rng = faiss.RandomGenerator(params["seed"])
            index.add(train)
        elif args.phase == "calibration":
            centroids = read_rows(path / "centroids.f32", info["dim"])
            quantizer = faiss.IndexFlatIP(info["dim"])
            quantizer.add(centroids)
            qtype = faiss.ScalarQuantizer.QT_4bit if params["bits"] == 4 else faiss.ScalarQuantizer.QT_8bit
            index = faiss.IndexIVFScalarQuantizer(quantizer, info["dim"], len(centroids), qtype, faiss.METRIC_INNER_PRODUCT, params["by_residual"])
            index.cp.seed = SEED
            index.train(train)
            assigned = np.fromfile(path / "assignments.u32", dtype="<u4").astype(np.int64)
            index.add_core(len(train), faiss.swig_ptr(train), None, faiss.swig_ptr(assigned))
        else:
            index = faiss.read_index(str(index_path))
        save = lambda: faiss.write_index(index, str(index_path))
        search = lambda q, p: index.search(q, K)[1]
        if args.system == "faiss_hnsw":
            configure = lambda p: setattr(index.hnsw, "efSearch", p["ef"])
        else:
            configure = lambda p: setattr(index, "nprobe", p["nprobe"])
    elif args.system == "hnswlib":
        import hnswlib

        start = time.perf_counter_ns()
        index = hnswlib.Index(space="ip", dim=info["dim"])
        if args.phase == "calibration":
            index.init_index(max_elements=len(train), M=params["M"], ef_construction=params["ef_construction"], random_seed=params["seed"])
            index.add_items(train, np.arange(len(train)), num_threads=1)
        else:
            index.load_index(str(index_path))
        index.set_num_threads(1)
        save = lambda: index.save_index(str(index_path))
        search = lambda q, p: index.knn_query(q, k=K, num_threads=1)[0]
        configure = lambda p: index.set_ef(p["ef"])
    else:
        import rabitqlib

        start = time.perf_counter_ns()
        cls = rabitqlib.IvfIndex if args.system in ("rabitq_ivf", "rabitq_ivf_current") else rabitqlib.SymqgIndex
        if args.phase == "validation":
            index = cls.load(str(index_path))
        elif args.system in ("rabitq_ivf", "rabitq_ivf_current"):
            centroids = read_rows(path / "centroids.f32", info["dim"])
            assignments = np.fromfile(path / "assignments.u32", dtype="<u4")
            index = cls(dim=info["dim"], max_elements=len(train), num_clusters=len(centroids), nbits=params["bits"], metric="ip")
            index.build(train, centroids, assignments, num_threads=1)
        else:
            index = cls(dim=info["dim"], max_degree=params["M"], metric="ip", quantization_bits=params["bits"])
            index.build(train, ef_construction=params["ef_construction"], num_threads=1, init=params["init"])
        save = lambda: index.save(str(index_path))
        configure = lambda p: None
        if args.system in ("rabitq_ivf", "rabitq_ivf_current"):
            search = lambda q, p: index.search(q, k=K, nprobe=p["nprobe"], num_threads=1)[0]
        else:
            search = lambda q, p: index.search(q, k=K, ef=p["ef"], num_threads=1)[0]
    construction_ns = time.perf_counter_ns() - start
    construction_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    construction_peak_rss_bytes = construction_rss if sys.platform == "darwin" else construction_rss * 1024
    if args.phase == "calibration":
        save()
    index_bytes = index_path.stat().st_size
    index_sha = sha(index_path)
    configs = []
    for p in search_params:
        configure(p)
        for q in queries:
            search(q[None], p)
        timings = []
        batch_timings = []
        predictions = None
        before = os.getloadavg()[0]
        for repeat in range(args.repeats):
            order = np.random.default_rng(SEED + repeat).permutation(len(queries))
            ordered = [None] * len(queries)
            clocked = [None] * len(queries)
            for query_id in order:
                query = queries[query_id:query_id + 1]
                tick = time.perf_counter_ns()
                found = search(query, p)
                duration = time.perf_counter_ns() - tick
                ordered[query_id] = np.asarray(found[0], dtype=np.int64).tolist()
                clocked[query_id] = duration
            if predictions is not None and predictions != ordered:
                raise ValueError("predictions vary across repetitions")
            predictions = ordered
            timings.append(clocked)
            tick = time.perf_counter_ns()
            batch_labels = search(queries, p)
            batch_timings.append(time.perf_counter_ns() - tick)
            if np.asarray(batch_labels, dtype=np.int64).tolist() != predictions:
                raise ValueError("native batch predictions differ from batch1 calls")
        after = os.getloadavg()[0]
        configs.append({**p, "predictions": predictions, "elapsed_ns": timings, "batch_elapsed_ns": batch_timings,
                        "batch_boundary": "one Python call enclosing native search over entire query matrix, num_threads1; FAISS may use batched centroid routing",
                        "load_before": before, "load_after": after, "load_gate_met": before < 3 and after < 3})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    if peak_rss_bytes > 8 * (1 << 30):
        raise ValueError("worker exceeded 8GiB RSS campaign limit")
    write(args.out, {"schema_version": 1, "system": args.system, "phase": args.phase, "build": params,
          "dim": info["dim"], "database_rows": len(train), "query_rows": len(queries),
          "build_ns": construction_ns if args.phase == "calibration" else None,
          "load_ns": construction_ns if args.phase == "validation" else None,
          "index_bytes": index_bytes, "index_sha256": index_sha, "index_path": str(index_path),
          "peak_rss_bytes": peak_rss_bytes, "construction_peak_rss_bytes": construction_peak_rss_bytes, "package_version": expected_version,
          "configs": configs, "measurement": "batch1, one thread, repeated individual Python binding calls"})


def run_baselines(args):
    with timing_lock():
        run_baseline_arms(args)


def run_baseline_arms(args):
    info = load_input(args.input)
    path = Path(args.input)
    deadline = time.monotonic() + args.timeout
    for system in args.systems.split(","):
        if system not in SYSTEMS:
            raise ValueError("unknown baseline")
        for build_id, params in enumerate(builds(system)):
            if info["database_rows"] == 10000 and (system == "rabitq_ivf_current" or (system == "rabitq_ivf" and params["bits"] in (5, 6))):
                continue
            name = f"{system}-{build_id}"
            command = worker_command(system, "worker", "--input", path, "--system", system,
                        "--phase", "calibration", "--build", json.dumps(params),
                        "--search", json.dumps(searches(system, SCALES[info["database_rows"]])),
                        "--index", path / f"{name}.index", "--repeats", args.repeats,
                        "--out", path / f"{name}-calibration.json")
            if (path / f"{name}-calibration-execution.json").exists():
                previous = json.loads((path / f"{name}-calibration-execution.json").read_text())
                previous_result = json.loads((path / f"{name}-calibration.json").read_text())
                if previous["exit_code"] != 0 or previous_result["build"] != params or sha(path / f"{name}.index") != previous_result["index_sha256"]:
                    raise ValueError("existing campaign arm is incomplete or changed")
                continue
            print(f"{path.name}: {name}", flush=True)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("bounded baseline campaign budget expired")
            execute(command, path / f"{name}-calibration-execution.json", timeout=min(1800, remaining))


def run_hms(args):
    with timing_lock():
        run_hms_arms(args)


def run_hms_arms(args):
    info = load_input(args.input)
    path = Path(args.input)
    binary = Path(args.binary).resolve(strict=True)
    binary_sha = sha(binary)
    deadline = time.monotonic() + args.timeout
    requested = [tuple(map(int, value.split("/"))) for value in args.builds.split(",")]
    if any(build not in ((4, 4), (8, 0), (8, 4), (4, 5)) for build in requested) or len(set(requested)) != len(requested):
        raise ValueError("unregistered candidate build")
    for bits, residual_bits in requested:
        prefix = f"hms_ivf-{args.label}" if args.label else "hms_ivf"
        name = f"{prefix}-{bits}-{residual_bits}"
        output = path / f"{name}-calibration.json"
        index = path / f"{name}.index"
        probes = [entry["nprobe"] for entry in searches("rabitq_ivf", SCALES[info["database_rows"]])]
        command = [str(binary), "--train", str(path / "train.f32"), "--queries", str(path / "calibration.f32"),
                   "--centroids", str(path / "centroids.f32"), "--assignments", str(path / "assignments.u32"),
                   "--dim", str(info["dim"]), "--bits", str(bits), "--residual-bits", str(residual_bits),
                   "--nprobe", ",".join(map(str, probes)), "--rerank", "0,64,256,1024", "--repeats", str(args.repeats),
                   "--out", str(output), "--index-out", str(index)]
        if output.exists() or index.exists():
            prior = json.loads(output.read_text())
            if prior["binary_sha256"] != binary_sha or prior["build"] != {"bits": bits, "residual_bits": residual_bits} or prior["index_sha256"] != sha(index):
                raise ValueError("existing candidate arm changed")
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("bounded HMS campaign budget expired")
        receipt = execute(command, path / f"{name}-calibration-execution.json", timeout=min(1800, remaining))
        if sha(binary) != binary_sha:
            raise ValueError("candidate binary changed during campaign")
        result = json.loads(output.read_text())
        result.update({"system": "hms_ivf", "phase": "calibration", "build": {"bits": bits, "residual_bits": residual_bits},
                       "index_path": str(index), "index_sha256": sha(index), "binary_sha256": binary_sha,
                       "peak_rss_bytes": receipt["external_peak_rss_bytes"],
                       "measurement": "batch1, one thread, native Rust query loop; mismatched Python competitor overhead precludes timing superiority"})
        write(output, result, exclusive=False)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--dataset", choices=DATASETS, required=True)
    prepare_parser.add_argument("--rows", type=int, choices=SCALES, required=True)
    prepare_parser.add_argument("--out", default=OUT)
    prepare_parser.add_argument("--inputs-only", action="store_true")
    cluster_parser = sub.add_parser("cluster")
    cluster_parser.add_argument("--input", required=True)
    run_parser = sub.add_parser("baselines")
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--systems", default=",".join(SYSTEMS))
    run_parser.add_argument("--repeats", type=int, choices=(1, 3, 5), default=3)
    run_parser.add_argument("--timeout", type=int, default=1800)
    hms_parser = sub.add_parser("hms")
    hms_parser.add_argument("--input", required=True)
    hms_parser.add_argument("--binary", required=True)
    hms_parser.add_argument("--repeats", type=int, choices=(1, 3, 5), default=3)
    hms_parser.add_argument("--timeout", type=int, default=1800)
    hms_parser.add_argument("--builds", default="4/4,8/0,8/4")
    hms_parser.add_argument("--label", choices=("", "v2"), default="")
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--input", required=True)
    worker_parser.add_argument("--system", choices=SYSTEMS, required=True)
    worker_parser.add_argument("--phase", choices=("calibration", "validation"), required=True)
    worker_parser.add_argument("--build", required=True)
    worker_parser.add_argument("--search", required=True)
    worker_parser.add_argument("--index", required=True)
    worker_parser.add_argument("--repeats", type=int, choices=(1, 3, 5), required=True)
    worker_parser.add_argument("--out", required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    {"prepare": prepare, "cluster": cluster, "worker": worker, "baselines": run_baselines, "hms": run_hms}[args.command](args)
