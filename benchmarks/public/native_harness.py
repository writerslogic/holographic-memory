# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy==2.2.6", "h5py==3.14.0", "faiss-cpu==1.15.1", "hnswlib==0.8.0"]
# ///
"""Frozen train calibration and a common native ANN query boundary.

No timing command is implicit: preregister and tune never read test datasets.
Build and round are separate, explicitly invoked stages run by the coordinator.
"""

import argparse
import fcntl
import gc
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

THREAD_ENV = {
    name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS",
    )
}
os.environ.update(THREAD_ENV)

import faiss
import h5py
import hnswlib
import numpy as np

faiss.omp_set_num_threads(1)
K = 10
EPSILON = 1e-3
SEED = 20261009
VERSIONS = {"numpy": "2.2.6", "h5py": "3.14.0", "faiss-cpu": "1.15.1", "hnswlib": "0.8.0"}
COMMITS = {
    "faiss": "7ea7339886edf8b4d9e719f593a5dfccff158274",
    "hnswlib": "3f3429661187e4c24a490a0f148fc6bc89042b3d",
    "ann_benchmarks": "2e081ad32c1eccab72dcb739ad886c310b90f715",
}
SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parents[1]
SYSTEMS = ("hms", "faiss_hnsw", "faiss_ivfpq", "hnswlib")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def artifact(path, **extra):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size, **extra}


def verify_artifact(item):
    path = Path(item["path"])
    if path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
        raise ValueError(f"artifact fingerprint differs: {path}")
    return path


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(f"frozen artifact already exists: {path}")
    temp = path.with_name(path.name + f".{os.getpid()}.new")
    with temp.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if exclusive:
        try:
            os.link(temp, path)
        finally:
            temp.unlink()
    else:
        os.replace(temp, path)


def save_array(array, directory):
    array = np.ascontiguousarray(array)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = memoryview(array).cast("B")
    sha = hashlib.sha256(raw).hexdigest()
    path = directory / (sha + ".bin")
    if not path.exists():
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    result = artifact(path, dtype=array.dtype.str, shape=list(array.shape))
    if result["sha256"] != sha:
        raise ValueError("deduplicated array bytes differ")
    return result


def read_array(item):
    path = verify_artifact(item)
    dtype = np.dtype(item["dtype"])
    shape = tuple(item["shape"])
    if np.prod(shape, dtype=np.int64) * dtype.itemsize != item["bytes"]:
        raise ValueError("array shape/bytes differ")
    return np.fromfile(path, dtype=dtype).reshape(shape)


def import_bridge(path):
    loader = importlib.machinery.ExtensionFileLoader("hms_native_bridge", str(Path(path).resolve()))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def check_environment():
    actual = {name: importlib.metadata.version(name) for name in VERSIONS}
    if actual != VERSIONS or platform.python_version() != "3.13.9":
        raise ValueError(f"tool versions differ from pinned protocol: {actual}")
    return actual


def affinity(cpu=None):
    if not hasattr(os, "sched_setaffinity"):
        return {"status": "unsupported_by_os", "cpus": None}
    available = sorted(os.sched_getaffinity(0))
    selected = available[0] if cpu is None else cpu
    if selected not in available:
        raise ValueError("requested CPU is outside process affinity")
    os.sched_setaffinity(0, {selected})
    return {"status": "applied", "cpus": sorted(os.sched_getaffinity(0))}


def machine():
    return {"architecture": platform.machine(), "os": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__,
            "faiss_compile_options": faiss.get_compile_options(),
            "thread_environment": {name: os.environ[name] for name in THREAD_ENV},
            "faiss_threads": faiss.omp_get_max_threads()}


def native_runtime_artifacts(system, bridge_path):
    if system == "hms":
        paths = [Path(bridge_path)]
    elif system == "hnswlib":
        paths = [Path(hnswlib.__file__)]
    else:
        package = Path(faiss.__file__).parent
        paths = [p for p in package.rglob("*") if p.is_file() and p.suffix in {".so", ".dylib"}]
        for folder in [package.parent / "faiss_cpu.libs", package / ".dylibs"]:
            if folder.exists():
                paths.extend(p for p in folder.rglob("*") if p.is_file() and (p.suffix == ".dylib" or ".so" in p.name))
    return [artifact(p) for p in sorted(set(paths))]


def load_sample():
    return {"unix_time": time.time(), "monotonic_ns": time.perf_counter_ns(), "one_minute": os.getloadavg()[0]}


def idle_sample(max_wait):
    deadline = time.monotonic() + max_wait
    while True:
        sample = load_sample()
        if sample["one_minute"] < 3:
            return sample
        if time.monotonic() >= deadline:
            raise RuntimeError("one-minute load gate >= 3; no timing started")
        time.sleep(min(5, max(0, deadline - time.monotonic())))


def timer_start():
    return {"clock": "perf_counter_ns", "start_unix_time": time.time(), "start_ns": time.perf_counter_ns()}


def timer_end(timer):
    end = time.perf_counter_ns()
    return {**timer, "end_ns": end, "elapsed_ns": end - timer["start_ns"], "end_unix_time": time.time()}


def rss_bytes():
    if platform.system() == "Darwin":
        class ProcTaskInfo(__import__("ctypes").Structure):
            _fields_ = [("virtual_size", __import__("ctypes").c_uint64), ("resident_size", __import__("ctypes").c_uint64), ("rest", __import__("ctypes").c_uint8 * 80)]
        import ctypes
        library = ctypes.CDLL("/usr/lib/libproc.dylib")
        info = ProcTaskInfo()
        library.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        n = library.proc_pidinfo(os.getpid(), 4, 0, ctypes.byref(info), ctypes.sizeof(info))
        if n < 16:
            raise RuntimeError("proc_pidinfo failed")
        return int(info.resident_size)
    with Path("/proc/self/statm").open() as handle:
        return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if platform.system() == "Darwin" else peak * 1024)


def normalized(rows, query=False):
    rows = np.asarray(rows, dtype=np.float32, order="C")
    if rows.ndim != 2 or not np.isfinite(rows).all():
        raise ValueError("vectors must be a finite float32 matrix")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if query and np.any(norms == 0):
        raise ValueError("angular zero vector cannot be served")
    return np.ascontiguousarray(rows / np.where(norms == 0, 1, norms), dtype=np.float32)


def load_train(dataset):
    verify_artifact(dataset["hdf5"])
    with h5py.File(dataset["hdf5"]["path"], "r") as file:
        train = np.asarray(file["train"], dtype=np.float32)
    if train.shape != (dataset["n_train"], dataset["dim"]):
        raise ValueError("train shape differs")
    return train


def grids(name, dim):
    ef = [16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096]
    variants = []
    def add(system, build, search):
        build_id = fingerprint({"dataset": name, "system": system, "build": build})
        variants.append({"system": system, "build_id": build_id, "build_config": build,
                         "search_grid": [{"point_id": fingerprint({"build_id": build_id, "search": p}), "search_config": p} for p in search]})
    add("hms", {"degree": 64 if dim == 256 else 32, "build_ef": 200, "alpha": 1.0,
                "seed": 0x5EED, "bits": 8, "residual_bits": 8, "align": True,
                "reorder": False, "id_bytes": 3},
        [{"ef": e, "rerank": r, "patience": p} for e in ef for r in ([16, 64] if dim == 256 else [16]) for p in [0, 256 if dim == 256 else 384]])
    for m in [16, 32]:
        add("faiss_hnsw", {"M": m, "ef_construction": 200, "seed": SEED}, [{"ef": e} for e in ef])
        add("hnswlib", {"M": m, "ef_construction": 200, "seed": SEED}, [{"ef": e} for e in ef])
    for nlist in [4096]:
        for divisor in [4, 1]:
            add("faiss_ivfpq", {"nlist": nlist, "pq_m": dim // divisor, "pq_bits": 8, "train_rows": 262144,
                                "niter": 20, "seed": SEED, "metric": "inner_product", "by_residual": True},
                [{"nprobe": p} for p in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] if p <= nlist])
    return variants


def preregister(args):
    check_environment()
    references = []
    for rel in ["ann_benchmarks/plotting/metrics.py", "ann_benchmarks/distance.py"]:
        url = f"https://raw.githubusercontent.com/erikbern/ann-benchmarks/{COMMITS['ann_benchmarks']}/{rel}"
        path = Path(args.reference_dir) / Path(rel).name
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
        if path.exists() and path.read_bytes() != content:
            raise ValueError("pinned reference bytes changed")
        path.write_bytes(content)
        references.append({**artifact(path), "url": url, "source_path": rel, "source_commit": COMMITS["ann_benchmarks"]})
    datasets = {}
    for name in ["nytimes-256-angular", "glove-100-angular"]:
        path = Path(args.data_dir) / f"{name}.hdf5"
        with h5py.File(path, "r") as file:
            n, dim = file["train"].shape
            if file.attrs.get("distance", "angular") != "angular":
                raise ValueError("native harness accepts angular datasets only")
            zero_count = sum(int(np.sum(~np.any(file["train"][start:start + 65536] != 0, axis=1))) for start in range(0, n, 65536))
        datasets[name] = {"hdf5": artifact(path), "n_train": n, "dim": dim, "n_test": 10000, "zero_train_rows": zero_count, "variants": grids(name, dim)}
    sources = [SOURCE_DIR / "native_harness.py", SOURCE_DIR / "build_native_bridge.py", *sorted((SOURCE_DIR / "native_bridge").rglob("*.rs")),
               SOURCE_DIR / "native_bridge/Cargo.toml", SOURCE_DIR / "native_bridge/Cargo.lock",
               *sorted((ROOT / "src/core/qgraph").glob("*.rs"))]
    binary = artifact(args.bridge)
    receipt = json.loads(Path(args.build_receipt).read_text())
    if receipt["binary"] != binary or receipt["exit_code"] != 0 or not receipt["source_identity_verified"]:
        raise ValueError("bridge build receipt differs")
    archived = json.loads(Path(args.archive_manifest).read_text())
    mapping = {entry["original"]["path"]: entry["archived"] for entry in archived["mappings"]}
    for source in sources:
        original = artifact(source)
        copy = mapping[original["path"]]
        verify_artifact(copy)
        if original["sha256"] != copy["sha256"]:
            raise ValueError("archived producer source differs")
    value = {"schema_version": 1, "frozen": True, "created_unix_time": time.time(), "producer_artifacts": [mapping[str(p.resolve())] for p in sources],
             "bridge": binary, "build_receipt": artifact(args.build_receipt), "bridge_build": mapping[str(Path(args.build_receipt).resolve())],
             "archive_manifest": artifact(args.archive_manifest), "archived_source_map": archived["mappings"],
             "generated_sources": receipt["generated_sources"], "bridge_linked_libraries": receipt["linked_libraries"],
             "mandatory_preregistration": artifact(args.preregistration),
             "references": references, "versions": VERSIONS, "commits": COMMITS,
             "hms_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "machine": machine(), "compiler": subprocess.check_output(["rustc", "--version"], text=True).strip(),
             "rustflags": receipt["environment"]["RUSTFLAGS"], "datasets": datasets,
             "targets": [0.90, 0.95], "k": K, "recall_definition": "ann-benchmarks angular distance <= tenth true distance + 1e-3",
             "epsilon": EPSILON, "recall_epsilon": EPSILON, "load_threshold": 3,
             "bootstrap": {"seed": SEED, "samples": 10000, "confidence": 0.95, "statistic": "paired_geometric_qps_ratio"},
             "coverage": {mode: "all_10000_test_queries_each_of_11_rounds" for mode in ["single_query_loop", "batch_throughput", "one_query_latency"]},
             "runtime_artifacts": {system: native_runtime_artifacts(system, args.bridge) for system in SYSTEMS},
             "serving_accounting": "index plus all distributed system-native binary files; common Python/NumPy runtime and OS libraries reported separately",
             "common_runtime_artifacts": [artifact(sys.executable), *[artifact(p) for p in sorted(Path(np.__file__).parent.rglob("*")) if p.is_file() and p.suffix in {".so", ".dylib"}]],
             "tuning": {"seed": SEED, "holdout_queries": 512, "calibration_base_rows": 65536, "calibration_cushion": 0.015,
                        "eligible_train_rows": "nonzero raw vector rows; sample from eligible IDs without replacement, first512 query IDs and remaining65536 base IDs, both sorted",
                        "candidate_rule": "first_three_at_target_plus_cushion_per_build_and_highest_train_recall_point_and_largest_search_without_patience",
                        "headroom_rule": {"additional_larger_values": 2, "search_depth_keys": ["ef", "nprobe"],
                                          "other_fields_unchanged": True, "recursive": False,
                                          "seed_points": "all_train_selected_points_including_train_best_and_fixed_ceiling",
                                          "selection": "next_two_larger_existing_search_grid_depth_values"},
                        "expected_failure_mode": "full-size neighbor distribution may need more work than train calibration; frozen strongest point can still miss target",
                        "test_driven_curve_expansion": False},
             "timing": {"rounds": 11, "alternation": "reverse_point_order_on_odd_rounds", "warmup_queries": 64,
                        "modes": ["single_query_loop", "batch_throughput", "one_query_latency"], "batch_size": 64,
                        "latency_queries": 10000, "load_gate": 3, "timer": "perf_counter_ns", "single_thread": True,
                        "max_lock_seconds": 900, "query_interface": "C-contiguous float32 matrix -> NumPy 2D integer labels; common int64 assignment inside timer"}}
    value["timing"]["retry_policy"] = {"maximum_attempts": 3, "eligible_causes": ["load_gate_not_met", "timeout_before_remaining_modes"],
                                          "retain_first_completed_gated_mode": True, "parameters_and_query_ids_unchanged": True,
                                          "choose_attempt_by_qps": False, "retry_only_missing_or_failed_modes": True,
                                          "finish_current_point_before_next": True, "gate_wait_counts_as_attempt": False,
                                          "completed_mode_after_worker_timeout": "retain if scored, gated, and enclosed by a released lock shorter than900s"}
    write_json(args.out, value, exclusive=True)
    print(canonical({"preregistration": artifact(args.out), "test_opened": False}))


def read_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get("schema_version") != 1 or protocol.get("frozen") is not True:
        raise ValueError("protocol is not frozen v1")
    check_environment()
    for item in protocol["producer_artifacts"] + protocol["references"] + [protocol["bridge"], protocol["build_receipt"], protocol["mandatory_preregistration"]]:
        verify_artifact(item)
    if machine()["architecture"] != protocol["machine"]["architecture"]:
        raise ValueError("machine architecture differs from frozen build")
    if file_hash(Path(__file__).resolve()) not in {a["sha256"] for a in protocol["producer_artifacts"]}:
        raise ValueError("executing producer differs from frozen source")
    return protocol


class Adapter:
    def __init__(self, system, build, bridge):
        self.system, self.build = system, build
        self.bridge = bridge
        self.index = None

    def fit(self, train):
        d = train.shape[1]
        p = self.build
        if self.system == "hms":
            self.index = self.bridge.NativeIndex.build(train, canonical(p))
        elif self.system == "faiss_hnsw":
            self.index = faiss.IndexHNSWFlat(d, p["M"], faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = p["ef_construction"]
            self.index.hnsw.rng = faiss.RandomGenerator(p["seed"])
            self.index.add(train)
        elif self.system == "hnswlib":
            self.index = hnswlib.Index(space="ip", dim=d)
            self.index.init_index(max_elements=len(train), M=p["M"], ef_construction=p["ef_construction"], random_seed=p["seed"])
            self.index.set_num_threads(1)
            self.index.add_items(train, np.arange(len(train), dtype=np.int64), num_threads=1)
        elif self.system == "faiss_ivfpq":
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFPQ(quantizer, d, p["nlist"], p["pq_m"], p["pq_bits"], faiss.METRIC_INNER_PRODUCT)
            self.index.cp.seed = self.index.pq.cp.seed = p["seed"]
            self.index.cp.niter = self.index.pq.cp.niter = p["niter"]
            self.index.by_residual = p["by_residual"]
            ids = np.sort(np.random.default_rng(p["seed"]).choice(len(train), min(len(train), p["train_rows"]), replace=False))
            self.index.train(np.ascontiguousarray(train[ids]))
            self.index.add(train)
        else:
            raise ValueError("unknown native system")

    def save(self, path):
        if self.system == "hms":
            self.index.save(str(path))
        elif self.system == "hnswlib":
            self.index.save_index(str(path))
        else:
            faiss.write_index(self.index, str(path))

    def load(self, path, dim):
        if self.system == "hms":
            self.index = self.bridge.NativeIndex.load(str(path))
        elif self.system == "hnswlib":
            self.index = hnswlib.Index(space="ip", dim=dim)
            self.index.load_index(str(path))
            self.index.set_num_threads(1)
        else:
            self.index = faiss.read_index(str(path))

    def configure(self, search):
        self.search = search
        if self.system == "hnswlib":
            self.index.set_ef(search["ef"])
        elif self.system == "faiss_hnsw":
            self.index.hnsw.efSearch = search["ef"]
        elif self.system == "faiss_ivfpq":
            self.index.nprobe = search["nprobe"]

    def query(self, q, k):
        if self.system == "hms":
            return self.index.query(q, k, self.search["ef"], self.search["rerank"], self.search["patience"])
        if self.system == "hnswlib":
            return self.index.knn_query(q, k=k, num_threads=1)[0]
        return self.index.search(q, k)[1]

    def memory(self):
        if self.system == "hms":
            payload, allocated, workspace = self.index.memory()
            return {"index_payload_bytes": payload, "index_allocated_bytes": allocated, "persistent_workspace_bytes": workspace,
                    "internal_accounting": "exact Rust field sizes and heap capacities; allocator metadata excluded"}
        return {"index_payload_bytes": None, "index_allocated_bytes": None, "persistent_workspace_bytes": None,
                "internal_accounting": "native internal allocations unavailable; serialized bytes and process RSS measured separately"}


def exact_distances(raw_train, raw_queries, labels):
    distances = np.full(labels.shape, np.inf, dtype="<f4")
    for i, (query, found) in enumerate(zip(raw_queries, labels, strict=True)):
        qnorm = np.sum(query ** 2) ** 0.5
        seen = set()
        for j, label in enumerate(found):
            label = int(label)
            if label < 0:
                continue
            if label >= len(raw_train) or label in seen:
                raise ValueError("native result is duplicate or out of range")
            seen.add(label)
            vector = raw_train[label]
            vnorm = np.sum(vector ** 2) ** 0.5
            with np.errstate(invalid="ignore", divide="ignore"):
                distances[i, j] = 1 - np.dot(query, vector) / (qnorm * vnorm)
    return distances


def scores(raw_train, raw_queries, labels, truth_ids, truth_distances):
    distances = exact_distances(raw_train, raw_queries, labels)
    hits = np.sum(distances <= truth_distances[:, K - 1, None] + EPSILON, axis=1, dtype=np.int16)
    strict = np.array([len(set(map(int, found)) & set(map(int, truth[:K]))) / K for found, truth in zip(labels, truth_ids, strict=True)])
    return {"distances_array": distances, "hits_array": hits.astype("<i2"),
            "recall_at_10": float(np.mean(hits) / K), "strict_id_recall_at_10": float(np.mean(strict))}


def select_curve_points(points, search_grid, targets, cushion, headroom_steps):
    retained = set()
    for target in targets:
        qualifying = [point for point in points if point["recall_at_10"] >= target + cushion]
        retained.update(point["point_id"] for point in qualifying[:3])
    retained.add(max(points, key=lambda point: point["recall_at_10"])["point_id"])
    ceiling = [point for point in search_grid if point["search_config"].get("patience", 0) == 0][-1]
    retained.add(ceiling["point_id"])
    seeds = [point for point in search_grid if point["point_id"] in retained]
    for seed in seeds:
        search = seed["search_config"]
        depth_keys = set(search) & {"ef", "nprobe"}
        if len(depth_keys) != 1:
            raise ValueError("search grid needs exactly one frozen depth key")
        depth = depth_keys.pop()
        other = {key: value for key, value in search.items() if key != depth}
        larger = [point for point in search_grid
                  if point["search_config"].get(depth, 0) > search[depth]
                  and {key: value for key, value in point["search_config"].items() if key != depth} == other]
        larger.sort(key=lambda point: point["search_config"][depth])
        retained.update(point["point_id"] for point in larger[:headroom_steps])
    return [point for point in search_grid if point["point_id"] in retained]


def tune(args):
    protocol = read_protocol(args.protocol)
    bridge = import_bridge(protocol["bridge"]["path"])
    records = []
    selected = []
    for name, dataset in protocol["datasets"].items():
        raw = load_train(dataset)
        eligible = np.flatnonzero(np.any(raw != 0, axis=1))
        sample = np.random.default_rng(SEED).choice(eligible, protocol["tuning"]["holdout_queries"] + protocol["tuning"]["calibration_base_rows"], replace=False)
        ids = np.sort(sample[:protocol["tuning"]["holdout_queries"]])
        keep = np.sort(sample[protocol["tuning"]["holdout_queries"]:])
        queries, base = normalized(raw[ids], query=True), normalized(raw[keep])
        flat = faiss.IndexFlatIP(dataset["dim"])
        flat.add(base)
        _, truth = flat.search(queries, K)
        true_distances = exact_distances(raw[keep], raw[ids], truth)
        split = {"query_ids": save_array(ids.astype("<i8"), args.artifacts_dir), "base_ids": save_array(keep.astype("<i8"), args.artifacts_dir),
                 "truth_ids": save_array(truth.astype("<i8"), args.artifacts_dir), "truth_distances": save_array(true_distances, args.artifacts_dir)}
        del flat
        for variant in dataset["variants"]:
            adapter = Adapter(variant["system"], variant["build_config"], bridge)
            adapter.fit(base)
            points = []
            for point in variant["search_grid"]:
                adapter.configure(point["search_config"])
                found = np.asarray(adapter.query(queries, K), dtype="<i8")
                score = scores(raw[keep], raw[ids], found, truth, true_distances)
                entry = {**point, "recall_at_10": score["recall_at_10"], "strict_id_recall_at_10": score["strict_id_recall_at_10"],
                         "predictions": save_array(found, args.artifacts_dir), "distances": save_array(score["distances_array"], args.artifacts_dir),
                         "hits": save_array(score["hits_array"], args.artifacts_dir)}
                points.append(entry)
            chosen = select_curve_points(points, variant["search_grid"], protocol["targets"],
                                         protocol["tuning"]["calibration_cushion"], protocol["tuning"]["headroom_rule"]["additional_larger_values"])
            record = {"dataset": name, **variant, "split": split, "points": points, "retained_point_ids": [p["point_id"] for p in chosen]}
            records.append(record)
            selected.append({"dataset": name, "system": variant["system"], "build_id": variant["build_id"], "build_config": variant["build_config"], "points": chosen})
            write_json(args.out, {"schema_version": 1, "complete": False, "preregistration": artifact(args.protocol), "test_opened": False, "records": records})
            del adapter
            gc.collect()
    tuning = {"schema_version": 1, "complete": True, "preregistration": artifact(args.protocol), "test_opened": False, "records": records}
    write_json(args.out, tuning)
    frozen_at = time.time()
    flat_points = [{"dataset": v["dataset"], "system": v["system"], "build_id": v["build_id"], "build_config": v["build_config"], **p} for v in selected for p in v["points"]]
    write_json(args.curve_out, {"schema_version": 1, "frozen": True, "created_unix_time": frozen_at, "frozen_at_unix": frozen_at,
                              "preregistration": artifact(args.protocol), "tuning": artifact(args.out), "variants": selected,
                              "points": flat_points, "modes": protocol["timing"]["modes"], "round_count": 11,
                              "test_driven_curve_expansion": False}, exclusive=True)
    print(canonical({"tuning": artifact(args.out), "frozen_curve": artifact(args.curve_out), "test_opened": False}))


def read_curve(path, protocol_path):
    curve = json.loads(Path(path).read_text())
    if curve.get("frozen") is not True or curve.get("test_driven_curve_expansion") is not False:
        raise ValueError("curve is not frozen")
    if curve["preregistration"] != artifact(protocol_path):
        raise ValueError("curve protocol fingerprint differs")
    verify_artifact(curve["tuning"])
    return curve


def build(args):
    protocol = read_protocol(args.protocol)
    curve = read_curve(args.curve, args.protocol)
    variant = next(v for v in curve["variants"] if v["build_id"] == args.build_id)
    dataset = protocol["datasets"][variant["dataset"]]
    bridge = import_bridge(protocol["bridge"]["path"])
    placement = affinity(args.cpu)
    baseline = rss_bytes()
    train = normalized(load_train(dataset))
    adapter = Adapter(variant["system"], variant["build_config"], bridge)
    before = idle_sample(args.max_wait)
    clock = timer_start()
    adapter.fit(train)
    clock = timer_end(clock)
    after = load_sample()
    after_build = rss_bytes()
    index_path = Path(args.index_dir) / (variant["build_id"] + ".index")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        raise FileExistsError("frozen full-train index already exists")
    adapter.save(index_path)
    info = artifact(index_path)
    memory = adapter.memory()
    del train
    gc.collect()
    serving = [info] + protocol["runtime_artifacts"][variant["system"]]
    row = {"resource_id": fingerprint({"build_id": variant["build_id"], "index": info}),
           "dataset": variant["dataset"], "system": variant["system"], "build_id": variant["build_id"], "build_config": variant["build_config"],
           "index_artifact": info, "index_bytes": info["bytes"], "serving_artifacts": serving, "serving_file_bytes": sum(a["bytes"] for a in serving),
           "common_runtime_artifacts": protocol["common_runtime_artifacts"], "common_runtime_file_bytes": sum(a["bytes"] for a in protocol["common_runtime_artifacts"]),
           "full_serving_file_bytes": sum(a["bytes"] for a in serving) + sum(a["bytes"] for a in protocol["common_runtime_artifacts"]),
           "bootstrap_sidecars": [],
           "index_payload_bytes": memory["index_payload_bytes"], "index_allocated_bytes": memory["index_allocated_bytes"],
           "workspace": {"persistent_bytes": memory["persistent_workspace_bytes"], "query_interface_peak_bytes": 0,
                         "observed_rss_growth_peak_bytes": None, "observed_rss_endpoint_growth_bytes": None,
                         "internal_accounting": memory["internal_accounting"]},
           "rss": {"baseline_bytes": baseline, "after_build_bytes": after_build, "after_load_bytes": None,
                   "peak_build_process_bytes": peak_rss_bytes(), "peak_serving_process_bytes": None},
           "build": {"timer": clock, "load_samples": [before, after], "load_gate_met": before["one_minute"] < 3 and after["one_minute"] < 3,
                     "preliminary": not (before["one_minute"] < 3 and after["one_minute"] < 3),
                     "timing_claim": before["one_minute"] < 3 and after["one_minute"] < 3},
           "machine": machine(), "affinity": placement, "preregistration": artifact(args.protocol), "frozen_curve": artifact(args.curve)}
    write_json(args.out, row, exclusive=True)
    print(canonical({"resource": artifact(args.out), "build_id": args.build_id}))


def frame_queries(queries, batch_size):
    return [queries[start:start + batch_size] for start in range(0, len(queries), batch_size)]


def run_query_loop(adapter, frames, out, per_query_latency=False):
    latencies = []
    offset = 0
    for frame in frames:
        start = time.perf_counter_ns() if per_query_latency else None
        result = adapter.query(frame, K)
        out[offset:offset + len(frame)] = result
        if per_query_latency:
            latencies.append(time.perf_counter_ns() - start)
        offset += len(frame)
    return np.asarray(latencies, dtype="<i8")


def round_point(args):
    protocol = read_protocol(args.protocol)
    curve = read_curve(args.curve, args.protocol)
    lock = verify_inherited_lock(args.lock_record, args.lock_id)
    if not 0 <= args.round < protocol["timing"]["rounds"]:
        raise ValueError("round outside frozen protocol")
    if not 0 <= args.attempt < protocol["timing"]["retry_policy"]["maximum_attempts"]:
        raise ValueError("attempt outside frozen retry policy")
    if Path(args.out).exists():
        raise FileExistsError("round attempt output already exists; retain prior evidence")
    modes = protocol["timing"]["modes"] if args.modes is None else args.modes
    if modes != [mode for mode in protocol["timing"]["modes"] if mode in modes] or not modes:
        raise ValueError("mode subset must preserve frozen order without duplicates")
    all_points = [(v, p) for v in curve["variants"] for p in v["points"] if v["dataset"] == args.dataset]
    if args.round % 2:
        all_points.reverse()
    order = [p["point_id"] for _, p in all_points]
    variant, point = next((v, p) for v, p in all_points if p["point_id"] == args.point_id)
    resource_row = json.loads(Path(args.resource).read_text())
    if resource_row["build_id"] != variant["build_id"] or resource_row["frozen_curve"] != artifact(args.curve):
        raise ValueError("resource and frozen curve differ")
    verify_artifact(resource_row["index_artifact"])
    dataset = protocol["datasets"][args.dataset]
    bridge = import_bridge(protocol["bridge"]["path"])
    placement = affinity(args.cpu)
    baseline = rss_bytes()
    adapter = Adapter(variant["system"], variant["build_config"], bridge)
    adapter.load(resource_row["index_artifact"]["path"], dataset["dim"])
    adapter.configure(point["search_config"])
    after_load = rss_bytes()
    # Test arrays are opened only after every frozen artifact and parameter is validated.
    verify_artifact(dataset["hdf5"])
    with h5py.File(dataset["hdf5"]["path"], "r") as file:
        if len(file["test"]) != dataset["n_test"]:
            raise ValueError("test query count differs")
        if not 0 <= args.query_start < args.query_end <= dataset["n_test"]:
            raise ValueError("query shard outside frozen dataset")
        raw_queries = np.asarray(file["test"][args.query_start:args.query_end], dtype=np.float32)
        truth = np.asarray(file["neighbors"][args.query_start:args.query_end, :K], dtype=np.int64)
        truth_distances = np.asarray(file["distances"][args.query_start:args.query_end, :K], dtype=np.float32)
    queries = normalized(raw_queries, query=True)
    for frame in frame_queries(queries[:protocol["timing"]["warmup_queries"]], 1):
        adapter.query(frame, K)
    measurements = []
    envelope = {"schema_version": 1, "preregistration": artifact(args.protocol), "frozen_curve": artifact(args.curve),
                "resource": artifact(args.resource), "complete": False, "pending_scoring": True,
                "requested_modes": modes, "rows": measurements}
    process_deadline = time.monotonic() + protocol["timing"]["max_lock_seconds"] - 10
    for mode in modes:
        if time.monotonic() >= process_deadline:
            envelope["incomplete_reason"] = "timeout_before_remaining_modes"
            break
        count = len(queries) if mode != "one_query_latency" else min(len(queries), protocol["timing"]["latency_queries"])
        batch_size = protocol["timing"]["batch_size"] if mode == "batch_throughput" else 1
        frames = frame_queries(queries[:count], batch_size)
        out = np.empty((count, K), dtype="<i8")
        rss_before = rss_bytes()
        try:
            load_before = idle_sample(min(args.max_wait, max(0, process_deadline - time.monotonic())))
        except RuntimeError:
            envelope["incomplete_reason"] = "load_gate_not_met"
            break
        clock = timer_start()
        latencies = run_query_loop(adapter, frames, out, mode == "one_query_latency")
        clock = timer_end(clock)
        load_after = load_sample()
        rss_after = rss_bytes()
        met = load_before["one_minute"] < 3 and load_after["one_minute"] < 3
        memory = adapter.memory()
        row = {"dataset": args.dataset, "system": variant["system"], "build_id": variant["build_id"], "point_id": args.point_id,
               "search_config": point["search_config"], "resource_id": resource_row["resource_id"], "round": args.round,
               "order_index": order.index(args.point_id), "order": order, "mode": mode, "batch_size": batch_size,
               "lock_id": lock["lock_id"],
               "attempt": args.attempt,
               "n_queries": count, "query_start": args.query_start, "query_end": args.query_start + count,
               "predictions": save_array(out, args.artifacts_dir), "latencies_ns": save_array(latencies, args.artifacts_dir) if len(latencies) else None,
               "timer": clock, "load_samples": [load_before, load_after], "load_gate_met": met, "preliminary": not met, "timing_claim": met,
               "workspace": {"persistent_bytes": memory["persistent_workspace_bytes"],
                             "query_interface_peak_bytes": out.nbytes + batch_size * K * (8 if variant["system"] == "hms" else 12),
                             "observed_rss_growth_peak_bytes": None, "observed_rss_endpoint_growth_bytes": max(0, rss_after - rss_before),
                             "internal_accounting": memory["internal_accounting"]},
               "rss": {"baseline_bytes": baseline, "after_load_bytes": after_load,
                       "before_query_bytes": rss_before, "after_query_bytes": rss_after, "peak_serving_process_bytes": peak_rss_bytes()},
               "machine": machine(), "affinity": placement}
        measurements.append(row)
        write_json(args.out, envelope)
    raw_train = load_train(dataset) if measurements else None
    for row in measurements:
        labels = read_array(row["predictions"])
        n = row["n_queries"]
        score = scores(raw_train, raw_queries[:n], labels, truth[:n], truth_distances[:n])
        row.update({"recall_at_10": score["recall_at_10"], "strict_id_recall_at_10": score["strict_id_recall_at_10"],
                    "distances": save_array(score["distances_array"], args.artifacts_dir), "hits": save_array(score["hits_array"], args.artifacts_dir)})
    write_json(args.out, {**envelope, "complete": len(measurements) == len(modes), "pending_scoring": False})
    print(canonical({"round_result": artifact(args.out), "all_load_gates_met": all(r["load_gate_met"] for r in measurements)}))


def score_round(args):
    protocol = read_protocol(args.protocol)
    read_curve(args.curve, args.protocol)
    entry = json.loads(Path(args.round_file).read_text())
    if entry["preregistration"] != artifact(args.protocol) or entry["frozen_curve"] != artifact(args.curve):
        raise ValueError("recovered round frozen identity differs")
    if not entry["rows"]:
        raise ValueError("no completed modes to recover")
    names = {row["dataset"] for row in entry["rows"]}
    if len(names) != 1:
        raise ValueError("recovered round mixes datasets")
    dataset = protocol["datasets"][names.pop()]
    raw_train = load_train(dataset)
    for row in entry["rows"]:
        start, end = row["query_start"], row["query_end"]
        if not 0 <= start < end <= dataset["n_test"] or end - start != row["n_queries"]:
            raise ValueError("recovered query interval differs")
        with h5py.File(dataset["hdf5"]["path"], "r") as file:
            queries = np.asarray(file["test"][start:end], dtype=np.float32)
            truth = np.asarray(file["neighbors"][start:end, :K], dtype=np.int64)
            true_distances = np.asarray(file["distances"][start:end, :K], dtype=np.float32)
        labels = read_array(row["predictions"])
        score = scores(raw_train, queries, labels, truth, true_distances)
        row.update({"recall_at_10": score["recall_at_10"], "strict_id_recall_at_10": score["strict_id_recall_at_10"],
                    "distances": save_array(score["distances_array"], args.artifacts_dir), "hits": save_array(score["hits_array"], args.artifacts_dir)})
    entry.update({"complete": True, "pending_scoring": False, "recovered_from": artifact(args.round_file), "recovery_timing_claim": False})
    write_json(args.out, entry, exclusive=True)
    print(canonical({"recovered_round": artifact(args.out), "new_timing": False}))


def verify_inherited_lock(path, lock_id):
    lock = json.loads(Path(path).read_text())
    fd = int(os.environ["HMS_NATIVE_LOCK_FD"])
    expected = os.stat("./.hms-target/timing.lock")
    current = os.fstat(fd)
    if lock["status"] != "held" or lock["lock_id"] != lock_id or (expected.st_dev, expected.st_ino) != (current.st_dev, current.st_ino):
        raise ValueError("timing lock proof differs")
    probe = os.open("./.hms-target/timing.lock", os.O_RDWR)
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return lock
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise ValueError("inherited timing lock was not held")
    finally:
        os.close(probe)


def lock_run(args):
    command = args.worker_args
    if command[:1] == ["--"]:
        command = command[1:]
    if command[:1] != ["round-point"]:
        raise ValueError("lock-run accepts only a round-point invocation")
    idle_sample(args.max_wait)
    lock_path = "./.hms-target/timing.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    lock_id = str(uuid.uuid4())
    lock = None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = timer_start()
        lock = {"lock_id": lock_id, "status": "held", "pid": os.getpid(), "path": lock_path, "timer": acquired,
                "max_seconds": 900, "command": [sys.executable, str(Path(__file__).resolve()), *command]}
        write_json(args.lock_record, lock, exclusive=True)
        env = os.environ | {"HMS_NATIVE_LOCK_FD": str(fd)}
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), *command,
                                    "--lock-id", lock_id, "--lock-record", args.lock_record], env=env, pass_fds=(fd,), timeout=890)
        lock["exit_code"] = completed.returncode
    except subprocess.TimeoutExpired:
        lock["exit_code"] = 124
        lock["timeout"] = True
    finally:
        if lock is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
            lock["timer"] = timer_end(lock["timer"])
            lock["status"] = "released"
            lock["hold_seconds"] = lock["timer"]["elapsed_ns"] / 1e9
            lock["under_15_minutes"] = lock["hold_seconds"] < 900
            lock["preliminary"] = True
            lock["timing_claim"] = False
            lock["duration_scope"] = "operational lock hold; no ANN query claim"
            write_json(args.lock_record, lock)
        os.close(fd)
    if lock is None or lock["exit_code"] != 0 or not lock["under_15_minutes"]:
        raise SystemExit(1)


def select_attempts(attempt_rows, locks):
    by_lock = {lock["lock_id"]: lock for lock in locks}
    scored_fields = {"recall_at_10", "strict_id_recall_at_10", "distances", "hits"}
    found = {}
    for row in attempt_rows:
        key = (row["point_id"], row["round"], row["mode"], row["query_start"], row["query_end"], row["attempt"])
        if key in found:
            previous = found[key]
            if {k: v for k, v in previous.items() if k not in scored_fields} != {k: v for k, v in row.items() if k not in scored_fields}:
                raise ValueError("conflicting immutable attempt row")
            if scored_fields <= previous.keys() and scored_fields <= row.keys() and previous != row:
                raise ValueError("conflicting scored attempt row")
            if scored_fields <= row.keys():
                found[key] = row
        else:
            found[key] = row
    attempt_rows = sorted(found.values(), key=lambda r: r["timer"]["start_ns"])
    selected = {}
    preliminary = []
    for row in attempt_rows:
        lock = by_lock[row["lock_id"]]
        valid = (scored_fields <= row.keys() and row["load_gate_met"] and lock["status"] == "released"
                 and lock["exit_code"] in (0, 124)
                 and lock["under_15_minutes"] and lock["timer"]["start_ns"] <= row["timer"]["start_ns"]
                 < row["timer"]["end_ns"] <= lock["timer"]["end_ns"])
        key = (row["point_id"], row["round"], row["mode"], row["query_start"], row["query_end"])
        if valid and key not in selected:
            selected[key] = row
        else:
            preliminary.append({**row, "preliminary": True, "timing_claim": False,
                                "retention_reason": "earlier_valid_attempt_retained" if valid else "incomplete_or_failed_gate"})
    rows = list(selected.values())
    return rows, attempt_rows, preliminary


def assemble(args):
    protocol = read_protocol(args.protocol)
    curve = read_curve(args.curve, args.protocol)
    resources = [json.loads(Path(path).read_text()) for path in args.resources]
    rounds = [json.loads(Path(path).read_text()) for path in args.round_files]
    locks = [json.loads(Path(path).read_text()) for path in args.lock_records]
    attempt_rows = [row for entry in rounds for row in entry["rows"]]
    prior = None
    if args.command == "collect" and Path(args.out).exists():
        prior = json.loads(Path(args.out).read_text())
        if prior["preregistration"] != artifact(args.protocol) or prior["frozen_curve"] != artifact(args.curve):
            raise ValueError("collector frozen identity differs")
        resources.extend(prior["resources"])
        attempt_rows.extend(prior["attempt_rows"])
        locks.extend(prior["locks"])
    def unique(items, key):
        found = {}
        for item in items:
            identity = key(item)
            if identity in found and found[identity] != item:
                raise ValueError("conflicting incremental row")
            found[identity] = item
        return list(found.values())
    resources = unique(resources, lambda r: r["resource_id"])
    locks = unique(locks, lambda r: r["lock_id"])
    rows, attempt_rows, preliminary = select_attempts(attempt_rows, locks)
    for entry in rounds:
        if entry["preregistration"] != artifact(args.protocol) or entry["frozen_curve"] != artifact(args.curve):
            raise ValueError("round producer fingerprint differs")
    present = {(r["point_id"], r["round"], r["mode"]): [] for r in rows}
    for r in rows:
        present[(r["point_id"], r["round"], r["mode"])].append((r["query_start"], r["query_end"]))
    coverage_ok = True
    for point in curve["points"]:
        for round_index in range(11):
            for mode in curve["modes"]:
                intervals = sorted(present.get((point["point_id"], round_index, mode), []))
                position = 0
                for start, end in intervals:
                    if start != position:
                        coverage_ok = False
                    position = end
                coverage_ok &= position == protocol["datasets"][point["dataset"]]["n_test"]
    systems = {system: {"version": "frozen_hms_bridge" if system == "hms" else VERSIONS["hnswlib" if system == "hnswlib" else "faiss-cpu"],
                        "source_commit": protocol["hms_source_commit"] if system == "hms" else COMMITS["hnswlib" if system == "hnswlib" else "faiss"],
                        "architecture": platform.machine(), "compile_options": "-C target-cpu=native" if system == "hms" else (faiss.get_compile_options() if system.startswith("faiss") else None),
                        "module_artifacts": protocol["runtime_artifacts"][system], "thread_environment": THREAD_ENV} for system in SYSTEMS}
    datasets = {name: {k: v for k, v in data.items() if k != "variants"} for name, data in protocol["datasets"].items()}
    complete = coverage_ok and all(r["load_gate_met"] for r in rows)
    artifacts = [artifact(p) for p in args.resources + args.round_files + args.lock_records]
    if prior is not None:
        artifacts.extend(prior["artifacts"])
    artifacts = unique(artifacts, lambda a: (a["path"], a["sha256"]))
    serving_maxima = []
    for r in resources:
        matching = [row for row in rows if row["resource_id"] == r["resource_id"]]
        serving_maxima.append({"resource_id": r["resource_id"],
                               "peak_serving_process_bytes": max([row["rss"]["peak_serving_process_bytes"] for row in matching], default=None),
                               "persistent_workspace_bytes": max([row["workspace"]["persistent_bytes"] for row in matching if row["workspace"]["persistent_bytes"] is not None], default=None),
                               "query_interface_peak_bytes": max([row["workspace"]["query_interface_peak_bytes"] for row in matching], default=None),
                               "observed_rss_endpoint_growth_bytes": max([row["workspace"]["observed_rss_endpoint_growth_bytes"] for row in matching], default=None)})
    value = {"schema_version": 1, "complete": complete, "preregistration": artifact(args.protocol),
             "mandatory_preregistration": protocol["mandatory_preregistration"], "tuning": curve["tuning"],
             "frozen_curve": artifact(args.curve), "artifacts": artifacts,
             "datasets": datasets, "systems": systems, "resources": resources, "rows": rows, "locks": locks,
             "attempt_rows": attempt_rows, "preliminary_rows": preliminary, "serving_resource_maxima": serving_maxima}
    write_json(args.out, value)
    print(canonical({"result": artifact(args.out), "complete": complete, "rows": len(rows)}))


def archive(args):
    receipt = json.loads(Path(args.build_receipt).read_text())
    if receipt["exit_code"] != 0 or not receipt["source_identity_verified"]:
        raise ValueError("cannot archive failed or drifting bridge build")
    files = [SOURCE_DIR / "native_harness.py", SOURCE_DIR / "build_native_bridge.py", SOURCE_DIR / "test_native_harness.py",
             *[Path(a["path"]) for a in receipt["source_before"]], *[Path(a["path"]) for a in receipt["generated_sources"]],
             Path(args.build_receipt), Path(receipt["build_log"]["path"]), *[Path(p) for p in args.validation_artifacts]]
    directory = Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    mappings = []
    for file in sorted(set(files)):
        original = artifact(file)
        copy = directory / (original["sha256"] + "-" + file.name)
        if copy.exists() and file_hash(copy) != original["sha256"]:
            raise ValueError("archive bytes differ")
        if not copy.exists():
            shutil.copyfile(file, copy)
        mappings.append({"original": original, "archived": artifact(copy)})
    write_json(args.out, {"schema_version": 1, "mappings": mappings, "bridge_build": artifact(args.build_receipt)}, exclusive=True)
    print(canonical({"archive_manifest": artifact(args.out), "artifacts": len(mappings)}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("preregister")
    register.add_argument("--bridge", required=True)
    register.add_argument("--data-dir", default=str(Path.home() / ".cache/hms-bench/downloads"))
    register.add_argument("--reference-dir", required=True)
    register.add_argument("--build-receipt", required=True)
    register.add_argument("--preregistration", required=True)
    register.add_argument("--archive-manifest", required=True)
    register.add_argument("--out", required=True)
    register.set_defaults(run=preregister)
    tuning = sub.add_parser("tune")
    tuning.add_argument("--protocol", required=True)
    tuning.add_argument("--artifacts-dir", required=True)
    tuning.add_argument("--out", required=True)
    tuning.add_argument("--curve-out", required=True)
    tuning.set_defaults(run=tune)
    builder = sub.add_parser("build")
    builder.add_argument("--protocol", required=True)
    builder.add_argument("--curve", required=True)
    builder.add_argument("--build-id", required=True)
    builder.add_argument("--index-dir", required=True)
    builder.add_argument("--cpu", type=int)
    builder.add_argument("--max-wait", type=float, default=0)
    builder.add_argument("--out", required=True)
    builder.set_defaults(run=build)
    rounds = sub.add_parser("round-point")
    rounds.add_argument("--protocol", required=True)
    rounds.add_argument("--curve", required=True)
    rounds.add_argument("--resource", required=True)
    rounds.add_argument("--dataset", required=True)
    rounds.add_argument("--point-id", required=True)
    rounds.add_argument("--round", type=int, required=True)
    rounds.add_argument("--attempt", type=int, default=0)
    rounds.add_argument("--modes", nargs="+")
    rounds.add_argument("--query-start", type=int, default=0)
    rounds.add_argument("--query-end", type=int, default=10000)
    rounds.add_argument("--cpu", type=int)
    rounds.add_argument("--max-wait", type=float, default=0)
    rounds.add_argument("--artifacts-dir", required=True)
    rounds.add_argument("--out", required=True)
    rounds.add_argument("--lock-id", required=True)
    rounds.add_argument("--lock-record", required=True)
    rounds.set_defaults(run=round_point)
    recovered = sub.add_parser("score-round")
    recovered.add_argument("--protocol", required=True)
    recovered.add_argument("--curve", required=True)
    recovered.add_argument("--round-file", required=True)
    recovered.add_argument("--artifacts-dir", required=True)
    recovered.add_argument("--out", required=True)
    recovered.set_defaults(run=score_round)
    locked = sub.add_parser("lock-run")
    locked.add_argument("--lock-record", required=True)
    locked.add_argument("--max-wait", type=float, default=0)
    locked.add_argument("worker_args", nargs=argparse.REMAINDER)
    locked.set_defaults(run=lock_run)
    for name in ["assemble", "collect"]:
        assembly = sub.add_parser(name)
        assembly.add_argument("--protocol", required=True)
        assembly.add_argument("--curve", required=True)
        assembly.add_argument("--resources", nargs="+", required=True)
        assembly.add_argument("--round-files", nargs="+", required=True)
        assembly.add_argument("--lock-records", nargs="+", required=True)
        assembly.add_argument("--out", required=True)
        assembly.set_defaults(run=assemble)
    archiver = sub.add_parser("archive")
    archiver.add_argument("--build-receipt", required=True)
    archiver.add_argument("--validation-artifacts", nargs="+", required=True)
    archiver.add_argument("--directory", required=True)
    archiver.add_argument("--out", required=True)
    archiver.set_defaults(run=archive)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
