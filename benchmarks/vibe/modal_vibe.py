# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run the VIBE benchmark (HMS plus the published CPU leaders) on Modal, one container per dataset.

  uv run --with modal modal run --detach benchmarks/vibe/modal_vibe.py --resume
  uv run --with modal modal run benchmarks/vibe/modal_vibe.py --status
  uv run --with modal modal run --detach benchmarks/vibe/modal_vibe.py --resume --hms-only --hms-rev <sha>
  uv run --with modal modal run --detach benchmarks/vibe/modal_vibe.py --resume --retry-failed

VIBE is the dcondrey/vibe fork at VIBE_REV (upstream 7ac76c1 plus vibe/algorithms/hms). Every job is
`run.py --dataset D --algorithm A --local --parallelism 1` with VIBE's defaults (k=100, 5 runs, 24 h
per-run timeout) and single-threaded math libraries. HMS and the competitors are compiled in the
container that runs them (target-cpu=native / the image.def flags), because Modal containers land on
different CPUs; the CPU model and AVX-512 status go into every ledger and result record.

State lives on the "hms-bench-run" Volume under /vibe/<ns>/ (ns = LIVE_NS, or "test" with --test):
  data/<name>.hdf5 (+ .sha256)          dataset, verified against the Hugging Face LFS sha256
  ledger/<dataset>/<job>.json           one record per job; job = "hms@<rev>" or the competitor name
  results/<dataset>/<job>/results/...   VIBE's result tree for that job, synced every minute
  logs/, cpu/, standings/<dataset>@<rev>.json, plan.json, stop.json
A job is done only when every expected (index, query-args) group has its completion marker, opens,
and holds one row per test query. A retry keeps the complete groups and deletes everything else.
Spend is estimated from container wall time at Modal's published CPU and memory rates (the
"hms-vibe-spend" Dict, mirrored to the Volume): no job starts at --cap-usd, running jobs stop at
--hard-cap-usd.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import modal

VIBE_REPO = "https://github.com/dcondrey/vibe.git"
VIBE_REV = "60625ecd00c639ebe1b6ce19434e89dae7fedff0"
HMS_REPO = "https://github.com/writerslogic/holographic-memory.git"
HF = "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/{}.hdf5"
HF_TREE = "https://huggingface.co/api/datasets/vector-index-bench/vibe/tree/main"
PUBLISHED = "https://raw.githubusercontent.com/vector-index-bench/vector-index-bench.github.io/HEAD/{}"

# Float datasets this runner owns. Quantized and chamfer variants are out of scope because HMS indexes
# float vectors only. llama-128-ip and yi-128-ip joined later with their own HMS rev (--datasets).
DATASETS = [
    "agnews-mxbai-1024-euclidean", "arxiv-nomic-768-normalized", "ccnews-nomic-768-normalized",
    "celeba-resnet-2048-cosine", "coco-nomic-768-normalized", "codesearchnet-jina-768-cosine",
    "cqadupstack-lemur-2048-ip", "cqadupstack-muvera-5120-ip", "dpr-jina-768-normalized", "glove-200-cosine",
    "gooaq-distilroberta-768-normalized", "hotpotqa-harrier-640-normalized", "imagenet-align-640-normalized",
    "imagenet-clip-512-normalized", "inaturalist-resnet-2048-cosine", "laion-clip-512-normalized",
    "landmark-dino-768-cosine", "landmark-nomic-768-normalized", "msmarco-qwen-1024-normalized",
    "simplewiki-openai-3072-normalized", "yahoo-minilm-384-normalized", "yandex-200-cosine",
    "llama-128-ip", "yi-128-ip",
]
# Competitors run in addition to the published leaders, with the reason recorded in the ledger.
NAMED = {"llama-128-ip": ["roargraph"], "yi-128-ip": ["roargraph"]}
SMOKE = "smoke-64-cosine"  # generated in the container for --test
MAX_MEM_GIB = 224
LIVE_NS = "run-s-20261023"
THRESHOLDS = ("0.9", "0.95", "0.99")
# Competitors this image can build: algorithm -> (installer, micromamba env, needs AVX-512).
INSTALLABLE = {
    "ngt-qg": ("ngt", "base", False), "ngt-onng": ("ngt", "base", False), "glass": ("glass", "base", False),
    "lorann": ("lorann", "base", False), "roargraph": ("roargraph", "base", False), "pag": ("pag", "base", True),
    "symphonyqg": ("rabitq", "vibe-rabitq", False), "rabitq-hnsw": ("rabitq", "vibe-rabitq", False),
    "rabitq-ivf": ("rabitq", "vibe-rabitq", False), "mlann-rf": ("mlann", "base", False),
}
KNOWN_CPUS = {
    ("AuthenticAMD", "25", "17"): "AMD EPYC 9004 (Zen 4, Genoa; family 25 model 17)",
    ("AuthenticAMD", "25", "1"): "AMD EPYC 7003 (Zen 3, Milan; family 25 model 1)",
    ("AuthenticAMD", "23", "49"): "AMD EPYC 7002 (Zen 2, Rome; family 23 model 49)",
    ("GenuineIntel", "6", "143"): "Intel Xeon 4th gen (Sapphire Rapids; family 6 model 143)",
    ("GenuineIntel", "6", "106"): "Intel Xeon 3rd gen (Ice Lake-SP; family 6 model 106)",
    ("GenuineIntel", "6", "85"): "Intel Xeon (Skylake/Cascade Lake-SP; family 6 model 85)",
}
CPU_REQ, CPU_LIMIT = 2.0, 8.0
USD_CORE_H, USD_GIB_H = 0.0000131 * 3600, 0.00000222 * 3600  # Modal published rates
FN_TIMEOUT_S = 24 * 3600
CONTAINER_BUDGET_S = 23 * 3600  # hand over to a fresh container before Modal's 24 h limit
HEARTBEAT_S = 60
STALE_S = 15 * 60
MAX_FAILURES = 3
MM = "/opt/env/micromamba"
ENV_BIN = {"base": f"{MM}/bin", "vibe-rabitq": f"{MM}/envs/vibe-rabitq/bin", "plot": f"{MM}/envs/plot/bin"}
THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TBB_NUM_THREADS",
                               "NUMEXPR_NUM_THREADS", "VECLIB_NUM_THREADS", "BLIS_NUM_THREADS", "GOTO_NUM_THREADS",
                               "NUMBA_NUM_THREADS", "JULIA_NUM_THREADS", "RAYON_NUM_THREADS")}
THREAD_ENV |= {"OMP_DYNAMIC": "false", "MKL_DYNAMIC": "false", "HDF5_USE_FILE_LOCKING": "FALSE"}

VOL = modal.Volume.from_name("hms-bench-run", create_if_missing=True)
SPEND = modal.Dict.from_name("hms-vibe-spend", create_if_missing=True)
LEASE = modal.Dict.from_name("hms-vibe-lease", create_if_missing=True)
app = modal.App("hms-vibe")

image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("build-essential", "git", "curl", "wget", "cmake", "bzip2", "ca-certificates", "pkg-config", "libnuma1",
                 "libnuma-dev", "libblas-dev", "liblapack-dev", "libboost-dev", "libboost-program-options-dev", "util-linux")
    .run_commands(
        f"git clone -q {VIBE_REPO} /opt/vibe && git -C /opt/vibe checkout -q {VIBE_REV}",
        "curl -fL --retry 5 -o /tmp/mm.tar.bz2 https://micro.mamba.pm/api/micromamba/linux-64/2.6.2"
        " && tar -xjf /tmp/mm.tar.bz2 -C /usr/local bin/micromamba && rm /tmp/mm.tar.bz2",
        f"MAMBA_ROOT_PREFIX={MM} micromamba config set use_lockfiles false"
        f" && MAMBA_ROOT_PREFIX={MM} micromamba install -y -q -n base -f /opt/vibe/environment.yml",
        f"MAMBA_ROOT_PREFIX={MM} micromamba create -y -q -n vibe-rabitq -f /opt/vibe/environment.yml"
        f" && MAMBA_ROOT_PREFIX={MM} micromamba install -y -q -n vibe-rabitq -c conda-forge -c pytorch python=3.12 faiss-cpu=1.14.3"
        f" && {MM}/envs/vibe-rabitq/bin/pip install --no-cache-dir scikit-build-core",
        f"MAMBA_ROOT_PREFIX={MM} micromamba create -y -q -n plot -c conda-forge python=3.11 seaborn=0.13.2 pyarrow=23.0.1"
        " scipy=1.17.1 polars=1.39.0 scikit-learn=1.8.0 h5py=3.16.0 tqdm=4.67.3 networkx=3.6.1",
        f"MAMBA_ROOT_PREFIX={MM} micromamba clean --all --yes",
        f"{MM}/bin/pip install --no-cache-dir maturin==1.15.0 meson ninja scikit-build-core",
        "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.96.0",
    )
)
plan_image = modal.Image.debian_slim(python_version="3.12").pip_install("polars==1.39.0")

# Each installer follows vibe/algorithms/<name>/image.def; PATH puts the job's env first.
INSTALL_SH = r"""
set -ex
cd /work/build
ngt() {
  git clone -q https://github.com/intel/x86-simd-sort.git && cd x86-simd-sort && git checkout -q v7.0
  meson setup build -Dbuildtype=release && ninja -C build && meson install -C build && cd ..
  git clone -q https://github.com/NGT-labs/ngt.git && cd ngt && git reset --hard d46c0e46d46cdb698ba10884323e90c2901b3f57
  mkdir build && cd build
  cmake -DNGTQG_NO_ROTATION=ON -DNGTQG_ZERO_GLOBAL=ON -DNGTQG_X86SIMDSORT=ON -DNGT_DISABLE_NORMALIZATION_ERROR_CHECK=ON ..
  make -j "$(nproc)" && make install && cd .. && ldconfig
  cd python && python3 setup.py bdist_wheel && pip install --no-cache-dir dist/ngt-*-linux_x86_64.whl
}
glass() {
  git clone -q https://github.com/zilliztech/pyglass && cd pyglass && git reset --hard d2296ec447d2374ee8f88c6d3b85be1b1e434ad3
  pip install -v --no-build-isolation -e python
}
lorann() { pip install --no-cache-dir lorann==1.0; }
mlann() {
  git clone -q https://github.com/ejaasaari/mlann && cd mlann && git reset --hard de8f9d6ea977312f5a8943a9ccdbd11870ac590c
  pip install .
}
roargraph() {
  git clone -q --recursive https://github.com/matchyc/RoarGraph.git && cd RoarGraph && git checkout -q add_pyroar
  git reset --hard f2b49b6f47f16bccde9bc13d8fceffd5ff2dd4aa && cd pyroar
  sed -i '/faiss/d' index_bindings.cpp; sed -i '/faiss/d' setup.py
  pip install . --no-build-isolation
}
pag() {
  git clone -q https://github.com/KejingLu-810/PAG.git && cd PAG && git reset --hard ee34ed78b7a28225b2db6210ad942152e1cd540d
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPAG_BUILD_EXAMPLES=OFF -DPAG_NATIVE_ARCH=ON
  cmake --build build -j "$(nproc)" && cmake --build build --target install && ldconfig
  pip install --no-cache-dir --no-build-isolation --no-deps .
}
rabitq() {
  git clone -q --depth 1 --branch v0.3.7 https://github.com/VectorDB-NTU/RaBitQ-Library.git && cd RaBitQ-Library
  pip install --no-cache-dir --no-build-isolation --no-deps .
}
hms() {
  git clone -q "$HMS_REPO" hms && cd hms && git checkout -q "$HMS_REV"
  RUSTFLAGS="-C target-cpu=native" CARGO_BUILD_JOBS="$(nproc)" maturin build --release --locked --out /work/hms-dist
  pip install --no-cache-dir --force-reinstall --no-deps /work/hms-dist/*.whl
  cd / && python -c 'from holographic_memory import VectorIndex; print(VectorIndex)'
}
"""

# Runs inside a job directory with the env's python: expected groups, validation, partial cleanup.
JOBTOOL = r'''
import glob, json, os, re, sys
import h5py
from vibe.definitions import get_definitions
from vibe.main import parse_dataset_string
from vibe.results import build_result_filepath, _completion_marker_filepath

mode, ds, algo, defs_dir = sys.argv[1:5]
dim, dist, pt = parse_dataset_string(ds)
defs = [d for d in get_definitions(dimension=dim, point_type=pt, distance_metric=dist, count=100, base_dir=defs_dir)
        if d.algorithm == algo and not d.disabled]
expected = [build_result_filepath(ds, 100, d, q) for d in defs for q in (d.query_argument_groups or [[]])]
root = os.path.join("results", ds, "100", algo)

def opens(path):
    try:
        with h5py.File(path, "r") as f:
            return list(f.keys())
    except Exception:
        return None

if mode == "clean":
    removed = []
    for path in glob.glob(os.path.join(root, "*.hdf5")):
        groups = opens(path)
        if groups is None:
            os.remove(path)
            removed.append(path)
            continue
        stale = [g for g in groups if not os.path.exists(_completion_marker_filepath(path, g))]
        if stale:
            with h5py.File(path, "a") as f:
                for g in stale:
                    del f[g]
            removed += [f"{path}:{g}" for g in stale]
    for marker_dir in glob.glob(os.path.join(root, "*.hdf5.complete")):
        path = marker_dir[: -len(".complete")]
        groups = set(opens(path) or []) if os.path.exists(path) else set()
        for m in os.listdir(marker_dir):
            body = open(os.path.join(marker_dir, m)).read()
            if body not in groups:
                os.remove(os.path.join(marker_dir, m))
                removed.append(os.path.join(marker_dir, m))
    print(json.dumps({"removed": removed}))
else:
    with h5py.File(os.path.join("data", ds + ".hdf5"), "r") as f:
        n_test = int(f["test"].shape[0])
        has_learn = "learn" in f and "learn_neighbors" in f
    # An OOD-capable HMS entry must have trained on the learn split; learn_queries=0 means VIBE called fit().
    need_learn = {id(d) for d in defs if algo.startswith("hms") and has_learn and d.ood}
    valid, problems = 0, []
    for (path, sp), d in zip(expected, [d for d in defs for q in (d.query_argument_groups or [[]])]):
        try:
            if sp and not os.path.exists(_completion_marker_filepath(path, sp)):
                raise ValueError("no completion marker")
            with h5py.File(path, "r") as f:
                g = f[sp] if sp else f[list(f.keys())[0]]
                for key in ("times", "neighbors", "distances"):
                    if g[key].shape[0] != n_test:
                        raise ValueError(f"{key} has {g[key].shape[0]} rows, dataset has {n_test} queries")
                if id(d) in need_learn:
                    m = re.search(r"learn_queries=(\d+)", str(g.attrs.get("name", "")))
                    if not m or int(m.group(1)) == 0:
                        raise ValueError(f"OOD dataset but the run did not train on learn queries: {g.attrs.get('name')!r}")
            valid += 1
        except Exception as e:
            problems.append(f"{os.path.basename(path)}:{sp}: {type(e).__name__}: {e}"[:300])
    markers = len(glob.glob(os.path.join(root, "*.hdf5.complete", "*")))
    print(json.dumps({"expected": len(expected), "valid": valid, "test_count": n_test, "markers": markers,
                      "done": bool(expected) and valid == len(expected), "problems": problems[:20]}))
'''

# Loaded into every VIBE process through PYTHONPATH. Modal's gVisor sandbox reports one NUMA node but
# fails numa_node_to_cpus, so VIBE's worker placement raised before running anything. On a single-node
# machine node 0 holds every CPU the process may use; nothing else in VIBE changes.
NUMA_SHIM = r'''
import os, sys
sys.path.insert(0, "/opt/vibe")
try:
    from vibe import numa, workers
    _node_to_cpus, _allowed_mems = numa.node_to_cpus, workers.allowed_mems
    def node_to_cpus(node):
        try:
            return _node_to_cpus(node)
        except RuntimeError:
            if node == 0 and numa.get_max_node() == 0:
                return set(os.sched_getaffinity(0))
            raise
    numa.node_to_cpus = node_to_cpus
    workers.allowed_mems = lambda: _allowed_mems() or {0}
except Exception as e:
    print("numa shim not applied:", repr(e), file=sys.stderr)
finally:
    sys.path.remove("/opt/vibe")
'''

# Best QPS per algorithm at each recall threshold, from export_results.py's summary.parquet.
STANDING = r'''
import json, sys
import polars as pl
df = pl.read_parquet(sys.argv[1]).filter(pl.col("k") == 100)
out = {}
for algo in sorted(df["algorithm"].unique().to_list()):
    rows = {}
    for t in (0.90, 0.95, 0.99):
        s = df.filter((pl.col("algorithm") == algo) & (pl.col("recall") >= t)).sort("qps", descending=True)
        rows[f"{t:.2f}"] = None if s.is_empty() else {k: s[0, k] for k in ("qps", "recall", "params", "build_time", "index_size")}
    out[algo] = rows
print(json.dumps(out))
'''

# Tiny dataset for --test, VIBE layout with exact ground truth: `cosine` draws queries from the base
# distribution; `ip` gives the base vectors varying norms and shifts the query and learn distributions
# (out-of-distribution, as llama/yi are), with VIBE's negative-inner-product distance.
SMOKE_GEN = r'''
import sys
import h5py
import numpy as np
metric = sys.argv[2]
rng = np.random.default_rng(7)
n, nq, nl, d, K = 100_000, 1000, 5000, 64, 100
centers = rng.normal(size=(64, d)).astype(np.float32)
def sample(m):
    return (centers[rng.integers(0, 64, m)] + 0.6 * rng.normal(size=(m, d))).astype(np.float32)
train, test, learn = sample(n), sample(nq), sample(nl)
if metric == "ip":
    train *= rng.uniform(0.5, 1.5, (n, 1)).astype(np.float32)
    shift = (0.8 * rng.normal(size=d)).astype(np.float32)
    test, learn = test + shift, learn + shift
    base = train.T
    def prep(q):
        return q
    def dist(q):
        return -(q @ base)
else:
    base = (train / np.linalg.norm(train, axis=1, keepdims=True)).T
    def prep(q):
        return q / np.linalg.norm(q, axis=1, keepdims=True)
    def dist(q):
        return 1 - q @ base
def truth(q):
    q = prep(q)
    nb, ds = [], []
    for i in range(0, len(q), 500):
        dist = globals()["dist"](q[i:i + 500])
        idx = np.argsort(dist, axis=1, kind="stable")[:, :K]
        nb.append(idx); ds.append(np.take_along_axis(dist, idx, axis=1))
    return np.vstack(nb).astype(np.int32), np.vstack(ds).astype(np.float32)
nb, ds = truth(test)
lnb, _ = truth(learn)
with h5py.File(sys.argv[1], "w") as f:
    f.attrs.update(distance=metric, dimension=d, point_type="float")
    for k, v in dict(train=train, test=test, neighbors=nb, distances=ds, learn=learn, learn_neighbors=lnb,
                     avg_distances=ds.mean(axis=1)).items():
        f.create_dataset(k, data=v)
'''

# --test definitions, one config.yml per algorithm folder: short grids (minutes) and, for the cosine
# smoke, a config whose constructor does not exist.
SMOKE_DEFS = {}
SMOKE_DEFS["hms"] = """float:
  ip:
  - name: hms
    module: vibe.algorithms.hms
    constructor: HMS
    singularity_image: hms
    disabled: false
    ood: true
    base_args: ['@metric']
    run_groups:
      query-aware:
        args: {degree: [32], build_ef: [200], code_bits: [8], learn_queries: [2000]}
        query_args: {ef: [100, 200], rerank: [400], reservoir: [0, 400], patience: [0, 8]}
  cosine:
  - name: hms
    module: vibe.algorithms.hms
    constructor: HMS
    singularity_image: hms
    disabled: false
    ood: true
    base_args: ['@metric']
    run_groups:
      base:
        args: {degree: [16, 24, 32, 48, 64, 96], build_ef: [400], code_bits: [4], learn_queries: [2000]}
        query_args: {ef: [100, 200, 400], rerank: [100, 200]}
  - name: hms-broken
    module: vibe.algorithms.hms
    constructor: HMSMissing
    singularity_image: hms
    disabled: false
    base_args: ['@metric']
    run_groups:
      base:
        args: {degree: [16], build_ef: [100]}
        query_args: {ef: [100]}
"""
SMOKE_DEFS["roargraph"] = """float:
  any:
  - name: roargraph
    module: vibe.algorithms.roargraph
    constructor: RoarGraph
    singularity_image: roargraph
    disabled: false
    ood: true
    base_args: ['@metric']
    run_groups:
      roargraph:
        args: {M_sq: [100], M_pjbp: [32], L_pjpq: [500]}
        query_args: {L_pq: [100, 400]}
"""
SMOKE_DEFS["mlann"] = """float:
  any:
  - name: mlann-rf
    module: vibe.algorithms.mlann
    constructor: MLANN
    singularity_image: mlann
    disabled: false
    ood: true
    base_args: ['@metric']
    run_groups:
      small:
        args: {index_type: [RF], n_trees: [10], depth: [14]}
        query_args: {voting_threshold: [0.0025, 0.00025]}
"""


def now() -> float:
    return time.time()


def iso(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t or now()))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def usd(rec: dict) -> float:
    return rec.get("usd", 0.0)


def spend_records(ns: str, vol_root: Path | None) -> dict:
    """Container spend records: the live Dict, plus Volume mirrors for entries the Dict has dropped."""
    recs = {}
    if vol_root is not None:
        for p in (vol_root / "spend").glob("*.json"):
            r = read_json(p)
            if r:
                recs[p.stem] = r
    for key, r in SPEND.items():
        cid = key[len(ns) + 1:]
        if key.startswith(f"{ns}:") and (cid not in recs or r["last"] >= recs[cid]["last"]):
            recs[cid] = r
    return recs


def spend_total(ns: str, vol_root: Path | None) -> float:
    return sum(usd(r) for r in spend_records(ns, vol_root).values())


def cpu_info() -> dict:
    text = Path("/proc/cpuinfo").read_text()
    field = lambda key: next((ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.split(":")[0].strip() == key), "?")
    model = field("model name")
    if model in ("?", "unknown"):  # gVisor hides the brand string; family/model still identify the part
        ident = (field("vendor_id"), field("cpu family"), field("model"))
        model = KNOWN_CPUS.get(ident, "{} family {} model {}".format(*ident))
    flags = set(field("flags").split())
    avx512 = all(f in flags for f in ("avx512f", "avx512bw", "avx512vl", "avx512dq"))
    return {"cpu_model": model, "avx512": avx512, "nproc": os.cpu_count(),
            "flags_sha": hashlib.sha256(" ".join(sorted(flags)).encode()).hexdigest()[:12]}


def proc_table() -> dict[tuple[int, int], dict]:
    """Live processes keyed by (pid, start time), so a reused pid is a new process."""
    table, tick = {}, os.sysconf("SC_CLK_TCK")
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            f = stat.read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        # After the comm field: state ppid pgrp ... utime(11) stime(12) ... starttime(19)
        if f[0] == "Z":
            continue
        table[(int(stat.parent.name), int(f[19]))] = {"ppid": int(f[1]), "pgid": int(f[2]),
                                                      "cpu": (int(f[11]) + int(f[12])) / tick}
    return table


def descendants(table: dict[tuple[int, int], dict], root_pid: int) -> set[tuple[int, int]]:
    """Keys of root_pid and every process below it in `table`."""
    children: dict[int, list[tuple[int, int]]] = {}
    for key, p in table.items():
        children.setdefault(p["ppid"], []).append(key)
    out = {k for k in table if k[0] == root_pid}
    stack = [k[0] for k in out]
    while stack:
        for key in children.get(stack.pop(), []):
            if key not in out:
                out.add(key)
                stack.append(key[0])
    return out


def cpu_progress(prev: dict[tuple[int, int], float], cur: dict[tuple[int, int], float]) -> float:
    """CPU seconds the tracked processes gained since `prev`. Per process, so a process that exits does
    not take its CPU out of the total (VIBE runs one process per index config), and a new one counts from 0."""
    return sum(max(0.0, c - prev.get(k, 0.0)) for k, c in cur.items())


def kill_keys(keys: set[tuple[int, int]], grace_s: float = 30.0) -> list[int]:
    """SIGTERM the process groups of the live `keys`, then SIGKILL whatever is left. Returns killed pids."""
    alive = lambda: [k for k in keys if k in proc_table()]
    targets = alive()
    for pgid in {p["pgid"] for k, p in proc_table().items() if k in targets}:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass  # the group already exited; leftovers get SIGKILL below
    deadline = now() + grace_s
    while alive() and now() < deadline:
        time.sleep(1)
    for pid, _ in alive():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # exited between the check and the kill
    return [pid for pid, _ in targets]


@app.function(image=plan_image, volumes={"/vol": VOL}, timeout=900)
def plan(ns: str, replan: bool) -> dict:
    """Per dataset: size, sha256, memory, skip reason, and the published CPU leaders at each threshold."""
    import polars as pl

    VOL.reload()
    path = Path("/vol/vibe") / ns / "plan.json"
    if not replan and (old := read_json(path)) and all(d in old["datasets"] for d in DATASETS):
        return old
    tree = {e["path"]: e for e in json.load(urllib.request.urlopen(HF_TREE, timeout=60))}
    work = Path("/tmp/pub")
    work.mkdir(exist_ok=True)
    for name in ("results/summary.parquet", "algorithms_basics.csv"):
        urllib.request.urlretrieve(PUBLISHED.format(name), work / name.split("/")[-1])
    gpu = set(pl.read_csv(work / "algorithms_basics.csv").filter(pl.col("is_gpu"))["algorithm"].to_list())
    df = pl.read_parquet(work / "summary.parquet").filter((pl.col("k") == 100) & ~pl.col("algorithm").is_in(list(gpu)))
    out = {"created": iso(), "vibe_rev": VIBE_REV, "gpu_excluded": sorted(gpu), "datasets": {}}
    for ds in DATASETS:
        entry = tree.get(f"{ds}.hdf5")
        rec = {"skip": None, "leaders": {}}
        if entry is None:
            rec["skip"] = "not on Hugging Face"
        else:
            size_gb = entry["size"] / 1e9
            mem = math.ceil(3 * size_gb + 6)
            rec |= {"size_bytes": entry["size"], "sha256": entry["lfs"]["oid"], "mem_gib": max(8, mem)}
            if mem > MAX_MEM_GIB:
                rec["skip"] = f"needs ~{mem} GiB (3 x {size_gb:.1f} GB + 6) > {MAX_MEM_GIB} GiB cap"
        for t in THRESHOLDS:
            s = (df.filter((pl.col("dataset") == ds) & (pl.col("recall") >= float(t))).group_by("algorithm")
                 .agg(pl.col("qps").max()).sort("qps", descending=True))
            rec["leaders"][t] = [[a, round(q, 1)] for a, q in s.rows()]
        out["datasets"][ds] = rec
    write_json(path, out)
    VOL.commit()
    return out


def substitute(leaders: dict, avx512: bool, exclude: set[str]) -> str | None:
    """The best published competitor that runs here and is not excluded (installer failed, already chosen)."""
    for t in THRESHOLDS:
        for algo, _ in leaders.get(t) or []:
            spec = INSTALLABLE.get(algo)
            if spec and algo not in exclude and (avx512 or not spec[2]):
                return algo
    return None


def choose(leaders: dict, avx512: bool, named: list[str] = ()) -> tuple[list[str], dict]:
    """Union of the best published QPS at each threshold among competitors that run here, plus `named`."""
    chosen, why = [], {}
    for t in THRESHOLDS:
        ranked = leaders.get(t) or []
        if not ranked:
            continue
        leader, qps = ranked[0]
        pick, note = None, ""
        for algo, q in ranked:
            spec = INSTALLABLE.get(algo)
            if spec and (avx512 or not spec[2]):
                pick = algo
                if algo != leader:
                    reason = "needs AVX-512, absent here" if leader in INSTALLABLE else "no installer in the VIBE fork image"
                    note = f"substitute for {leader} ({qps} QPS; {reason}): {q} QPS"
                else:
                    note = f"leader at {q} QPS"
                break
        if pick is None:
            continue
        if pick not in chosen:
            chosen.append(pick)
        why.setdefault(pick, []).append(f"recall>={t}: {note}")
    for algo in named:
        if algo in chosen or algo not in INSTALLABLE:
            continue
        best = [(t, q) for t in THRESHOLDS for a, q in (leaders.get(t) or []) if a == algo]
        chosen.append(algo)
        why[algo] = ["named competitor arm" + (f"; published {best[0][1]} QPS at recall>={best[0][0]}" if best else "")]
    return chosen, why


class Runner:
    def __init__(self, ns: str, dataset: str, hms_rev: str, test: bool, cap: float, hard_cap: float, hms_only: bool = False):
        self.ns, self.ds, self.rev, self.test, self.cap, self.hard_cap = ns, dataset, hms_rev, test, cap, hard_cap
        # An --hms-only container takes its own lease so it can run next to a container still working
        # through the dataset's competitors; run_job refuses a job another live container is running.
        self.lease_key = f"{ns}:{dataset}" + (":hms-only" if hms_only else "")
        self.baseline = set(proc_table())  # processes that belong to the container, not to a job
        self.root = Path("/vol/vibe") / ns
        self.cid = os.environ.get("MODAL_TASK_ID", f"local-{os.getpid()}")
        self.started = now()
        self.phase_cores = CPU_LIMIT
        self.mem_gib = 8
        self.main_beat = now()  # the main thread's last sign of life; the lease lapses without it
        self.long_phase = False  # builds and the export may block the main thread for hours
        self.retry_failed = False
        self.released = threading.Event()  # set before the lease is dropped so the heartbeat stops renewing it
        self.hard_stop = threading.Event()
        self.cpu = cpu_info()
        self.defs = "/work/testdefs" if test else "vibe/algorithms"

    # ---- spend, lease and heartbeat (background thread) ----
    def _beat(self) -> None:
        rec = {"start": self.started, "last": self.started, "usd": 0.0, "mem_gib": self.mem_gib, "dataset": self.ds,
               "container": self.cid}
        while True:
            t = now()
            # Billing is the larger of request and use; builds may burst to the core limit.
            rec["usd"] += (t - rec["last"]) / 3600 * (self.phase_cores * USD_CORE_H + self.mem_gib * USD_GIB_H)
            rec["last"] = t
            try:
                SPEND[f"{self.ns}:{self.cid}"] = rec
                if not self.released.is_set() and (self.long_phase or t - self.main_beat < STALE_S):
                    LEASE[self.lease_key] = {"container": self.cid, "at": t}
                write_json(self.root / "spend" / f"{self.cid}.json", rec)
                if spend_total(self.ns, None) >= self.hard_cap:
                    self.hard_stop.set()
            except Exception as e:  # a transient Dict error must not kill the run
                print("heartbeat:", repr(e)[:200], flush=True)
            time.sleep(HEARTBEAT_S)

    def release(self) -> None:
        self.released.set()
        LEASE.pop(self.lease_key, None)

    def acquire(self) -> bool:
        # A container lost mid-job stops renewing its lease; a Modal retry waits for it to go stale rather
        # than giving up, so recovery needs no manual --resume. A live holder keeps renewing and we leave.
        deadline = now() + STALE_S + 2 * HEARTBEAT_S
        while (held := LEASE.get(self.lease_key)) and held["container"] != self.cid \
                and now() - held["at"] < STALE_S and now() < deadline:
            time.sleep(HEARTBEAT_S)
        if held and held["container"] != self.cid and now() - held["at"] < STALE_S:
            print(f"{self.ds}: container {held['container']} holds the lease ({now() - held['at']:.0f}s old); exiting", flush=True)
            return False
        LEASE[self.lease_key] = {"container": self.cid, "at": now()}
        time.sleep(3)
        return LEASE.get(self.lease_key, {}).get("container") == self.cid

    # ---- ledger ----
    def ledger_path(self, job: str) -> Path:
        return self.root / "ledger" / self.ds / f"{job}.json"

    def save(self, rec: dict) -> None:
        write_json(self.ledger_path(rec["job"]), rec)
        VOL.commit()

    def stop_reason(self, reason: str) -> None:
        write_json(self.root / "stop.json", {"at": iso(), "reason": reason, "container": self.cid, "dataset": self.ds})
        VOL.commit()

    # ---- setup ----
    def sh(self, script: str, env_name: str, log: Path, extra: dict | None = None, timeout: int = 7200) -> int:
        env = {**os.environ, "PATH": f"{ENV_BIN[env_name]}:/root/.cargo/bin:{os.environ['PATH']}",
               "HMS_REPO": HMS_REPO, "HMS_REV": self.rev, **(extra or {})}
        with log.open("a") as fh:
            return subprocess.run(["bash", "-c", INSTALL_SH + "\n" + script], env=env, stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode

    def fetch_dataset(self, meta: dict) -> str | None:
        """Download once into the Volume, verified against the LFS sha256. Returns an error or None."""
        data = self.root / "data"
        data.mkdir(parents=True, exist_ok=True)
        dst, side = data / f"{self.ds}.hdf5", data / f"{self.ds}.sha256"
        if self.test:
            if not dst.exists():
                tmp = Path("/work") / dst.name
                subprocess.run([f"{ENV_BIN['base']}/python", "-c", SMOKE_GEN, str(tmp), self.ds.rsplit("-", 1)[1]], check=True)
                shutil.copy(tmp, data / f".{dst.name}.part")
                os.replace(data / f".{dst.name}.part", dst)
                VOL.commit()
            return None
        if dst.exists() and side.exists() and side.read_text().strip() == meta["sha256"] and dst.stat().st_size == meta["size_bytes"]:
            return None
        err = None
        for attempt in range(3):
            part = data / f".{self.ds}.hdf5.part"
            h = hashlib.sha256()
            try:
                with urllib.request.urlopen(HF.format(self.ds), timeout=120) as r, part.open("wb") as out:
                    while chunk := r.read(16 << 20):
                        h.update(chunk)
                        out.write(chunk)
                if h.hexdigest() != meta["sha256"]:
                    raise ValueError(f"sha256 {h.hexdigest()} != {meta['sha256']}")
                os.replace(part, dst)
                side.write_text(meta["sha256"])
                VOL.commit()
                return None
            except Exception as e:
                err = f"attempt {attempt + 1}: {type(e).__name__}: {e}"[:500]
                part.unlink(missing_ok=True)
                time.sleep(30 * 2 ** attempt)
        return err

    def job_dir(self, job: str) -> Path:
        d = Path("/work/jobs") / job
        if not d.exists():
            d.mkdir(parents=True)
            for entry in Path("/opt/vibe").iterdir():
                if entry.name not in ("results", "data"):
                    (d / entry.name).symlink_to(entry)
            (d / "data").symlink_to("/work/data")
            (d / "results").mkdir()
            (d / "jobtool.py").write_text(JOBTOOL)
        return d

    def vol_results(self, job: str) -> Path:
        return self.root / "results" / self.ds / job

    def sync(self, job: str, mirror: bool = False) -> None:
        """Copy the job's local result tree to the Volume. Markers are listed before data files are copied,
        so a marker on the Volume never names a group missing from the copied file."""
        src, dst = self.job_dir(job) / "results", self.vol_results(job) / "results"
        markers = [p for p in src.rglob("*") if p.is_file() and p.parent.name.endswith(".complete")]
        for p in src.rglob("*.hdf5"):
            target = dst / p.relative_to(src)
            if not target.exists() or target.stat().st_size != p.stat().st_size or target.stat().st_mtime < p.stat().st_mtime:
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(f".{target.name}.tmp")
                shutil.copy2(p, tmp)
                os.replace(tmp, target)
        for p in markers:
            target = dst / p.relative_to(src)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
        if mirror and dst.exists():
            for p in sorted(dst.rglob("*"), reverse=True):
                if not (src / p.relative_to(dst)).exists():
                    p.unlink() if p.is_file() else shutil.rmtree(p)
        VOL.commit()

    def restore(self, job: str) -> None:
        src, dst = self.vol_results(job) / "results", self.job_dir(job) / "results"
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    def tool(self, job: str, algo: str, mode: str, env_name: str) -> dict:
        r = subprocess.run([f"{ENV_BIN[env_name]}/python", "jobtool.py", mode, self.ds, algo, self.defs],
                           cwd=self.job_dir(job), capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            return {"done": False, "markers": 0, "error": (r.stdout + r.stderr)[-2000:]}
        return json.loads(r.stdout.strip().splitlines()[-1])

    # ---- one job ----
    def run_job(self, job: str, algo: str, env_name: str, why) -> str:
        """Returns 'done', 'failed', 'stopped' (spend), 'handover' (container budget) or 'busy'."""
        rec = read_json(self.ledger_path(job)) or {
            "job": job, "dataset": self.ds, "algorithm": algo, "hms_rev": self.rev, "vibe_rev": VIBE_REV,
            "state": "pending", "attempts": 0, "failures": 0, "history": []}
        rec |= {"chosen_why": why, "hms_rev": self.rev}
        if rec["state"] == "done":
            print(f"{self.ds}/{job}: done, skipped", flush=True)
            return "done"
        if rec["state"] == "running" and rec.get("container_id") != self.cid and now() - rec.get("heartbeat", 0) < STALE_S:
            print(f"{self.ds}/{job}: running in live container {rec.get('container_id')}, skipped", flush=True)
            return "busy"
        if rec["state"] == "failed" and self.retry_failed:
            rec |= {"state": "pending", "failures": 0}
        if rec["state"] not in ("done", "failed") and not rec.get("orphan_reset") and any(
                str(h.get("outcome", "")).startswith("hung") for h in rec["history"]):
            # Runners before the per-process CPU check killed healthy jobs as hung and left VIBE's index
            # process running beside the next attempt, so groups from those attempts may carry contended
            # timings. Start such a job from zero, once.
            shutil.rmtree(self.vol_results(job), ignore_errors=True)
            shutil.rmtree(self.job_dir(job) / "results", ignore_errors=True)
            (self.job_dir(job) / "results").mkdir()
            rec["orphan_reset"] = {"at": iso(), "container": self.cid, "failures_before": rec["failures"],
                                   "reason": "attempts killed as hung by the pre-fix stall check; results discarded"}
            rec |= {"state": "pending", "failures": 0}
        if rec["state"] == "failed":
            print(f"{self.ds}/{job}: failed earlier ({rec.get('error_tail', '')[-200:]!r}), skipped", flush=True)
            return "failed"
        log_vol = self.root / "logs" / self.ds / f"{job}.log"
        log_vol.parent.mkdir(parents=True, exist_ok=True)
        if rec["state"] == "running":
            # The container that ran it is gone (we hold the lease). It counts as a failure only if it made no progress.
            age = now() - rec.get("heartbeat", 0)
            self.restore(job)
            markers = self.tool(job, algo, "check", env_name).get("markers", 0)
            progressed = markers > rec.get("markers_at_start", 0)
            rec["history"].append({"attempt": rec["attempts"], "container": rec.get("container_id"), "end": iso(),
                                   "outcome": f"interrupted (heartbeat {age:.0f}s old)", "progress": progressed})
            if not progressed:
                rec["failures"] += 1
        if rec["failures"] >= MAX_FAILURES:
            rec["state"] = "failed"
            self.save(rec)
            return "failed"
        self.restore(job)
        cleaned = self.tool(job, algo, "clean", env_name)
        self.sync(job, mirror=True)
        while True:
            if spend_total(self.ns, self.root) >= self.cap:
                rec |= {"state": "pending", "stop_reason": f"spend guard: estimate >= ${self.cap}"}
                self.save(rec)
                self.stop_reason(rec["stop_reason"])
                return "stopped"
            status = self.tool(job, algo, "check", env_name)
            if status.get("done"):
                rec |= {"state": "done", "end": iso(), "validation": status}
                self.save(rec)
                return "done"
            rec["attempts"] += 1
            rec |= {"state": "running", "start": iso(), "container_id": self.cid, "heartbeat": now(), "cleaned": cleaned,
                    "markers_at_start": status.get("markers", 0), "expected": status.get("expected"), **self.cpu,
                    "result_path": str(self.vol_results(job) / "results"), "log_path": str(log_vol)}
            self.save(rec)
            outcome = self.attempt(job, algo, env_name, rec, log_vol)
            self.sync(job)
            status = self.tool(job, algo, "check", env_name)
            rec["validation"] = status
            rec["history"].append({"attempt": rec["attempts"], "container": self.cid, "end": iso(), "outcome": outcome,
                                   "markers": status.get("markers")})
            if status.get("done"):
                rec |= {"state": "done", "end": iso()}
                self.save(rec)
                return "done"
            if outcome in ("handover", "stopped"):
                rec |= {"state": "pending", "stop_reason": outcome}
                self.save(rec)
                if outcome == "stopped":
                    self.stop_reason(f"hard cap: estimate >= ${self.hard_cap}")
                return outcome
            rec["failures"] += 1
            rec["error_tail"] = log_vol.read_text(errors="replace")[-4000:] + "\n" + json.dumps(status.get("problems", []))[:2000]
            if rec["failures"] >= MAX_FAILURES:
                rec |= {"state": "failed", "end": iso()}
                self.save(rec)
                return "failed"
            self.save(rec)
            self.tool(job, algo, "clean", env_name)
            time.sleep(60 * 2 ** (rec["failures"] - 1))

    def attempt(self, job: str, algo: str, env_name: str, rec: dict, log_vol: Path) -> str:
        jd = self.job_dir(job)
        cmd = [f"{ENV_BIN[env_name]}/python", "run.py", "--dataset", self.ds, "--algorithm", algo, "--local", "--parallelism", "1"]
        if self.test:
            cmd += ["--definitions", self.defs]
        shim = Path("/work/shim/sitecustomize.py")
        shim.parent.mkdir(parents=True, exist_ok=True)
        shim.write_text(NUMA_SHIM)
        env = {**os.environ, "PATH": f"{ENV_BIN[env_name]}:{os.environ['PATH']}", "PYTHONPATH": str(shim.parent), **THREAD_ENV}
        env.pop("OMP_PROC_BIND", None)
        self.phase_cores = CPU_REQ
        with log_vol.open("a") as fh:
            # Nothing from an earlier attempt may run beside this one.
            table = proc_table()
            if strays := kill_keys(set(table) - self.baseline - descendants(table, os.getpid())):
                fh.write(f"=== {iso()} killed processes left from an earlier attempt: {strays}\n")
            fh.write(f"\n=== {iso()} attempt {rec['attempts']} container {self.cid} cpu {self.cpu['cpu_model']}: {' '.join(cmd)}\n")
            fh.flush()
            proc = subprocess.Popen(cmd, cwd=jd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            # VIBE starts each index config in its own session, so the job is tracked as every process ever
            # seen below run.py; one that is orphaned stays tracked until it exits.
            tracked: dict[tuple[int, int], float] = {}
            total_cpu, last_change = 0.0, now()
            outcome = None
            while proc.poll() is None:
                self.main_beat = now()
                try:
                    proc.wait(timeout=HEARTBEAT_S)
                    break
                except subprocess.TimeoutExpired:
                    pass  # still running: fall through to the heartbeat
                table = proc_table()
                cur = {k: table[k]["cpu"] for k in descendants(table, proc.pid) | (tracked.keys() & table.keys())}
                gained = cpu_progress(tracked, cur)
                tracked = cur
                total_cpu += gained
                if gained > 1:
                    last_change = now()
                rec["heartbeat"] = now()
                rec["child_cpu_s"] = round(total_cpu, 1)
                self.sync(job)
                write_json(self.ledger_path(job), rec)
                write_json(self.root / "heartbeat" / f"{self.ds}.json",
                           {"job": job, "at": iso(), "container": self.cid, "cpu_s": total_cpu})
                VOL.commit()
                if now() - last_change > STALE_S:
                    outcome = f"hung: no CPU progress for {STALE_S // 60} min"
                elif self.hard_stop.is_set():
                    outcome = "stopped"
                elif now() - self.started > CONTAINER_BUDGET_S:
                    outcome = "handover"
                if outcome:
                    table = proc_table()
                    kill_keys(descendants(table, proc.pid) | (tracked.keys() & table.keys()), grace_s=60)
                    proc.wait()
                    break
            table = proc_table()
            if left := kill_keys(tracked.keys() & table.keys()):
                fh.write(f"=== {iso()} killed job processes still running after run.py: {left}\n")
            fh.write(f"=== {iso()} exit {proc.returncode} {outcome or ''}\n")
        return outcome or f"exit {proc.returncode}"

    def standing(self, jobs: list[tuple[str, str]]) -> None:
        """Export VIBE's summary for this dataset (current HMS rev plus competitors) and keep the best QPS per threshold."""
        m = Path("/work/merged")
        shutil.rmtree(m, ignore_errors=True)
        (m / "data").mkdir(parents=True)
        (m / "data" / f"{self.ds}.hdf5").symlink_to(f"/work/data/{self.ds}.hdf5")
        for job, algo in jobs:
            src = self.vol_results(job) / "results" / self.ds / "100" / algo
            if src.exists():
                shutil.copytree(src, m / "results" / self.ds / "100" / algo)
        log = self.root / "logs" / self.ds / "export.log"
        with log.open("w") as fh:
            r = subprocess.run([f"{ENV_BIN['plot']}/python", "/opt/vibe/export_results.py", "--results", str(m / "results"),
                                "--data", str(m / "data"), "--output", str(m / "out"), "--skip-pca-mahalanobis"],
                               cwd="/opt/vibe", stdout=fh, stderr=subprocess.STDOUT, timeout=4 * 3600)
        (self.root / "standings").mkdir(parents=True, exist_ok=True)
        if r.returncode != 0 or not (m / "out" / "summary.parquet").exists():
            print(f"{self.ds}: export_results failed, see {log}", flush=True)
            return
        best = subprocess.run([f"{ENV_BIN['plot']}/python", "-c", STANDING, str(m / "out" / "summary.parquet")],
                              capture_output=True, text=True, check=True).stdout
        shutil.copy(m / "out" / "summary.parquet", self.root / "standings" / f"{self.ds}@{self.rev[:12]}.summary.parquet")
        write_json(self.root / "standings" / f"{self.ds}@{self.rev[:12]}.json", {
            "dataset": self.ds, "hms_commit": self.rev, "vibe_commit": VIBE_REV, "k": 100, "generated": iso(),
            "container": self.cid, **self.cpu, "best_qps_at_recall": json.loads(best)})
        VOL.commit()


@app.function(image=image, volumes={"/vol": VOL}, cpu=(CPU_REQ, CPU_LIMIT), memory=8 * 1024, timeout=FN_TIMEOUT_S,
              retries=modal.Retries(max_retries=3, backoff_coefficient=2.0, initial_delay=60.0))
def run_dataset(dataset: str, meta: dict, hms_rev: str, hms_only: bool, test: bool, cap: float, hard_cap: float) -> dict:
    """All jobs of one dataset, HMS first. Memory is set per call through with_options."""
    ns = "test" if test else LIVE_NS
    VOL.reload()
    r = Runner(ns, dataset, hms_rev, test, cap, hard_cap, hms_only)
    r.mem_gib = meta.get("mem_gib", 8)
    r.retry_failed = bool(meta.get("retry_failed"))
    if not r.acquire():
        return {"dataset": dataset, "status": "busy"}
    threading.Thread(target=r._beat, daemon=True).start()
    for d in ("/work/build", "/work/data"):
        Path(d).mkdir(parents=True, exist_ok=True)
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    write_json(r.root / "cpu" / f"{r.cid}.json", {"dataset": dataset, "at": iso(), **r.cpu, "lscpu": lscpu})
    VOL.commit()
    print(f"{dataset}: container {r.cid} on {r.cpu['cpu_model']} avx512={r.cpu['avx512']} nproc={r.cpu['nproc']}", flush=True)
    if spend_total(ns, r.root) >= cap:
        r.stop_reason(f"spend guard: estimate >= ${cap} before {dataset} started")
        r.release()
        return {"dataset": dataset, "status": "stopped"}

    if err := r.fetch_dataset(meta):
        write_json(r.root / "ledger" / dataset / "_download.json", {"job": "_download", "dataset": dataset, "state": "failed",
                                                                    "error": err, "at": iso(), "container_id": r.cid})
        VOL.commit()
        r.release()
        return {"dataset": dataset, "status": "download failed", "error": err}
    write_json(r.root / "ledger" / dataset / "_download.json", {"job": "_download", "dataset": dataset, "state": "done",
                                                                "sha256": meta.get("sha256"), "at": iso()})
    (Path("/work/data") / f"{dataset}.hdf5").symlink_to(r.root / "data" / f"{dataset}.hdf5")
    if test:
        for folder, text in SMOKE_DEFS.items():
            Path("/work/testdefs", folder).mkdir(parents=True, exist_ok=True)
            Path("/work/testdefs", folder, "config.yml").write_text(text)

    hms_job = f"hms@{hms_rev[:12]}"
    jobs = [(hms_job, "hms", "base", ["--test: short HMS grid" if test else "HMS runs on every dataset"])]
    if test:
        if dataset == SMOKE:
            jobs.append(("hms-broken", "hms-broken", "base", ["--test: forced failure"]))
        else:
            jobs += [(a, a, INSTALLABLE[a][1], ["--test: short grid"]) for a in ("roargraph", "mlann-rf")]
    elif not hms_only:
        chosen, why = choose(meta["leaders"], r.cpu["avx512"], NAMED.get(dataset, []))
        jobs += [(a, a, INSTALLABLE[a][1], why[a]) for a in chosen]
    # Build what this container's pending jobs need, for this CPU. A competitor whose installer fails is
    # replaced by the next published leader that runs here (the ledger keeps the failed record).
    built, broken, results = set(), set(), {}
    i = 0
    while i < len(jobs):
        job, algo, env_name, why = jobs[i]
        i += 1
        rec = read_json(r.ledger_path(job)) or {}
        if rec.get("state") == "done" or (rec.get("state") == "failed" and not r.retry_failed):
            results[job] = rec["state"]
            continue
        installer = "hms" if algo.startswith("hms") else INSTALLABLE[algo][0]
        if installer in broken:
            results[job] = "install failed earlier"
            continue
        if installer not in built:
            r.phase_cores = CPU_LIMIT
            log = r.root / "logs" / dataset / f"install-{installer}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            r.long_phase = True
            rc = r.sh(installer, env_name, log)
            r.long_phase, r.main_beat = False, now()
            VOL.commit()
            if rc != 0:
                rec = read_json(r.ledger_path(job)) or {"job": job, "dataset": dataset, "algorithm": algo, "attempts": 0,
                                                        "failures": 0, "history": []}
                rec |= {"failures": rec.get("failures", 0) + 1, "hms_rev": hms_rev, "chosen_why": why, **r.cpu,
                        "error_tail": f"install {installer} exit {rc}\n" + log.read_text(errors="replace")[-4000:]}
                rec["state"] = "failed" if rec["failures"] >= MAX_FAILURES else "pending"
                r.save(rec)
                results[job] = f"install failed ({rec['state']})"
                broken.add(installer)
                if not test and not algo.startswith("hms"):
                    taken = {a for _, a, _, _ in jobs} | {a for a, spec in INSTALLABLE.items() if spec[0] in broken}
                    if sub := substitute(meta["leaders"], r.cpu["avx512"], taken):
                        jobs.append((sub, sub, INSTALLABLE[sub][1], [f"substitute for {algo}: its installer failed here"]))
                continue
            built.add(installer)
        out = r.run_job(job, algo, env_name, why)
        results[job] = out
        if out in ("stopped", "handover"):
            break
    if any(v == "handover" for v in results.values()):
        r.release()
        call = run_dataset.with_options(memory=r.mem_gib * 1024).spawn(dataset, meta, hms_rev, hms_only, test, cap, hard_cap)
        return {"dataset": dataset, "container": r.cid, "jobs": results, "handed_over_to": call.object_id}
    if not test or results.get(hms_job) == "done":
        r.long_phase = True
        r.standing([(j, a) for j, a, _, _ in jobs if (read_json(r.ledger_path(j)) or {}).get("state") == "done"])
        r.long_phase = False
    r.release()
    return {"dataset": dataset, "container": r.cid, **r.cpu, "jobs": results,
            "spend_usd": round(spend_total(ns, r.root), 2)}


def status_report(ns: str) -> None:
    root = f"/vibe/{ns}"
    counts, rows = {}, []
    try:
        entries = list(VOL.iterdir(f"{root}/ledger", recursive=True))
    except Exception:
        entries = []
    for e in entries:
        if not e.path.endswith(".json"):
            continue
        rec = json.loads(b"".join(VOL.read_file(e.path)))
        counts[rec["state"]] = counts.get(rec["state"], 0) + 1
        rows.append(rec)
    spend, recs = 0.0, {}
    for key, r in SPEND.items():
        if key.startswith(f"{ns}:"):
            recs[key] = r
            spend += usd(r)
    live = [r for r in recs.values() if now() - r["last"] < 3 * HEARTBEAT_S]
    print(f"namespace {ns}: {counts} | estimated spend ${spend:.2f} over {len(recs)} containers, {len(live)} live")
    for rec in sorted(rows, key=lambda x: (x["dataset"], x["job"])):
        extra = rec.get("stop_reason") or (rec.get("error_tail") or rec.get("error") or "")[-120:].replace("\n", " ")
        print(f"  {rec['dataset']:38s} {rec['job']:22s} {rec['state']:8s} attempts={rec.get('attempts', 0)} "
              f"failures={rec.get('failures', 0)} cpu={rec.get('cpu_model', '-')[:32]} {extra}")
    try:
        print("stop:", b"".join(VOL.read_file(f"{root}/stop.json")).decode())
    except Exception:  # no stop.json means nothing has stopped the run
        print("stop: none recorded")


def snapshot_report(ns: str, out: Path, hms_rev: str) -> None:
    root = f"/vibe/{ns}"
    manifest = read_json(Path(__file__).resolve().parents[2] /
                         "benchmarks/results/vibe_live_s_20261023/manifest.json", {})

    def remote_json(path: str) -> dict:
        return json.loads(b"".join(VOL.read_file(path)))

    plan_data = remote_json(f"{root}/plan.json")
    ledgers = {}
    for entry in VOL.iterdir(f"{root}/ledger", recursive=True):
        if entry.path.endswith(".json"):
            rec = remote_json(entry.path)
            ledgers.setdefault(rec["dataset"], {})[rec["job"]] = rec
    standings = {}
    try:
        for entry in VOL.iterdir(f"{root}/standings", recursive=False):
            if entry.path.endswith(f"@{hms_rev[:12]}.json"):
                rec = remote_json(entry.path)
                if rec.get("hms_commit") == hms_rev:
                    standings[rec["dataset"]] = rec
    except (FileNotFoundError, modal.exception.NotFoundError):
        pass
    spend_by_container = {}
    try:
        for entry in VOL.iterdir(f"{root}/spend", recursive=False):
            if entry.path.endswith(".json"):
                spend_by_container[Path(entry.path).stem] = remote_json(entry.path)
    except (FileNotFoundError, modal.exception.NotFoundError):
        pass
    for key, rec in SPEND.items():
        if key.startswith(f"{ns}:"):
            cid = key[len(ns) + 1:]
            if cid not in spend_by_container or rec["last"] >= spend_by_container[cid]["last"]:
                spend_by_container[cid] = rec
    spent = {}
    for rec in spend_by_container.values():
        spent[rec["dataset"]] = spent.get(rec["dataset"], 0.0) + usd(rec)
    rows = []
    for ds, meta in plan_data["datasets"].items():
        records = ledgers.get(ds, {})
        hms = records.get(f"hms@{hms_rev[:12]}", {})
        download = records.get("_download", {})
        standing = standings.get(ds, {})
        measured = standing.get("best_qps_at_recall", {})
        competitors = {}
        for algo, record in records.items():
            if algo in ("_download", f"hms@{hms_rev[:12]}"):
                continue
            same_container = bool(hms.get("container_id")) and record.get("container_id") == hms.get("container_id")
            competitors[algo] = {
                "state": record.get("state"), "cpu_model": record.get("cpu_model"),
                "container_id": record.get("container_id"), "same_container_as_hms": same_container,
                "fair_comparison": bool(hms.get("state") == "done" and record.get("state") == "done" and same_container),
                "metrics": measured.get(algo), "error": record.get("error_tail") or record.get("error"),
            }
        comparison_status = ("comparable" if any(r["fair_comparison"] for r in competitors.values()) else
                             "noncomparable" if hms.get("state") == "done" and any(
                                 r["state"] == "done" for r in competitors.values()) else "pending")
        rows.append({
            "dataset": ds, "dataset_sha256": meta.get("sha256"), "dataset_bytes": meta.get("size_bytes"),
            "requested_memory_gib": meta.get("mem_gib"), "download_state": download.get("state", "pending"),
            "state": hms.get("state", "failed" if download.get("state") == "failed" else "pending"),
            "attempts": hms.get("attempts", 0),
            "failures": hms.get("failures", 0), "hms_commit": hms_rev,
            "vibe_fork_commit": VIBE_REV, "source_sha256": manifest.get("source_sha256"),
            "query_threads": 1, "cpu_model": hms.get("cpu_model"), "avx512": hms.get("avx512"),
            "nproc": hms.get("nproc"), "container_id": hms.get("container_id"),
            "hardware_cpu_record": (f"hms-bench-run:{root}/cpu/{hms['container_id']}.json"
                                    if hms.get("container_id") else None),
            "validation": hms.get("validation"), "metrics": measured.get("hms"),
            "peak_rss_bytes": None, "peak_rss_note": "VIBE exporter records index-size delta, not peak process RSS",
            "competitors": competitors, "comparison_status": comparison_status,
            "published_leaders_diagnostic": meta.get("leaders"),
            "estimated_cost_usd": round(spent.get(ds, 0.0), 4),
            "ledger_prefix": f"hms-bench-run:{root}/ledger/{ds}",
            "results_prefix": f"hms-bench-run:{root}/results/{ds}",
            "hms_log": hms.get("log_path"),
            "error": hms.get("error_tail") or download.get("error"),
        })
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, {"generated": iso(), "namespace": ns, "hms_commit": hms_rev,
                     "vibe_commit": VIBE_REV, "estimated_cost_usd": round(sum(spent.values()), 4),
                     "rows": rows})
    print(f"wrote {len(rows)} dataset rows to {out}")


@app.local_entrypoint()
def main(resume: bool = False, status: bool = False, hms_rev: str = "", hms_only: bool = False, retry_failed: bool = False,
         datasets: str = "", test: bool = False, replan: bool = False, cap_usd: float = 140.0, hard_cap_usd: float = 150.0,
         snapshot: str = ""):
    ns = "test" if test else LIVE_NS
    if snapshot:
        if not hms_rev:
            raise ValueError("--snapshot requires --hms-rev")
        snapshot_report(ns, Path(snapshot), hms_rev)
        return
    if status:
        status_report(ns)
        return
    if not resume:
        print("pass --resume to launch or resume, --status for the ledger summary")
        return
    if not hms_rev:
        out = subprocess.run(["git", "ls-remote", HMS_REPO, "refs/heads/main"], capture_output=True, text=True, check=True).stdout
        hms_rev = out.split()[0]
    if test:
        targets = datasets.split(",") if datasets else [SMOKE]
        meta_all = {t: {"mem_gib": 8, "leaders": {}} for t in targets}
    else:
        p = plan.remote(ns, replan)
        meta_all = p["datasets"]
        wanted = datasets.split(",") if datasets else DATASETS
        targets = [d for d in wanted if not meta_all[d]["skip"]]
        for d in wanted:
            if meta_all[d]["skip"]:
                print(f"skip {d}: {meta_all[d]['skip']}")
    spent = spend_total(ns, None)
    if spent >= cap_usd:
        print(f"spend estimate ${spent:.2f} >= ${cap_usd}; nothing launched")
        return
    print(f"hms_rev {hms_rev}; launching {len(targets)} datasets; spend so far ${spent:.2f}")
    calls = []
    for d in targets:
        meta = meta_all[d] | ({"retry_failed": True} if retry_failed else {})
        calls.append((d, run_dataset.with_options(memory=meta["mem_gib"] * 1024).spawn(d, meta, hms_rev, hms_only, test, cap_usd, hard_cap_usd)))
    for d, call in calls:
        try:
            print(json.dumps(call.get(), default=str))
        except Exception as e:
            print(f"{d}: {type(e).__name__}: {str(e)[-400:]}")
