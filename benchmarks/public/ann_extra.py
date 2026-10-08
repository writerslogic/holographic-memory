"""Competitor ANN systems beyond FAISS and hnswlib, for `evaluate.py ann --extra <system>...`.

Every system is built with all cores and timed through evaluate.py's single-threaded,
one-query-at-a-time loop (`timed_queries`), at the parameters its authors recommend (the
ann-benchmarks configuration they maintain, or their own benchmark settings). A system that does
not build on this machine is recorded with the reason instead of a number.

Extra Python packages are not in evaluate.py's inline dependencies; pass them on the command
line: `uv run --with ngt --with 'rabitqlib>=0.5.2' --script evaluate.py ann ... --extra ...`.

evaluate.py runs each system in its own process (`python ann_extra.py <system> <dataset> ...`):
faiss-cpu and rabitqlib each bundle an OpenMP runtime, and a rabitqlib build segfaults once
FAISS has been imported into the same process.
"""

import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

K = 10
EF_SWEEP = (16, 32, 64, 128, 256, 512, 768, 1024)
NPROBE_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
# ann-benchmarks onng_ngt query_args: epsilon (search range factor) with the index's edge size.
NGT_EPSILONS = (0.6, 0.9, 1.0, 1.02, 1.03, 1.04, 1.05, 1.07, 1.1, 1.2)
LUCENE_VERSION = "9.12.3"
LUCENE_JAR_SHA256 = "b64a3f8098a7572034fb30085cdee01b34ec81fb0e5a31b471536af58dc6c01b"
LUCENE_URL = f"https://repo1.maven.org/maven2/org/apache/lucene/lucene-core/{LUCENE_VERSION}/lucene-core-{LUCENE_VERSION}.jar"

SYSTEMS = ("symphonyqg", "rabitq", "ngt", "lucene_hnsw", "glass", "diskann", "scann")

# Systems that cannot run on an Apple M4 (arm64 macOS), with the evidence (2026-10-07).
NOT_ON_M4 = {
    "glass": "PyPI `pyglass` 0.1.2 is an unrelated Python-2 package; zilliztech/pyglass 2.1.0 built from source "
             "fails under Apple clang on arm64 (its setup.py passes -march=native, -fopenmp and -lrt). x86 only here.",
    "diskann": "diskannpy 0.7.0 ships cp39-cp311 wheels for Linux and Windows x86-64 only, and DiskANN's own build "
               "needs MKL and does not support macOS.",
    "scann": "scann 1.4.2 ships manylinux wheels only (x86-64 and aarch64 Linux); no macOS build.",
}


def pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


class Ctx:
    """What a runner needs from evaluate.py: data, truth, the timing loop and the recall."""

    def __init__(self, name, meta, train, test, truth, repeats, timed, recall, workdir, max_load, max_wait_secs):
        self.name, self.meta, self.train, self.test, self.truth = name, meta, train, test, truth
        self.repeats, self.timed, self.recall, self.workdir = repeats, timed, recall, Path(workdir)
        self.max_load, self.max_wait_secs = max_load, max_wait_secs
        self.dim = meta["dim"]
        self.angular = meta["metric"] == "angular"
        self.workdir.mkdir(parents=True, exist_ok=True)

    def row(self, library, index, params, found, qps, runs, build_secs, index_bytes, ver, series, **extra):
        return {"library": library, "index": index, "series": series, "params": params,
                "recall_at_10": self.recall(found, self.truth), "qps_single_thread": qps, "build_secs": build_secs,
                "index_bytes": int(index_bytes), "version": ver, **runs, **extra}


def saved_bytes(index, path: Path) -> int:
    index.save(str(path))
    size = path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return size


def run_symphonyqg(c: Ctx) -> tuple[list, dict]:
    """SymphonyQG in the authors' library (rabitqlib.SymqgIndex, NEON on arm64): degree 32,
    ef_construction 200, PiPNN initialisation (the library's documented defaults), raw-vector
    refinement (quantization_bits 0, the paper's setting) and 4-bit refinement (compact)."""
    import rabitqlib

    ver = pkg_version("rabitqlib")
    metric = "ip" if c.angular else "l2"
    rows, series = [], {}
    for qb in (0, 4):
        index = rabitqlib.SymqgIndex(dim=c.dim, max_degree=32, metric=metric, quantization_bits=qb)
        t = time.perf_counter()
        index.build(c.train, ef_construction=200, num_threads=os.cpu_count(), init="pipnn")
        build = time.perf_counter() - t
        size = saved_bytes(index, c.workdir / f"symqg_qb{qb}.bin")
        label = f"SymphonyQG (rabitqlib, qb={qb})"
        for ef in EF_SWEEP:
            found, qps, runs = c.timed(lambda q, ef=ef: index.search(q, K, max(ef, K), 1)[0][0], c.test, c.repeats)
            r = c.row("rabitqlib", "SymphonyQG", {"max_degree": 32, "ef_construction": 200, "init": "pipnn",
                                                  "quantization_bits": qb, "ef": ef}, found, qps, runs, build, size, ver,
                      label)
            rows.append(r)
            series.setdefault(label, []).append((r["recall_at_10"], qps))
    return rows, series


def run_rabitq(c: Ctx) -> tuple[list, dict]:
    """RaBitQ in the authors' library: IVF with 4,096 clusters (their benchmark setting; 1,024
    below 500k vectors), extended codes 1+4 and 1+8 bits (nbits 5 and 9), nprobe swept; and
    HNSW RaBitQ 1+4 (M 16, ef_construction 200, 16 residual centroids, seed 42)."""
    import rabitqlib

    ver = pkg_version("rabitqlib")
    metric = "ip" if c.angular else "l2"
    n = len(c.train)
    rows, series = [], {}
    nlist = 4096 if n >= 500_000 else 1024
    t = time.perf_counter()
    km = rabitqlib.QGKMeans(c.dim, nlist, niter=10, num_threads=os.cpu_count())
    km.train(c.train)
    cluster_secs = time.perf_counter() - t
    for nbits in (5, 9):
        index = rabitqlib.IvfIndex(dim=c.dim, max_elements=n, num_clusters=nlist, nbits=nbits, metric=metric)
        t = time.perf_counter()
        index.build(c.train, km.centroids, km.assignments, num_threads=os.cpu_count())
        build = cluster_secs + time.perf_counter() - t
        size = saved_bytes(index, c.workdir / f"rabitq_ivf_{nbits}.bin")
        label = f"RaBitQ IVF 1+{nbits - 1}"
        for nprobe in NPROBE_SWEEP:
            if nprobe > nlist:
                break
            found, qps, runs = c.timed(lambda q, p=nprobe: index.search(q, K, p, None, 1)[0][0], c.test, c.repeats)
            r = c.row("rabitqlib", "IVF-RaBitQ", {"num_clusters": nlist, "nbits": nbits, "nprobe": nprobe},
                      found, qps, runs, build, size, ver, label)
            rows.append(r)
            series.setdefault(label, []).append((r["recall_at_10"], qps))
    t = time.perf_counter()
    km16 = rabitqlib.RaBitQKMeans(c.dim, 16, niter=10, num_threads=os.cpu_count())
    km16.train(c.train)
    cluster_secs = time.perf_counter() - t
    index = rabitqlib.HnswIndex(dim=c.dim, max_elements=n, M=16, ef_construction=200, nbits=5, metric=metric,
                                random_seed=42)
    t = time.perf_counter()
    index.build(c.train, km16.centroids, km16.assignments, num_threads=os.cpu_count())
    build = cluster_secs + time.perf_counter() - t
    size = saved_bytes(index, c.workdir / "rabitq_hnsw_5.bin")
    for ef in EF_SWEEP:
        found, qps, runs = c.timed(lambda q, ef=ef: index.search(q, K, max(ef, K), 1)[0][0], c.test, c.repeats)
        r = c.row("rabitqlib", "HNSW-RaBitQ", {"M": 16, "ef_construction": 200, "nbits": 5, "ef": ef},
                  found, qps, runs, build, size, ver, "RaBitQ HNSW 1+4")
        rows.append(r)
        series.setdefault("RaBitQ HNSW 1+4", []).append((r["recall_at_10"], qps))
    return rows, series


def run_ngt(c: Ctx) -> tuple[list, dict]:
    """NGT-onng through ngtpy and the `ngt` command (Homebrew), with the ann-benchmarks onng_ngt
    parameters: ANNG edge 100, epsilon 0.1, then reconstruct-graph outdegree 10 / indegree 120,
    search epsilon swept. NGT-qg needs the `qbg` command, which the macOS build lacks."""
    import ngtpy

    ngt = shutil.which("ngt")
    if ngt is None:
        raise RuntimeError("the `ngt` command is not installed (brew install ngt)")
    ver = pkg_version("ngt")
    work = c.workdir / "ngt"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    anng, onng = work / "anng", work / "onng"
    # NGT skips an all-zero row as invalid under normalized cosine, which would shift every later
    # id; such rows get a unit vector along axis 0 instead (nytimes has 239 of them).
    train = c.train
    zero = np.flatnonzero(np.linalg.norm(train, axis=1) == 0)
    if len(zero):
        train = train.copy()
        train[zero] = 0
        train[zero, 0] = 1
    t = time.perf_counter()
    subprocess.run([ngt, "create", "-it", "-p8", "-b500", "-ga", "-of", "-DE" if c.angular else "-D2",
                    f"-d{c.dim}", "-E100", "-S0", "-e0.1", "-P0", "-B30", "-T4", str(anng)], check=True)
    index = ngtpy.Index(path=str(anng))
    index.batch_insert(train, num_threads=os.cpu_count(), debug=False)
    index.save()
    index.close()
    subprocess.run([ngt, "reconstruct-graph", "-mS", "-o 10", "-i 120", str(anng), str(onng)], check=True)
    build = time.perf_counter() - t
    size = sum(p.stat().st_size for p in onng.rglob("*") if p.is_file())
    index = ngtpy.Index(str(onng), read_only=True)
    rows, series = [], {}
    for eps in NGT_EPSILONS:
        index.set(epsilon=eps - 1.0, edge_size=-2)
        found, qps, runs = c.timed(lambda q: np.asarray(index.search(q[0], K, with_distance=False)), c.test, c.repeats)
        r = c.row("ngt", "ONNG", {"edge": 100, "outdegree": 10, "indegree": 120, "build_epsilon": 0.1,
                                  "epsilon": eps, "edge_size": -2, "zero_rows_substituted": int(len(zero))},
                  found, qps, runs, build, size, ver, "NGT-onng")
        rows.append(r)
        series.setdefault("NGT-onng", []).append((r["recall_at_10"], qps))
    index.close()
    shutil.rmtree(anng, ignore_errors=True)
    return rows, series


def lucene_jar() -> Path:
    jar = Path.home() / ".cache" / "hms-bench" / "jars" / f"lucene-core-{LUCENE_VERSION}.jar"
    if not jar.exists():
        jar.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LUCENE_URL, jar)
    digest = hashlib.sha256(jar.read_bytes()).hexdigest()
    if digest != LUCENE_JAR_SHA256:
        raise RuntimeError(f"{jar}: sha256 {digest} != {LUCENE_JAR_SHA256}")
    return jar


def run_lucene_hnsw(c: Ctx) -> tuple[list, dict]:
    """Lucene HNSW (lucene-core, pinned) through lucene_hnsw/LuceneHnsw.java: M 16 and 32 with
    beamWidth 500 (the efConstruction of the ann-benchmarks luceneknn configuration), built with
    all cores, searched in a single-threaded Java loop with the same load gate."""
    java, javac = shutil.which("java"), shutil.which("javac")
    if not (java and javac):
        raise RuntimeError("java and javac are required")
    jar = lucene_jar()
    classes = c.workdir / "lucene_classes"
    classes.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).with_name("lucene_hnsw") / "LuceneHnsw.java"
    subprocess.run([javac, "-cp", str(jar), "-d", str(classes), str(src)], check=True)
    java_version = subprocess.run([java, "-version"], capture_output=True, text=True).stderr.splitlines()[0]
    ver = f"lucene-core {LUCENE_VERSION} (sha256 {LUCENE_JAR_SHA256[:12]}), {java_version}"
    data = c.meta["_dir"]
    rows, series = [], {}
    for m in (16, 32):
        out = subprocess.run(
            [java, "-Xmx6g", "-cp", f"{jar}:{classes}", "LuceneHnsw", str(data / "train.f32"), str(data / "test.f32"),
             str(c.dim), str(m), "500", ",".join(map(str, EF_SWEEP)), str(K), str(c.repeats), str(c.workdir / "lucene_ids"),
             str(len(c.test)), "angular" if c.angular else "euclidean", str(c.max_load), str(c.max_wait_secs),
             str(os.cpu_count())],
            check=True, capture_output=True, text=True)
        rep = json.loads(out.stdout)
        for e in rep["sweep"]:
            found = np.fromfile(e["ids_file"], dtype="<i4").reshape(-1, K)
            qps = float(np.median(e["qps_runs"]))
            runs = {"qps_runs": e["qps_runs"], "load_1m_before_runs": e["load_1m_before_runs"], "load_gate_met": e["load_gate_met"]}
            r = c.row("lucene", "HNSW", {"M": m, "beamWidth": 500, "ef": e["ef"]}, found, qps, runs, rep["build_secs"],
                      rep["index_bytes"], ver, f"Lucene HNSW M={m}", builder=rep["builder"],
                      graph_ram_bytes=rep["graph_ram_bytes"])
            rows.append(r)
            series.setdefault(f"Lucene HNSW M={m}", []).append((r["recall_at_10"], qps))
    return rows, series


RUNNERS = {"symphonyqg": run_symphonyqg, "rabitq": run_rabitq, "ngt": run_ngt, "lucene_hnsw": run_lucene_hnsw}


def run(system: str, c: Ctx) -> dict:
    """Runs one extra system; a failure to import, build or run is recorded, not raised."""
    if system in NOT_ON_M4:
        return {"system": system, "status": "not run on M4", "reason": NOT_ON_M4[system], "rows": [], "series": {}}
    t = time.perf_counter()
    try:
        rows, series = RUNNERS[system](c)
    except Exception as e:  # noqa: BLE001 - every failure mode is a result to record
        return {"system": system, "status": "failed", "reason": f"{type(e).__name__}: {e}"[:2000],
                "secs": time.perf_counter() - t, "rows": [], "series": {}}
    return {"system": system, "status": "ran", "secs": time.perf_counter() - t, "rows": rows, "series": series}


def main() -> None:
    """One system in one process; evaluate.py calls this and merges the JSON it writes."""
    import argparse

    import evaluate

    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=SYSTEMS)
    ap.add_argument("dataset")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--queries", type=int)
    ap.add_argument("--train-rows", type=int, help="smoke test only: the first N train rows (recall is meaningless)")
    ap.add_argument("--max-load", type=float, default=evaluate.MAX_LOAD)
    ap.add_argument("--max-wait-secs", type=int, default=evaluate.MAX_WAIT_SECS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    evaluate.MAX_LOAD, evaluate.MAX_WAIT_SECS = args.max_load, args.max_wait_secs
    d = evaluate.data_dir(args.dataset)
    meta = json.loads((d / "meta.json").read_text())
    train, test = evaluate.load_f32(d / "train.f32", meta["dim"]), evaluate.load_f32(d / "test.f32", meta["dim"])
    truth = np.fromfile(d / "neighbors.i32", dtype="<i4").reshape(len(test), -1)
    if args.queries:
        test, truth = test[:args.queries], truth[:args.queries]
    if args.train_rows:
        train = train[:args.train_rows]
    if meta["metric"] == "angular":
        train, test = evaluate.normalize(train), evaluate.normalize(test)
    ctx = Ctx(args.dataset, {**meta, "_dir": d}, train, test, truth, args.repeats, evaluate.timed_queries,
              evaluate.recall, d / "extra_work", args.max_load, args.max_wait_secs)
    res = run(args.system, ctx)
    Path(args.out).write_text(json.dumps(res) + "\n")
    print(f"{args.system}: {res['status']} {res.get('reason', '')}", flush=True)


if __name__ == "__main__":
    main()
