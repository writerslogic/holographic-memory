# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "faiss-cpu>=1.8", "hnswlib>=0.8", "pytrec-eval-terrier>=0.5.6"]
# ///
"""Score public-bench output and run comparison libraries on the same data.

  evaluate.py ann  <dataset> <hms-result.json>...   FAISS (flat, HNSW) and hnswlib sweeps
  evaluate.py beir <dataset> <hms-runs.json>        nDCG@10 / recall@100 with pytrec_eval

Writes benchmarks/results/public_<dataset>.json.
"""

import argparse
import json
import os
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

K = 10
RESULTS = Path(__file__).resolve().parents[1] / "results"


def data_dir(name: str) -> Path:
    return Path(os.environ.get("HMS_BENCH_DATA", Path.home() / ".cache" / "hms-bench")) / name


def environment() -> dict:
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    load = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout.strip()
    return {
        "cpu": cpu,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "load_average": load,
        "versions": {p: version(p) for p in ["faiss-cpu", "hnswlib", "numpy", "pytrec-eval-terrier"]},
    }


def load_f32(path: Path, dim: int) -> np.ndarray:
    return np.fromfile(path, dtype="<f4").reshape(-1, dim)


def normalize(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return (a / np.where(n == 0, 1, n)).astype(np.float32)


def recall(found: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean([len(set(f[:K]) & set(t[:K])) / K for f, t in zip(found, truth)]))


def timed_queries(search, queries: np.ndarray) -> tuple[np.ndarray, float]:
    """Single-threaded, one query at a time, as ann-benchmarks does."""
    out = np.empty((len(queries), K), dtype=np.int64)
    t = time.perf_counter()
    for i, q in enumerate(queries):
        out[i] = search(q[None, :])
    return out, len(queries) / (time.perf_counter() - t)


def run_ann(name: str, hms_files: list[str]) -> dict:
    import faiss
    import hnswlib

    d = data_dir(name)
    meta = json.loads((d / "meta.json").read_text())
    dim = meta["dim"]
    angular = meta["metric"] == "angular"
    train, test = load_f32(d / "train.f32", dim), load_f32(d / "test.f32", dim)
    truth = np.fromfile(d / "neighbors.i32", dtype="<i4").reshape(len(test), -1)
    if angular:
        train, test = normalize(train), normalize(test)
    space = "ip" if angular else "l2"
    rows = []

    faiss.omp_set_num_threads(os.cpu_count())
    t = time.perf_counter()
    flat = faiss.IndexFlatIP(dim) if angular else faiss.IndexFlatL2(dim)
    flat.add(train)
    build = time.perf_counter() - t
    faiss.omp_set_num_threads(1)
    found, qps = timed_queries(lambda q: flat.search(q, K)[1][0], test)
    rows.append({"library": "faiss", "index": "IndexFlat (exact)", "params": {}, "recall_at_10": recall(found, truth),
                 "qps_single_thread": qps, "build_secs": build, "index_bytes": int(train.nbytes)})

    for m in (16, 32):
        faiss.omp_set_num_threads(os.cpu_count())
        t = time.perf_counter()
        hnsw = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT if angular else faiss.METRIC_L2)
        hnsw.hnsw.efConstruction = 200
        hnsw.add(train)
        build = time.perf_counter() - t
        size = len(faiss.serialize_index(hnsw))
        faiss.omp_set_num_threads(1)
        for ef in (16, 32, 64, 128, 256, 512):
            hnsw.hnsw.efSearch = ef
            found, qps = timed_queries(lambda q: hnsw.search(q, K)[1][0], test)
            rows.append({"library": "faiss", "index": "IndexHNSWFlat", "params": {"M": m, "efConstruction": 200, "efSearch": ef},
                         "recall_at_10": recall(found, truth), "qps_single_thread": qps, "build_secs": build, "index_bytes": size})

    t = time.perf_counter()
    h = hnswlib.Index(space=space, dim=dim)
    h.init_index(max_elements=len(train), ef_construction=200, M=16)
    h.add_items(train, num_threads=os.cpu_count())
    build = time.perf_counter() - t
    h.set_num_threads(1)
    for ef in (16, 32, 64, 128, 256, 512):
        h.set_ef(max(ef, K))
        found, qps = timed_queries(lambda q: h.knn_query(q, k=K)[0][0], test)
        rows.append({"library": "hnswlib", "index": "HNSW", "params": {"M": 16, "ef_construction": 200, "ef": ef},
                     "recall_at_10": recall(found, truth), "qps_single_thread": qps, "build_secs": build,
                     "index_bytes": int(h.element_count * (dim * 4 + 16 * 2 * 4 + 8))})

    hms = [json.loads(Path(f).read_text()) for f in hms_files]
    return {"dataset": meta, "environment": environment(), "competitors": rows,
            "hms": [{k: v for k, v in r.items() if k != "dataset"} for r in hms],
            "notes": ["Queries run single-threaded one at a time for every system; builds use all cores.",
                      "HMS takes dense input through its sparse encoder; its latency includes encoding (query_total).",
                      "hnswlib index_bytes is an estimate (vectors + level-0 links); FAISS sizes are serialized bytes."]}


def run_beir(name: str, hms_file: str) -> dict:
    import pytrec_eval

    d = data_dir(name)
    meta = json.loads((d / "meta.json").read_text())
    qrels: dict = {}
    for line in (d / "qrels.tsv").read_text().splitlines():
        qid, did, rel = line.split("\t")
        qrels.setdefault(qid, {})[did] = int(rel)
    hms = json.loads(Path(hms_file).read_text())

    # Independent exact-cosine reference: HMS's dense run must reproduce it.
    corpus_ids = [json.loads(line)["id"] for line in (d / "corpus.jsonl").read_text().splitlines()]
    query_ids = [json.loads(line)["id"] for line in (d / "queries.jsonl").read_text().splitlines()]
    c = load_f32(d / "corpus.f32", meta["dim"])
    q = load_f32(d / "queries.f32", meta["dim"])
    top = np.argsort(-(q @ c.T), axis=1)[:, :100]
    runs = dict(hms["runs"])
    runs["numpy_dense_exact_reference"] = {qid: [corpus_ids[i] for i in row] for qid, row in zip(query_ids, top)}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.100"})
    scores = {}
    for run_name, ranked in runs.items():
        run = {qid: {did: float(len(docs) - r) for r, did in enumerate(docs)} for qid, docs in ranked.items()}
        per_query = evaluator.evaluate(run)
        scores[run_name] = {
            "ndcg_at_10": float(np.mean([v["ndcg_cut_10"] for v in per_query.values()])),
            "recall_at_100": float(np.mean([v["recall_100"] for v in per_query.values()])),
            "n_queries": len(per_query),
        }
    return {"dataset": meta, "environment": {**environment(), "hms": hms["environment"]}, "sparse_dim": hms["sparse_dim"],
            "ingest_secs": hms["ingest_secs"], "latency_us": hms["latency"], "metrics": scores}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["ann", "beir"])
    ap.add_argument("name")
    ap.add_argument("hms", nargs="+")
    args = ap.parse_args()
    report = run_ann(args.name, args.hms) if args.kind == "ann" else run_beir(args.name, args.hms[0])
    out = RESULTS / f"public_{args.name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
