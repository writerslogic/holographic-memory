# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26", "faiss-cpu>=1.8", "hnswlib>=0.8", "pytrec-eval-terrier>=0.5.6"]
# ///
"""Score public-bench output and run comparison libraries on the same data.

  evaluate.py ann  <dataset> <hms-result.json>...   FAISS (flat, HNSW) and hnswlib sweeps
  evaluate.py beir <dataset> <hms-runs.json>        nDCG@10 / recall@100 with pytrec_eval
  evaluate.py longmemeval s <hms-runs.json>         recall/nDCG@5,10 with LongMemEval's own eval_utils.py

Writes benchmarks/results/public_<dataset>.json.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import time
import urllib.request
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


MAX_LOAD = 3.0
MAX_WAIT_SECS = 10800  # total budget for load-gate waiting across one run of this script
_wait_deadline: float | None = None
EF_SWEEP = (16, 32, 64, 128, 256, 512, 768, 1024)
TARGET_RECALLS = (0.90, 0.95)


def wait_for_idle() -> tuple[float, bool]:
    """Waits until the 1-minute load is below MAX_LOAD or the script's total wait budget is
    spent; returns the load and whether the gate was met."""
    global _wait_deadline
    if _wait_deadline is None:
        _wait_deadline = time.monotonic() + MAX_WAIT_SECS
    while True:
        load = os.getloadavg()[0]
        if load < MAX_LOAD:
            return load, True
        if time.monotonic() >= _wait_deadline:
            return load, False
        time.sleep(5)


def timed_queries(search, queries: np.ndarray, repeats: int = 1) -> tuple[np.ndarray, float, dict]:
    """Single-threaded, one query at a time, as ann-benchmarks does. Each repeat waits for the
    load gate; returns the first repeat's results, the median QPS and the per-run record."""
    out = np.empty((len(queries), K), dtype=np.int64)
    runs, loads, gate = [], [], True
    for rep in range(repeats):
        load, met = wait_for_idle()
        loads.append(load)
        gate &= met
        res = out if rep == 0 else np.empty_like(out)
        t = time.perf_counter()
        for i, q in enumerate(queries):
            res[i] = search(q[None, :])
        runs.append(len(queries) / (time.perf_counter() - t))
    return out, float(np.median(runs)), {"qps_runs": runs, "load_1m_before_runs": loads, "load_gate_met": gate}


def qps_at_recall(points: list[tuple[float, float]], target: float) -> dict:
    """QPS at `target` recall on the Pareto frontier of (recall, qps) points, interpolated
    linearly in log(QPS) between the two frontier points that bracket the target."""
    frontier = []
    for r, q in sorted(points, key=lambda x: (-x[0], -x[1])):
        if not frontier or q > frontier[-1][1]:
            frontier.append((r, q))
    frontier.reverse()  # ascending recall, descending qps
    above = [(r, q) for r, q in frontier if r >= target]
    below = [(r, q) for r, q in frontier if r < target]
    if not above:
        return {"qps": None, "method": "not reached", "max_recall": max(r for r, _ in points)}
    hi = above[0]
    if not below:
        return {"qps": hi[1], "method": "lower bound (fastest swept point already exceeds the target)"}
    lo = below[-1]
    f = (target - lo[0]) / (hi[0] - lo[0])
    return {"qps": float(np.exp(np.log(lo[1]) + f * (np.log(hi[1]) - np.log(lo[1])))),
            "method": "interpolated", "between": [lo, hi]}


def run_ann(name: str, hms_files: list[str], repeats: int = 1) -> dict:
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
    found, qps, runs = timed_queries(lambda q: flat.search(q, K)[1][0], test)
    rows.append({"library": "faiss", "index": "IndexFlat (exact)", "params": {}, "recall_at_10": recall(found, truth),
                 "qps_single_thread": qps, "build_secs": build, "index_bytes": int(train.nbytes), **runs})

    for m in (16, 32):
        faiss.omp_set_num_threads(os.cpu_count())
        t = time.perf_counter()
        hnsw = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT if angular else faiss.METRIC_L2)
        hnsw.hnsw.efConstruction = 200
        hnsw.add(train)
        build = time.perf_counter() - t
        size = len(faiss.serialize_index(hnsw))
        faiss.omp_set_num_threads(1)
        for ef in EF_SWEEP:
            hnsw.hnsw.efSearch = ef
            found, qps, runs = timed_queries(lambda q: hnsw.search(q, K)[1][0], test, repeats)
            rows.append({"library": "faiss", "index": "IndexHNSWFlat", "params": {"M": m, "efConstruction": 200, "efSearch": ef},
                         "recall_at_10": recall(found, truth), "qps_single_thread": qps, "build_secs": build, "index_bytes": size,
                         **runs})

    t = time.perf_counter()
    h = hnswlib.Index(space=space, dim=dim)
    h.init_index(max_elements=len(train), ef_construction=200, M=16)
    h.add_items(train, num_threads=os.cpu_count())
    build = time.perf_counter() - t
    h.set_num_threads(1)
    for ef in EF_SWEEP:
        h.set_ef(max(ef, K))
        found, qps, runs = timed_queries(lambda q: h.knn_query(q, k=K)[0][0], test, repeats)
        rows.append({"library": "hnswlib", "index": "HNSW", "params": {"M": 16, "ef_construction": 200, "ef": ef},
                     "recall_at_10": recall(found, truth), "qps_single_thread": qps, "build_secs": build,
                     "index_bytes": int(h.element_count * (dim * 4 + 16 * 2 * 4 + 8)), **runs})

    series: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        if r["index"] == "IndexHNSWFlat":
            series.setdefault(f"faiss HNSW M={r['params']['M']}", []).append((r["recall_at_10"], r["qps_single_thread"]))
        elif r["library"] == "hnswlib":
            series.setdefault("hnswlib M=16", []).append((r["recall_at_10"], r["qps_single_thread"]))

    hms = [json.loads(Path(f).read_text()) for f in hms_files]
    for h in hms:
        if h.get("sweep"):
            series[h.get("system", "hms")] = [(r["recall_at_10"], r["qps_single_thread"]) for r in h["sweep"]]
    at_recall = {name: {f"{t:.2f}": qps_at_recall(pts, t) for t in TARGET_RECALLS} for name, pts in series.items()}
    return {"dataset": meta, "environment": environment(), "repeats": repeats, "max_load": MAX_LOAD,
            "qps_at_recall": at_recall, "competitors": rows,
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


# LongMemEval's retrieval metrics, fetched at a pinned commit and used unmodified.
LME_COMMIT = "9e0b455f4ef0e2ab8f2e582289761153549043fc"
LME_EVAL_URL = f"https://raw.githubusercontent.com/xiaowu0162/LongMemEval/{LME_COMMIT}/src/retrieval/eval_utils.py"
LME_KS = (5, 10)


def lme_eval_utils():
    path = data_dir("downloads") / f"longmemeval_eval_utils_{LME_COMMIT[:12]}.py"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(LME_EVAL_URL, headers={"User-Agent": "hms-public-bench/1.0"})
        path.write_bytes(urllib.request.urlopen(req).read())
    if not hasattr(np, "asfarray"):  # removed in NumPy 2; the file's only use is np.asfarray(x)
        np.asfarray = lambda a: np.asarray(a, dtype=float)
    spec = importlib.util.spec_from_file_location("lme_eval_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, hashlib.sha256(path.read_bytes()).hexdigest()


def run_longmemeval(variant: str, hms_file: str, part: set[str] | None = None) -> dict:
    ev, ev_sha = lme_eval_utils()
    d = data_dir(f"longmemeval_{variant}")
    meta = json.loads((d / "meta.json").read_text())
    questions = [json.loads(line) for line in (d / "questions.jsonl").read_text().splitlines()]
    hms = json.loads(Path(hms_file).read_text())
    qemb = load_f32(d / "queries.f32", meta["dim"])

    items, emb = {}, {}
    for gran in ("turn", "session"):
        per_q: dict[int, list] = {}
        for line in (d / f"{gran}.jsonl").read_text().splitlines():
            r = json.loads(line)
            per_q.setdefault(r["q"], []).append((r["id"], r["e"]))
        items[gran] = per_q
        emb[gran] = load_f32(d / f"{gran}.f32", meta["dim"])

    # The official script skips abstention questions and questions with no answer-bearing user turn.
    scored = [
        i for i, q in enumerate(questions)
        if "_abs" not in q["id"] and any("answer" in cid for cid, _ in items["turn"][i])
        and (part is None or q["id"] in part)
    ]
    types = sorted({questions[i]["type"] for i in scored})

    def full_ranking(ids: list[str], ranked: list[str]) -> list[int]:
        # Items a mode did not return (lexical score 0) follow in corpus order.
        index = {cid: j for j, cid in enumerate(ids)}
        head = [index[c] for c in ranked]
        seen = set(head)
        return head + [j for j in range(len(ids)) if j not in seen]

    def score(rankings_by_q: dict[int, list[int]], gran: str) -> dict:
        rows: dict[int, dict[str, dict[str, float]]] = {}
        for i in scored:
            ids = [cid for cid, _ in items[gran][i]]
            correct = list({c for c in ids if "answer" in c})
            rank = rankings_by_q[i]
            levels = {gran: ev.evaluate_retrieval}
            if gran == "turn":
                levels["session"] = ev.evaluate_retrieval_turn2session
            rows[i] = {}
            for level, fn in levels.items():
                m = {}
                for k in LME_KS:
                    r_any, r_all, nd = fn(rank, correct, ids, k=k)
                    m.update({f"recall_any@{k}": r_any, f"recall_all@{k}": r_all, f"ndcg_any@{k}": nd})
                rows[i][level] = m
        out = {}
        for group, members in [("overall", scored)] + [(t, [i for i in scored if questions[i]["type"] == t]) for t in types]:
            out[group] = {"n_questions": len(members)}
            for level in rows[members[0]]:
                out[group][level] = {
                    name: float(np.mean([rows[i][level][name] for i in members])) for name in rows[members[0]][level]
                }
        return out

    metrics: dict = {}
    for gran in ("turn", "session"):
        metrics[gran] = {}
        runs = hms["runs"][gran]["rankings"]
        qids = [q["id"] for q in questions]
        for mode, ranked in runs.items():
            metrics[gran][mode] = score(
                {i: full_ranking([c for c, _ in items[gran][i]], ranked[qids[i]]) for i in scored}, gran
            )
        # Independent exact-cosine reference over the same embeddings: HMS dense must reproduce it.
        ref = {}
        for i in scored:
            e = emb[gran][[row for _, row in items[gran][i]]]
            ref[i] = [int(j) for j in np.argsort(-(e @ qemb[i]), kind="stable")]
        metrics[gran]["numpy_dense_exact_reference"] = score(ref, gran)

    return {
        "dataset": meta,
        "evaluation": {
            "code": "xiaowu0162/LongMemEval src/retrieval/eval_utils.py",
            "commit": LME_COMMIT,
            "sha256": ev_sha,
            "scored_questions": len(scored),
            "excluded": "abstention (_abs) and questions without an answer-bearing user turn",
            "note": "Items are user turns only (official flat index); 'turn' is the paper's round granularity. "
                    "Session-level metrics of a turn run use evaluate_retrieval_turn2session. "
                    "recall_all and ndcg_any are the figures the official print_retrieval_metrics.py reports.",
        },
        "environment": {**environment(), "hms": hms["environment"]},
        "latency_us": {g: hms["runs"][g]["latency"] for g in ("turn", "session")},
        "metrics": metrics,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["ann", "beir", "longmemeval"])
    ap.add_argument("name")
    ap.add_argument("hms", nargs="+")
    ap.add_argument("--split", choices=["dev"], help="longmemeval: score only this part of longmemeval_split.json")
    ap.add_argument("--repeats", type=int, default=1, help="ann: timed runs per configuration (median reported)")
    ap.add_argument("--out", help="result file name under benchmarks/results (default public_<dataset>.json)")
    args = ap.parse_args()
    if args.kind == "longmemeval" and args.split:
        ids = set(json.loads((Path(__file__).with_name("longmemeval_split.json")).read_text())[args.split])
        report, out_name = run_longmemeval(args.name, args.hms[0], ids), f"longmemeval_{args.split}_minilm_baseline.json"
    elif args.kind == "longmemeval":
        report, out_name = run_longmemeval(args.name, args.hms[0]), f"public_longmemeval_{args.name}.json"
    else:
        report = run_ann(args.name, args.hms, args.repeats) if args.kind == "ann" else run_beir(args.name, args.hms[0])
        out_name = f"public_{args.name}.json"
    out = RESULTS / (args.out or out_name)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
