# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Orchestrates one (suite, dataset) run and writes benchmarks/results/competitive/<name>.json.

Every system runs in its own process per phase; the orchestrator measures wall time and peak RSS
of the child with wait4. Serve rounds are interleaved (HMS and every comparator once per round,
order rotated each round) so a transient load spike lands on all systems, not on one.
"""

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from . import SCHEMA_VERSION, data, envs, jars, registry, textdata
from .common import CACHE, RESULTS, ROOT, commit, dir_bytes, hardware, log, pass_summary, run_measured, write_json

WORK = Path(os.environ.get("HMS_BENCH_WORK", CACHE / "competitive" / "work"))
WORKER_DIR = Path(__file__).resolve().parent
_LIMITS = {"timeout": 7200}
# On datasets above 200k vectors these exact-scan or Python-heavy systems answer only the first N queries
# (a prefix of the same query file); recall and QPS are computed over that prefix and the row records it.
QUERY_CAPS = {"qdrant-local": 100, "sqlite-vec": 200, "faiss-flat": 1000, "duckdb-vss": 2000, "chroma": 2000, "milvus-lite": 2000, "lancedb": 2000}
SINGLE_THREAD = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "NUMBA_NUM_THREADS")}

SUITES = {
    # smoke: plumbing check only; its results are never committed.
    "smoke": {"ann": [("glove-100-angular", 5_000, 100)], "text": ["beir:nfcorpus"], "rounds": 1, "build_repeats": 1, "threads": 2, "timeout_s": 900},
    # light: hosted-runner subset; ratios only (see docs/benchmarks/README.md).
    "light": {"ann": [("glove-100-angular", 100_000, 1_000)], "text": ["beir:scifact", "beir:nfcorpus", "memory"],
              "rounds": 3, "build_repeats": 2, "threads": 4, "timeout_s": 1500},
    # standard: local workstation; full public sizes for the text sets, 250k glove and 250k SIFT.
    "standard": {"ann": [("glove-100-angular", 250_000, 2_000), ("sift-128-euclidean", 250_000, 2_000)],
                 "text": ["beir:scifact", "beir:nfcorpus", "memory", "beir:fiqa"],
                 "rounds": 5, "build_repeats": 3, "threads": 4, "timeout_s": 2700},
    # heavy: self-hosted or Modal; full GloVe-100 (1.18M) and SIFT-1M.
    "heavy": {"ann": [("glove-100-angular", None, 10_000), ("sift-128-euclidean", None, 10_000)],
              "text": ["beir:scifact", "beir:nfcorpus", "memory", "beir:fiqa"], "rounds": 3, "build_repeats": 2, "threads": 8, "timeout_s": 3600},
}


def _out_dir(spec):
    """Smoke runs are plumbing checks and stay out of the repository."""
    return CACHE / "competitive" / "smoke" if spec["name"] == "smoke" else RESULTS


# Cargo profile of the HMS binaries the harness runs and measures: `dist` is the release build
# without the symbol table (Cargo.toml), i.e. what is shipped. modal_run.py builds the same one.
HMS_PROFILE = "dist"


def hms_binary(name, example=False):
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    path = target / HMS_PROFILE / ("examples" if example else "") / name
    if not path.exists():
        log(f"building {name}")
        cmd = ["cargo", "build", "--profile", HMS_PROFILE, "--example" if example else "--bin", name]
        subprocess.run(cmd, check=True, cwd=ROOT)
    return path


def hms_install_facts(paths):
    """Install footprint of the HMS binaries used: size and non-OS dynamic libraries."""
    out = {}
    for path in paths:
        probe = ["otool", "-L", str(path)] if os.uname().sysname == "Darwin" else ["ldd", str(path)]
        text = subprocess.run(probe, capture_output=True, text=True).stdout
        non_system = [l for l in text.splitlines()[1:] if l.strip() and not any(
            m in l for m in ("/usr/lib/", "/System/", "libc.so", "libm.so", "libgcc_s", "libpthread", "libdl", "ld-linux", "linux-vdso", "librt"))]
        out[path.name] = {"bytes": path.stat().st_size, "non_system_dynamic_libraries": len(non_system),
                          "profile": HMS_PROFILE, "symbols": "stripped"}
    tree = subprocess.run(["cargo", "tree", "-e", "normal", "--prefix", "none", "-p", "holographic-memory", "--offline"],
                          capture_output=True, text=True, cwd=ROOT)
    crates = sorted({line.split()[0] for line in tree.stdout.splitlines() if line.strip()}) if tree.returncode == 0 else []
    out["_crate_dependencies"] = {"count": len(crates), "note": "unique crates in the default-feature normal dependency tree, including CLI-only crates; compiled in, not installed"}
    return out


def _rotate(items, r):
    k = r % len(items) if items else 0
    return items[k:] + items[:k]


def _parse_result(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    raise RuntimeError("worker printed no RESULT line: " + stdout[-500:])


def _worker(system, args, threads_env, timeout=None):
    timeout = timeout or _LIMITS["timeout"]
    env_name = registry.SYSTEMS[system]["env"]
    python = envs.ensure(env_name)
    t0 = time.time_ns()
    stdout, stderr, wall, peak = run_measured([str(python), "-I", str(WORKER_DIR / "ann_worker.py"), *args, "--t0-ns", str(t0)],
                                              env=threads_env, timeout=timeout)
    result = _parse_result(stdout)
    result["peak_rss_bytes"] = peak
    result["wall_s"] = wall
    return result


def run_ann(dataset, n_train, n_test, spec, systems, out_path=None, only_builds=None):
    directory, meta = data.ann_subset(dataset, n_train, n_test)
    threads = spec["threads"]
    hw = hardware()
    results = {"schema_version": SCHEMA_VERSION, "kind": "ann", "suite": spec["name"], "commit": commit(), "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "hardware": hw, "dataset": {k: meta[k] for k in ("name", "metric", "n_train", "n_test", "dim") if k in meta} | {
                   "subset_of": meta.get("subset_of"), "train_sha256": meta.get("train_sha256"), "test_sha256": meta.get("test_sha256")},
               "protocol": {"k": 10, "rounds": spec["rounds"], "build_repeats": spec["build_repeats"], "build_threads": threads,
                            "search_threads": 1, "queries": meta["n_test"],
                            "timing": "one query at a time, single client thread, per-query perf_counter_ns inside the worker process",
                            "ops": "100 replacements then 100 deletions; each mutation is acknowledged and explicitly flushed before the next",
                            "interleaving": "each round runs HMS and every comparator once, order rotated per round",
                            "query_caps": QUERY_CAPS if meta["n_train"] > 200_000 else {},
                            "recall_targets": [0.9, 0.95, 0.99], "selection": "pooled-median-QPS argmax over sweep rows with recall >= target"},
               "facts": registry.FACTS, "timing_boundary": registry.TIMING_BOUNDARY, "not_run": registry.NOT_RUN, "systems": {}, "skipped": {}}
    rounds = spec["rounds"]
    _LIMITS["timeout"] = spec.get("timeout_s", 7200)
    big = meta["n_train"] > 200_000
    env1 = {**SINGLE_THREAD}
    envb = {k: str(threads) for k in SINGLE_THREAD}
    plan = []  # (system, build_id)
    for system in systems:
        if system == registry.HMS:
            plan.append((system, "vertex-d32"))
            continue
        cls = _adapter_class(system)
        for build_id in cls.BUILDS:
            if only_builds is None or build_id in only_builds.get(system, cls.BUILDS):
                plan.append((system, build_id))
    state = {}
    # --- builds: every system builds and persists its index in a build process; serve rounds
    # open the persisted index in fresh processes, so build memory never counts as serving memory
    for system, build_id in plan:
        key = (system, build_id)
        workdir = WORK / f"{dataset}-{n_train}" / f"{system}-{build_id}"
        builds = []
        try:
            for _ in range(spec["build_repeats"]):
                if system == registry.HMS:
                    result = _hms_ann_build(directory, meta, threads, workdir)
                else:
                    result = _worker(system, ["build", "--system", system, "--build", build_id, "--data", str(directory), "--workdir", str(workdir),
                                              "--threads", str(threads)], envb)
                builds.append(result | {"index_bytes": dir_bytes(workdir)})
            state[key] = {"builds": builds, "workdir": workdir, "serves": []}
        except Exception as error:  # a system that cannot run is recorded, never silently dropped
            results["skipped"][f"{system}:{build_id}"] = str(error)[-600:]
            log(f"{system}:{build_id} build failed: {str(error)[-300:]}")
    # --- serve rounds
    for r in range(rounds):
        for system, build_id in _rotate(plan, r):
            key = (system, build_id)
            if key not in state:
                continue
            if system == registry.HMS:
                try:
                    state[key]["serves"].append(_hms_ann_round(directory, meta, state[key]["workdir"]))
                except Exception as error:
                    results["skipped"][f"{system}:{build_id}"] = str(error)[-600:]
                    log(f"hms round failed: {str(error)[-300:]}")
                continue
            try:
                merged = None
                for variant in _adapter_class(system).VARIANTS.get(build_id) or [None]:
                    args = ["serve", "--system", system, "--build", build_id, "--data", str(directory),
                            "--workdir", str(state[key]["workdir"]), "--repeats", "2"]
                    if big and system in QUERY_CAPS:
                        args += ["--queries", str(QUERY_CAPS[system])]
                    if variant:
                        args += ["--variant", json.dumps(variant)]
                    one = _worker(system, args, env1)
                    if merged is None:
                        merged = one
                    else:
                        merged["configs"] += one["configs"]
                        for field in ("peak_rss_bytes", "rss_steady"):
                            merged[field] = max(merged[field], one[field])
                state[key]["serves"].append(merged)
            except Exception as error:
                results["skipped"][f"{system}:{build_id}"] = str(error)[-600:]
                log(f"{system}:{build_id} serve failed: {str(error)[-300:]}")
                state.pop(key, None)
    # --- ops, once per system on a copy of the first build
    seen = set()
    for (system, build_id), st in list(state.items()):
        if system == registry.HMS:
            st["ops"] = {"supported": False, "reason": "VGraph is a static graph index built once; the document store handles updates with an exact scan, not an ANN index"}
            continue
        if system in seen:
            continue
        seen.add(system)
        copy = st["workdir"].with_name(st["workdir"].name + "-ops")
        shutil.rmtree(copy, ignore_errors=True)
        shutil.copytree(st["workdir"], copy)
        try:
            st["ops"] = _worker(system, ["ops", "--system", system, "--build", build_id, "--data", str(directory), "--workdir", str(copy)], env1)
        except Exception as error:
            st["ops"] = {"error": str(error)[-300:]}
        shutil.rmtree(copy, ignore_errors=True)
    for (system, build_id), st in state.items():
        entry = results["systems"].setdefault(system, {"display": registry.SYSTEMS[system]["display"], "builds": {}})
        entry["builds"][build_id] = {"build_runs": st["builds"], "serve_rounds": st["serves"],
                                      "params": _adapter_class(system).BUILDS[build_id] if system != registry.HMS else HMS_ANN_BUILDS[meta["metric"]]}
        if "ops" in st:
            entry["ops"] = st["ops"]
    _attach_versions_and_footprints(results)
    if system_hms := results["systems"].get(registry.HMS):
        system_hms["install"] = hms_install_facts([hms_binary("public-bench")])
    out_path = out_path or _out_dir(spec) / f"{spec['name']}-ann-{dataset}-n{meta['n_train']}-{results['hardware']['class']}.json"
    write_json(out_path, results)
    return out_path


# HMS configuration per dataset metric, each tuned on a train holdout (never on the benchmark queries).
HMS_ANN_BUILDS = {
    "angular": {"index": "vertex", "degree": 64, "build_ef": 500, "residual": True, "residual_bits": 8, "vertex_bits": 4,
                "id_bytes": 3, "align_rows": True, "reorder": True, "rerank": "64", "quantized_search": True,
                "ef": "64,96,128,192,256,384,512,736,1024",
                "source": "accepted configuration in benchmarks/results/qgraph_pool_paired_x86_glove_v4.json (tuned on a train holdout, not on these queries); "
                          "quantized construction search per benchmarks/results/ann_modal_confirm_v1.json (equal recall, 2.3x faster build)"},
    # 8-bit vertex codes: with 4-bit codes the traversal misses true SIFT neighbours (recall stops
    # at 0.990 even re-ranking 128 candidates; 8-bit reaches 0.9999). Degree and build_ef by the
    # fewest estimates per query at recall 0.99 over degree {32, 48, 64} x build_ef {200, 500}.
    "euclidean": {"index": "vertex", "degree": 64, "build_ef": 500, "residual": True, "residual_bits": 8, "vertex_bits": 8,
                  "id_bytes": 3, "align_rows": True, "reorder": True, "rerank": "32,64", "quantized_search": True,
                  "ef": "10,16,24,32,48,64,96,128,192,256",
                  "source": "SIFT-1M train-holdout sweeps in benchmarks/results/qgraph_l2_sweep_m4_sift1m_v1.json (4-bit and 8-bit codes, degree x build_ef)"},
}


def _hms_ann_args(build):
    """`public-bench ann-qgraph` arguments of one HMS_ANN_BUILDS entry."""
    return ["--index", build["index"], "--degree", str(build["degree"]), "--build-ef", str(build["build_ef"]),
            *(["--residual", "--residual-bits", str(build["residual_bits"])] if build["residual"] else []),
            "--vertex-bits", str(build["vertex_bits"]), "--id-bytes", str(build["id_bytes"]),
            *(["--align-rows"] if build["align_rows"] else []), *(["--reorder"] if build["reorder"] else []),
            "--rerank", build["rerank"], *(["--quantized-search"] if build["quantized_search"] else []), "--ef", build["ef"]]


def _hms_ann_command(directory, meta, out):
    return [str(hms_binary("public-bench")), "ann-qgraph", "--data", str(directory), "--out", str(out), "--metric", meta["metric"],
            *_hms_ann_args(HMS_ANN_BUILDS[meta["metric"]]), "--max-load", "1000000", "--max-wait-secs", "0"]


def _hms_ann_build(directory, meta, threads, workdir):
    """One `public-bench ann-qgraph --save-index` process: builds and encodes the index on
    `threads` threads and writes it to `workdir` (the index file and its build record), as each
    comparator's build process persists its index. Build time and build peak RSS come from here."""
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    out = WORK / "hms-ann-build.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    cmd = [*_hms_ann_command(directory, meta, out), "--save-index", str(workdir / "index.vgraph")]
    stdout, stderr, wall, peak = run_measured(cmd, env={"RAYON_NUM_THREADS": str(threads)}, timeout=14400)
    raw = json.loads(out.read_text())
    out.unlink()
    return {"build_s": raw["build"]["build_secs"], "peak_rss_bytes": peak, "wall_s": wall, "n": meta["n_train"],
            "in_memory_index_bytes": raw["index_bytes"], "residual_bytes": raw["residual_bytes"], "graph_hash": raw["build"]["graph_hash"],
            "memory_phases": raw["memory_phases"], "hms_threads": raw["build"]["threads"]}


def _hms_ann_round(directory, meta, workdir):
    """One serving process, single-threaded like every comparator's: `public-bench ann-qgraph
    --load-index` opens the saved index (memory-mapped and prefaulted before the timer, so its
    file-backed pages count in RSS) and runs the sweep. It never reads the train vectors."""
    out = WORK / "hms-ann-out.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    t0 = time.time_ns()
    cmd = [*_hms_ann_command(directory, meta, out), "--repeats", "2", "--latency", "--load-index", str(workdir / "index.vgraph")]
    stdout, stderr, wall, peak = run_measured(cmd, env=SINGLE_THREAD, timeout=14400)
    raw = json.loads(out.read_text())
    out.unlink()
    configs = [{"params": row["params"], "recall_at_10": row["recall_at_10"],
                "repeats": [{"qps": q, **({"p50_us": row["latency_us"]["p50"], "p99_us": row["latency_us"]["p99"]} if row.get("latency_us") else {})}
                            for q in row["qps_runs"]],
                "load_1m": row["load_1m_before_runs"][0], "load_gate_met": None} for row in raw["sweep"]]
    return {"index_bytes": raw["index_file"]["bytes"], "in_memory_index_bytes": raw["index_bytes"], "residual_bytes": raw["residual_bytes"],
            "rss_after_open": raw["rss_bytes"]["after_build"], "rss_steady": raw["rss_bytes"]["end"], "peak_rss_bytes": peak,
            "memory_phases": raw["memory_phases"], "graph_hash": raw["build"]["graph_hash"],
            "wall_s": wall, "n_queries": raw["n_queries"], "n": meta["n_train"], "configs": configs,
            "open_s": raw["index_file"]["open_secs"], "cold_start_s": (raw["first_query_unix_ns"] - t0) / 1e9}


def _adapter_class(system):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ann_worker_meta", WORKER_DIR / "ann_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SYSTEMS[system]


def _attach_versions_and_footprints(results):
    cache = {}
    for system, entry in results["systems"].items():
        reg = registry.SYSTEMS[system]
        if reg["family"] == "hms":
            entry["version"] = _hms_version()
            continue
        if system == "lucene":
            entry["version"] = {"lucene": jars.VERSION, "jvm": jars.java_version()}
            entry["install"] = jars.footprint()
            continue
        name = reg["env"]
        if name not in cache:
            pins = [line.split("==")[0] for line in envs.PINS.get(name, [])] or []
            cache[name] = {"versions": envs.versions(name, pins) if pins else {"python-sqlite3": _sqlite_version()},
                           "footprint": envs.footprint(name)}
        entry["version"] = cache[name]["versions"]
        entry["install"] = cache[name]["footprint"]


def _sqlite_version():
    import sqlite3
    return sqlite3.sqlite_version


def _hms_version():
    text = (ROOT / "Cargo.toml").read_text()
    for line in text.splitlines():
        if line.startswith("version"):
            return {"holographic-memory": line.split('"')[1], "commit": commit()}
    return {"commit": commit()}


# ------------------------------------------------------------------ text

def run_text(dataset, spec, systems, out_path=None):
    folder = textdata.load(dataset)
    meta = json.loads((folder / "meta.json").read_text())
    hw = hardware()
    results = {"schema_version": SCHEMA_VERSION, "kind": "text", "suite": spec["name"], "commit": commit(), "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "hardware": hw, "dataset": meta | {"id": dataset},
               "protocol": {"rounds": spec["rounds"], "build_repeats": spec["build_repeats"], "k": 100, "passes_per_round": 3,
                            "retrieval": "lexical only, OR over the lowercase alphanumeric query word set, top 100; each system's own analyzer and default BM25",
                            "timing": "one query at a time, single client thread; HMS timed inside its process, Python systems inside their worker",
                            "hms_rows": "hms = `--encode lexical` (BM25 index only, like-for-like); hms-eager = default eager encoding (chunk hypervectors at ingest); every build repeat is one cold process",
                            "interleaving": "each round runs HMS and every comparator once, order rotated per round",
                            "peak_rss": "wait4 rusage of that phase's own process, forked from a small helper so it never inherits the orchestrator's RSS (Linux)",
                            "disk": "allocated bytes of the store directory (du); where the filesystem reports sparse files as fully allocated, only non-zero 4 KiB blocks count",
                            "passes": "per timed pass: queries, total_ns and p50/p90/p99/max ns of the per-query latencies"},
               "facts": registry.FACTS, "timing_boundary": registry.TIMING_BOUNDARY, "not_run": registry.NOT_RUN, "systems": {}, "skipped": {}}
    input_path = folder / "input.json"
    # Every system's serve phase reads the queries alone, so no worker's serving RSS includes
    # the parsed corpus it does not need.
    serve_input_path = folder / "serve-input.json"
    serve_input_path.write_text(json.dumps({"documents": [], "queries": json.loads(input_path.read_text())["queries"]}))
    qrels = json.loads((folder / "qrels.json").read_text())
    results["qrels_queries"] = len(qrels)
    threads = {k: "1" for k in SINGLE_THREAD}
    stores = {}
    runs = {s: {"build_runs": [], "serve_rounds": []} for s in systems}
    for system in systems:
        store = WORK / f"text-{dataset.replace(':', '-')}" / system
        try:
            for _ in range(spec["build_repeats"]):
                shutil.rmtree(store, ignore_errors=True)
                store.parent.mkdir(parents=True, exist_ok=True)
                out = store.with_suffix(".build.json")
                stdout, stderr, wall, peak = _text_cmd(system, "build", input_path, store, out, threads)
                body = json.loads(out.read_text())
                runs[system]["build_runs"].append(body | {"peak_rss_bytes": peak, "wall_s": wall, "disk_bytes": dir_bytes(store)})
            stores[system] = store
        except Exception as error:
            results["skipped"][system] = str(error)[-600:]
            log(f"{system} text build failed: {str(error)[-300:]}")
    live = [s for s in systems if s in stores]
    for r in range(spec["rounds"]):
        for system in _rotate(live, r):
            out = stores[system].with_suffix(".serve.json")
            try:
                t0 = time.time_ns()
                stdout, stderr, wall, peak = _text_cmd(system, "serve", serve_input_path, stores[system], out, threads, t0=t0)
                body = json.loads(out.read_text())
                rows = body.pop("rows")
                body["quality"] = _quality(rows, qrels) if r == 0 else None
                body["passes"] = [pass_summary(ns) for ns in body.pop("passes_ns")]
                runs[system]["serve_rounds"].append(body | {"peak_rss_bytes": peak, "wall_s": wall})
            except Exception as error:
                results["skipped"][system] = str(error)[-600:]
    for system in live:
        copy = stores[system].with_name(stores[system].name + "-ops")
        shutil.rmtree(copy, ignore_errors=True)
        shutil.copytree(stores[system], copy)
        out = copy.with_suffix(".ops.json")
        try:
            _text_cmd(system, "ops", input_path, copy, out, threads)
            runs[system]["ops"] = json.loads(out.read_text())
        except Exception as error:
            runs[system]["ops"] = {"error": str(error)[-300:]}
        shutil.rmtree(copy, ignore_errors=True)
    for system, body in runs.items():
        if system in stores:
            results["systems"][system] = {"display": registry.SYSTEMS[system]["display"], **body}
    _attach_versions_and_footprints(results)
    for system in registry.HMS_TEXT_ENCODE:
        if system in results["systems"]:
            results["systems"][system]["install"] = hms_install_facts([hms_binary("competitive_text", example=True)])
            results["systems"][system]["encode"] = registry.HMS_TEXT_ENCODE[system]
    out_path = out_path or _out_dir(spec) / f"{spec['name']}-text-{dataset.replace(':', '-')}-{hw['class']}.json"
    write_json(out_path, results)
    return out_path


def _text_cmd(system, phase, input_path, store, out, env, t0=None):
    t0 = t0 or time.time_ns()
    if system in registry.HMS_TEXT_ENCODE:
        cmd = [str(hms_binary("competitive_text", example=True)), phase, "--input", str(input_path), "--store", str(store),
               "--out", str(out), "--t0-ns", str(t0), "--encode", registry.HMS_TEXT_ENCODE[system]]
    elif system == "lucene":
        classpath, _ = jars.ensure()
        cmd = ["java", "-cp", classpath, "LuceneBm25", phase, str(input_path), str(store), str(out), str(t0), "3", "100"]
    else:
        python = envs.ensure(registry.SYSTEMS[system]["env"])
        cmd = [str(python), "-I", str(WORKER_DIR / "text_worker.py"), phase, "--system", system, "--input", str(input_path),
               "--store", str(store), "--out", str(out), "--t0-ns", str(t0)]
    return run_measured(cmd, env=env, timeout=7200)


def _quality(rows, qrels):
    from .common import ndcg_at, recall_at
    ndcg, recall100, recall10 = [], [], []
    for row in rows:
        relevant = qrels[row["id"]]
        found = [h["documentId"] for h in row["results"]]
        ndcg.append(ndcg_at(found, relevant, 10))
        recall100.append(recall_at(found, relevant, 100))
        recall10.append(recall_at(found, relevant, 10))
    n = len(rows)
    return {"ndcg_at_10": sum(ndcg) / n, "recall_at_10": sum(recall10) / n, "recall_at_100": sum(recall100) / n,
            "per_query_ndcg_at_10": ndcg, "per_query_recall_at_100": recall100, "queries": n}
