# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Server-mode runs: hms-server against server comparators, written as results files with
`mode: "server"` and `reference: "hms-server"` so the table keeps them in their own sections.

Results use the same shape as the embedded runs (analyze.py reads both unchanged). Every phase
is one server_worker process that starts and stops the server tree itself; server memory is the
sampled tree RSS reported by the worker, never the client's wait4 figure.
"""

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

from . import SCHEMA_VERSION, data, envs, registry, server_proc, servers, textdata
from .common import CACHE, RESULTS, ROOT, commit, dir_bytes, hardware, log, run_measured, write_json
from .runner import WORK, _parse_result, _quality, _rotate
from .server_registry import DURABILITY, DURABILITY_CLASS, ENV, HMS_SERVER

WORKER = Path(__file__).resolve().parent / "server_worker.py"

PROTOCOL = {
    "mode": "server: each system is a separate server process tree on this host",
    "transport": "loopback TCP; one client process, one persistent connection, sequential queries",
    "timing": "client-side perf_counter_ns around each request (send, server work, receive, response decode); request bodies serialized before the timer",
    "build": "timer starts when the server answers readiness on an empty data directory; covers index creation, ingest and waiting until queryable "
             "(optimizer idle / refreshed and merged / index built / tasks succeeded); the server is then stopped cleanly",
    "serving": "a fresh server process per serve round on the built data directory",
    "cold_start": "server process start to the first answered query, including any load or index work the server needs after a restart",
    "rss": f"server process tree: {server_proc.RSS_METHOD}; peak samples every 250 ms during startup and ingest and between sweep points, "
           "never during a timed loop, so the peak is a lower bound; steady = sample after the sweep",
    "disk": "allocated bytes (du) of the server data directory after a clean stop (logs excluded)",
    "ops": "100 replacements then 100 deletions on a copy of the built data directory, one request each, each acknowledged per `durability`",
    "durability": DURABILITY,
    "durability_class": DURABILITY_CLASS,
    "hms_server": "ANN: the graph snapshot lives in memory only, so every serving process rebuilds it (POST /api/v1/index/maintain) before "
                  "its first query; its cold start and serving peak RSS include that build. Text: documents enter through the memory API "
                  "(remember) and queries through POST /api/v1/search (document search's BM25 ranking; ids and scores only)",
    "guards_lifted": "Weaviate's disk read-only guard (DISK_USE_READONLY_PERCENTAGE=100): a safety stop that trips on a nearly full host disk",
    "server_threads":"build threads = protocol build_threads where the server has a knob; per-query search parallelism pinned to 1 where a knob exists",
}


def hms_server_binary():
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target"))
    path = target / "release" / "hms-server"
    if not path.exists():
        log("building hms-server")
        subprocess.run(["cargo", "build", "--release", "--features", "server", "--bin", "hms-server"], check=True, cwd=ROOT)
    return path


def install(systems):
    """Fetch what `systems` need, outside any timing lock. Returns {system: reason} for those that
    cannot run on this host."""
    unavailable = {}
    envs.ensure(ENV)
    for system in systems:
        engine = registry.SYSTEMS[system]["engine"]
        try:
            if engine == "hms-server":
                hms_server_binary()
            else:
                servers.prepare(engine)
        except servers.NotAvailable as error:
            unavailable[system] = str(error)
    return unavailable


def _worker(args, threads, timeout):
    python = envs.ensure(ENV)
    env = {"RAYON_NUM_THREADS": str(threads), "CARGO_TARGET_DIR": str(hms_server_binary().parents[1])}
    workdir = Path(args[args.index("--workdir") + 1])
    try:
        stdout, stderr, wall, client_peak = run_measured([str(python), "-I", str(WORKER), *args], env=env, timeout=timeout)
    finally:
        # A client killed by the timeout never stopped its server.
        server_proc.kill_orphans(workdir / "logs")
    result = _parse_result(stdout)
    result["client_peak_rss_wait4_bytes"] = client_peak
    result["wall_s"] = wall
    return result


def _provenance(results):
    for system, entry in results["systems"].items():
        engine = registry.SYSTEMS[system]["engine"]
        if engine == "hms-server":
            text = (ROOT / "Cargo.toml").read_text()
            version = next((line.split('"')[1] for line in text.splitlines() if line.startswith("version")), "unknown")
            binary = hms_server_binary()
            entry["version"] = {"holographic-memory": version, "commit": commit(), "features": "server"}
            entry["install"] = {"bytes": binary.stat().st_size, "distributions": 1, "note": "hms-server binary (release, feature `server`)"}
            continue
        prov = servers.provenance(engine)
        entry["provenance"] = prov
        entry["version"] = {engine: prov.get("version") or prov.get("postgres") or prov.get("redis")}
        client = envs.versions(ENV, ["psycopg", "redis"])
        entry["version"]["client"] = ", ".join(f"{k} {v}" for k, v in client.items())
        if "bytes" in prov:
            entry["install"] = {"bytes": prov["bytes"], "distributions": 1, "note": "unpacked pinned release artifact"}


def _base(kind, spec, hw, dataset_meta):
    return {"schema_version": SCHEMA_VERSION, "kind": kind, "mode": "server", "reference": HMS_SERVER, "suite": spec["name"], "commit": commit(),
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "hardware": hw, "dataset": dataset_meta,
            "facts": registry.FACTS, "timing_boundary": registry.TIMING_BOUNDARY, "not_run": registry.NOT_RUN, "systems": {}, "skipped": {},
            "platform": f"{platform.system()} {platform.machine()}"}


def _out_dir(spec):
    return CACHE / "competitive" / "smoke" if spec["name"] == "smoke" else RESULTS


def run_ann(dataset, n_train, n_test, spec, systems, out_path=None):
    directory, meta = data.ann_subset(dataset, n_train, n_test)
    threads, timeout = spec["threads"], spec.get("timeout_s", 7200)
    hw = hardware()
    results = _base("ann", spec, hw, {k: meta[k] for k in ("name", "metric", "n_train", "n_test", "dim") if k in meta} | {
        "subset_of": meta.get("subset_of"), "train_sha256": meta.get("train_sha256"), "test_sha256": meta.get("test_sha256")})
    results["protocol"] = {**PROTOCOL, "k": 10, "rounds": spec["rounds"], "build_repeats": spec["build_repeats"], "build_threads": threads,
                           "search_threads": 1, "queries": meta["n_test"], "recall_targets": [0.9, 0.95, 0.99],
                           "selection": "pooled-median-QPS argmax over sweep rows with recall >= target"}
    from .server_worker import ANN
    if meta["metric"] != "angular" and HMS_SERVER in systems:
        results["skipped"][HMS_SERVER] = f"hms-server normalizes every vector (cosine only); dataset metric {meta['metric']}"
        systems = [s for s in systems if s != HMS_SERVER]
    results["skipped"].update(install(systems))
    plan = [(s, b) for s in systems if s not in results["skipped"] for b in ANN[s].BUILDS]
    state = {}
    root = WORK / "server" / f"{dataset}-{meta['n_train']}"
    for system, build_id in plan:
        workdir = root / f"{system}-{build_id}"
        builds = []
        try:
            for _ in range(spec["build_repeats"]):
                r = _worker(["ann", "build", "--system", system, "--build", build_id, "--data", str(directory), "--workdir", str(workdir),
                             "--threads", str(threads)], threads, timeout)
                builds.append(r | {"index_bytes": dir_bytes(workdir / "data")})
            state[(system, build_id)] = {"builds": builds, "workdir": workdir, "serves": []}
        except Exception as error:
            results["skipped"][f"{system}:{build_id}"] = str(error)[-600:]
            log(f"{system}:{build_id} build failed: {str(error)[-300:]}")
    for r in range(spec["rounds"]):
        for key in _rotate([k for k in plan if k in state], r):
            system, build_id = key
            try:
                one = _worker(["ann", "serve", "--system", system, "--build", build_id, "--data", str(directory),
                               "--workdir", str(state[key]["workdir"]), "--threads", str(threads), "--repeats", "2"], threads, timeout)
                one["peak_rss_bytes"] = one["server_peak_rss_bytes"]
                state[key]["serves"].append(one)
            except Exception as error:
                results["skipped"][f"{system}:{build_id}"] = str(error)[-600:]
                log(f"{system}:{build_id} serve failed: {str(error)[-300:]}")
                state.pop(key)
    seen = set()
    for (system, build_id), st in list(state.items()):
        if system in seen:
            continue
        seen.add(system)
        copy = st["workdir"].with_name(st["workdir"].name + "-ops")
        shutil.rmtree(copy, ignore_errors=True)
        shutil.copytree(st["workdir"], copy, symlinks=True)
        try:
            st["ops"] = _worker(["ann", "ops", "--system", system, "--build", build_id, "--data", str(directory), "--workdir", str(copy),
                                 "--threads", str(threads)], threads, timeout)
        except Exception as error:
            st["ops"] = {"error": str(error)[-300:]}
        shutil.rmtree(copy, ignore_errors=True)
    for (system, build_id), st in state.items():
        entry = results["systems"].setdefault(system, {"display": registry.SYSTEMS[system]["display"], "builds": {}})
        entry["builds"][build_id] = {"build_runs": st["builds"], "serve_rounds": st["serves"], "params": ANN[system].BUILDS[build_id]}
        if "ops" in st:
            entry["ops"] = st["ops"]
    _provenance(results)
    out_path = out_path or _out_dir(spec) / f"{spec['name']}-server-ann-{dataset}-n{meta['n_train']}-{hw['class']}.json"
    write_json(out_path, results)
    return out_path


def run_cli(a, spec, wanted):
    """`hms-bench run` in server mode: hms-server plus the requested (default: all) server systems."""
    from .server_registry import SERVER_ANN_SYSTEMS, SERVER_TEXT_SYSTEMS
    def pick(pool):
        # hms-server always runs; a kind is skipped when systems were named and none is of it.
        others = [s for s in pool if s != HMS_SERVER and (wanted is None or s in wanted)]
        return [HMS_SERVER] + others if others or wanted is None or wanted == {HMS_SERVER} else []

    if a.only in ("ann", "both") and (systems := pick(SERVER_ANN_SYSTEMS)):
        for dataset, n_train, n_test in spec["ann"]:
            if not a.datasets or dataset in a.datasets:
                print(run_ann(dataset, n_train, n_test, spec, systems))
    if a.only in ("text", "both") and (systems := pick(SERVER_TEXT_SYSTEMS)):
        for dataset in spec["text"]:
            if not a.datasets or dataset in a.datasets:
                print(run_text(dataset, spec, systems))


def run_text(dataset, spec, systems, out_path=None):
    folder = textdata.load(dataset)
    meta = json.loads((folder / "meta.json").read_text())
    threads, timeout = spec["threads"], spec.get("timeout_s", 7200)
    hw = hardware()
    results = _base("text", spec, hw, meta | {"id": dataset})
    results["protocol"] = {**PROTOCOL, "rounds": spec["rounds"], "build_repeats": spec["build_repeats"], "k": 100, "passes_per_round": 3,
                           "build_threads": threads,
                           "retrieval": "lexical, the lowercase alphanumeric query word set, top 100; each server's own analyzer and ranking "
                                        "(BM25 for OpenSearch only; see timing_boundary)"}
    from .server_worker import TEXT
    systems = [s for s in systems if s in TEXT]
    results["skipped"].update(install(systems))
    input_path = folder / "input.json"
    serve_input = folder / "serve-input.json"
    serve_input.write_text(json.dumps({"documents": [], "queries": json.loads(input_path.read_text())["queries"]}))
    qrels = json.loads((folder / "qrels.json").read_text())
    results["qrels_queries"] = len(qrels)
    root = WORK / "server" / f"text-{dataset.replace(':', '-')}"
    runs, live = {}, []
    for system in systems:
        if system in results["skipped"]:
            continue
        workdir = root / system
        runs[system] = {"build_runs": [], "serve_rounds": []}
        try:
            for _ in range(spec["build_repeats"]):
                r = _worker(["text", "build", "--system", system, "--input", str(input_path), "--workdir", str(workdir), "--threads", str(threads)],
                            threads, timeout)
                runs[system]["build_runs"].append(r | {"disk_bytes": dir_bytes(workdir / "data")})
            live.append(system)
        except Exception as error:
            results["skipped"][system] = str(error)[-600:]
            log(f"{system} text build failed: {str(error)[-300:]}")
    for r in range(spec["rounds"]):
        for system in _rotate(list(live), r):
            try:
                body = _worker(["text", "serve", "--system", system, "--input", str(serve_input), "--workdir", str(root / system),
                                "--threads", str(threads)], threads, timeout)
                rows = body.pop("rows")
                body["quality"] = _quality(rows, qrels) if r == 0 else None
                runs[system]["serve_rounds"].append(body)
            except Exception as error:
                results["skipped"][system] = str(error)[-600:]
                log(f"{system} text serve failed: {str(error)[-300:]}")
                live.remove(system)
    for system in live:
        copy = root / f"{system}-ops"
        shutil.rmtree(copy, ignore_errors=True)
        shutil.copytree(root / system, copy, symlinks=True)
        try:
            runs[system]["ops"] = _worker(["text", "ops", "--system", system, "--input", str(input_path), "--workdir", str(copy),
                                           "--threads", str(threads)], threads, timeout)
        except Exception as error:
            runs[system]["ops"] = {"error": str(error)[-300:]}
        shutil.rmtree(copy, ignore_errors=True)
    for system in live:
        results["systems"][system] = {"display": registry.SYSTEMS[system]["display"], **runs[system]}
    _provenance(results)
    out_path = out_path or _out_dir(spec) / f"{spec['name']}-server-text-{dataset.replace(':', '-')}-{hw['class']}.json"
    write_json(out_path, results)
    return out_path
