# /// script
# requires-python = ">=3.11"
# dependencies = ["aiohttp>=3.9", "numpy>=1.26", "psutil>=5.9", "fastapi>=0.110", "uvicorn>=0.29"]
# ///
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Load, scaling and soak harness for hms-server.

    uv run deploy/bench/server_bench.py compare --hms-bin PATH --qdrant-bin PATH --out FILE
    uv run deploy/bench/server_bench.py soak    --hms-bin PATH --minutes 10 --out FILE

`compare` drives three services with the same synthetic corpus, the same query vectors and the
same closed-loop client, one fresh server per (system, corpus size): hms-server, a Qdrant server
(HNSW and exact search) and a plain SQLite + FastAPI service (deploy/bench/baseline_sqlite_fastapi.py).
Every number in the output is measured here; recall@10 is computed against an exact numpy scan.
The corpus is synthetic Gaussian clusters, so this measures serving overhead and scan cost, not
retrieval quality on real embeddings. Client and servers share one machine; the load average
before and after every run is recorded because it moves the absolute numbers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import platform
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import aiohttp
import numpy as np
import psutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DIM = 384
TOP_K = 10


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()


def make_corpus(n: int, queries: int, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian clusters; queries are existing documents plus noise, so true neighbors exist."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(max(8, n // 200), DIM)).astype(np.float32)
    assign = rng.integers(0, len(centers), size=n)
    docs = centers[assign] + 0.6 * rng.normal(size=(n, DIM)).astype(np.float32)
    docs = np.round(docs, 5)
    pick = rng.integers(0, n, size=queries)
    qs = np.round(docs[pick] + 0.15 * rng.normal(size=(queries, DIM)).astype(np.float32), 5)
    return docs, qs


def exact_topk(docs: np.ndarray, qs: np.ndarray, k: int) -> np.ndarray:
    d = docs / np.maximum(np.linalg.norm(docs, axis=1, keepdims=True), 1e-12)
    q = qs / np.maximum(np.linalg.norm(qs, axis=1, keepdims=True), 1e-12)
    scores = q @ d.T
    return np.argsort(-scores, axis=1)[:, :k]


# ---------------------------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------------------------


class System:
    name = ""
    variant = ""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.port = free_port()
        self.proc: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{self.port}"

    def pids(self) -> list[int]:
        return [self.proc.pid] if self.proc else []

    def rss_mb(self) -> float:
        total = 0
        for pid in self.pids():
            try:
                p = psutil.Process(pid)
                for q in [p, *p.children(recursive=True)]:
                    total += q.memory_info().rss
            except psutil.Error:
                continue  # the process exited between listing and reading
        return total / 2**20

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    async def wait_http(self, path: str, timeout: float = 60) -> None:
        deadline = time.time() + timeout
        async with aiohttp.ClientSession() as s:
            while time.time() < deadline:
                try:
                    async with s.get(self.base + path) as r:
                        if r.status < 500:
                            return
                except aiohttp.ClientError:
                    await asyncio.sleep(0)  # not up yet; retried below
                await asyncio.sleep(0.2)
        raise RuntimeError(f"{self.name} did not become ready")

    # Overridden per system.
    def command(self) -> list[str]:
        raise NotImplementedError

    def env(self) -> dict[str, str]:
        return {}

    def ingest_request(self, ids: list[int], vecs: np.ndarray) -> tuple[str, str, bytes]:
        raise NotImplementedError

    def query_request(self, vec: list[float]) -> tuple[str, bytes]:
        raise NotImplementedError

    def parse_ids(self, body: dict) -> list[int]:
        raise NotImplementedError

    async def prepare(self, session: aiohttp.ClientSession) -> None:
        pass

    async def after_ingest(self, session: aiohttp.ClientSession) -> dict:
        return {}

    async def start(self) -> None:
        with open(self.workdir / f"{self.name}-{self.variant}.log", "wb") as log:
            # The child inherits the descriptor, so closing ours after the spawn is safe.
            self.proc = subprocess.Popen(
                self.command(), stdout=log, stderr=log, env={**os.environ, **self.env()}, cwd=self.workdir
            )
        await self.wait_http(self.ready_path)

    ready_path = "/healthz"


class Hms(System):
    name = "hms-server"

    def __init__(self, workdir: Path, binary: str, maintain: bool, work: int):
        super().__init__(workdir)
        self.binary, self.maintain, self.work = binary, maintain, work
        self.variant = "exact+index" if maintain else "exact"

    def command(self) -> list[str]:
        return [
            self.binary, "--no-auth", "--bind", f"127.0.0.1:{self.port}",
            "--data-dir", str(self.workdir / "hms-data"), "--input-dim", str(DIM),
            "--max-concurrent-work", str(self.work), "--max-queue", "4096",
            "--queue-timeout-ms", "60000", "--log-level", "warn",
        ]

    def ingest_request(self, ids, vecs):
        docs = [{"id": f"d{i}", "text": f"document {i}", "vector": v.tolist(), "metadata": {"i": i}} for i, v in zip(ids, vecs)]
        return "POST", "/api/v1/documents/batch", json.dumps(docs).encode()

    def query_request(self, vec):
        return "/api/v1/query", json.dumps({"query_vector": vec, "top_k": TOP_K}).encode()

    def parse_ids(self, body):
        return [int(m["id"][1:]) for m in body["matches"]]

    async def after_ingest(self, session):
        if not self.maintain:
            return {}
        t = time.time()
        async with session.post(self.base + "/api/v1/index/maintain", data=b"{}") as r:
            out = await r.json()
        return {"index_train_seconds": time.time() - t, "document_index_trained": out.get("documentIndexTrained")}


class Qdrant(System):
    name = "qdrant"
    ready_path = "/readyz"

    def __init__(self, workdir: Path, binary: str, exact: bool):
        super().__init__(workdir)
        self.binary, self.exact = binary, exact
        self.variant = "exact" if exact else "hnsw(m=16,ef_construct=100,ef=128)"
        self.grpc = free_port()

    def command(self) -> list[str]:
        return [self.binary]

    def env(self):
        return {
            "QDRANT__SERVICE__HTTP_PORT": str(self.port),
            "QDRANT__SERVICE__GRPC_PORT": str(self.grpc),
            "QDRANT__SERVICE__HOST": "127.0.0.1",
            "QDRANT__STORAGE__STORAGE_PATH": str(self.workdir / "qdrant-storage"),
            "QDRANT__STORAGE__SNAPSHOTS_PATH": str(self.workdir / "qdrant-snapshots"),
            "QDRANT__TELEMETRY_DISABLED": "true",
            "QDRANT__LOG_LEVEL": "WARN",
        }

    async def prepare(self, session):
        body = {"vectors": {"size": DIM, "distance": "Cosine"}, "hnsw_config": {"m": 16, "ef_construct": 100},
                "optimizers_config": {"indexing_threshold": 1}}
        async with session.put(self.base + "/collections/bench", json=body) as r:
            assert r.status == 200, await r.text()

    def ingest_request(self, ids, vecs):
        pts = [{"id": i, "vector": v.tolist(), "payload": {"id": f"d{i}", "text": f"document {i}", "i": i}} for i, v in zip(ids, vecs)]
        return "PUT", "/collections/bench/points?wait=true", json.dumps({"points": pts}).encode()

    def query_request(self, vec):
        params = {"exact": True} if self.exact else {"hnsw_ef": 128}
        body = {"vector": vec, "limit": TOP_K, "with_payload": True, "params": params}
        return "/collections/bench/points/search", json.dumps(body).encode()

    def parse_ids(self, body):
        return [int(p["id"]) for p in body["result"]]

    async def after_ingest(self, session):
        # Let the optimizer finish building HNSW before measuring.
        t = time.time()
        while time.time() - t < 600:
            async with session.get(self.base + "/collections/bench") as r:
                info = (await r.json())["result"]
            indexed = info.get("indexed_vectors_count", 0) or 0
            if info["status"] == "green" and (self.exact or indexed >= 0.99 * info["points_count"]):
                break
            await asyncio.sleep(0.5)
        return {"settle_seconds": time.time() - t, "indexed_vectors": info.get("indexed_vectors_count"), "segments": info.get("segments_count")}


class SqliteFastApi(System):
    name = "sqlite+fastapi"
    variant = "exact numpy scan, 1 worker"

    def command(self) -> list[str]:
        return [sys.executable, "-m", "uvicorn", "baseline_sqlite_fastapi:app", "--host", "127.0.0.1",
                "--port", str(self.port), "--workers", "1", "--log-level", "warning", "--app-dir", str(HERE)]

    def env(self):
        return {"BASELINE_DB": str(self.workdir / "baseline.sqlite"), "BASELINE_DIM": str(DIM)}

    def ingest_request(self, ids, vecs):
        docs = [{"id": f"d{i}", "text": f"document {i}", "vector": v.tolist(), "metadata": {"i": i}} for i, v in zip(ids, vecs)]
        return "POST", "/documents/batch", json.dumps(docs).encode()

    def query_request(self, vec):
        return "/query", json.dumps({"query_vector": vec, "top_k": TOP_K}).encode()

    def parse_ids(self, body):
        return [int(m["id"][1:]) for m in body["matches"]]


# ---------------------------------------------------------------------------------------------
# Load generation
# ---------------------------------------------------------------------------------------------


def _worker(args: tuple) -> dict:
    base, path, bodies, concurrency, duration, warmup, seed = args

    async def run() -> dict:
        rng = np.random.default_rng(seed)
        lat: list[float] = []
        errors = 0
        stop_at = time.perf_counter() + warmup + duration
        measure_from = time.perf_counter() + warmup
        conn = aiohttp.TCPConnector(limit=concurrency, force_close=False)
        async with aiohttp.ClientSession(base, connector=conn, timeout=aiohttp.ClientTimeout(total=120)) as s:
            async def one() -> None:
                nonlocal errors
                while True:
                    now = time.perf_counter()
                    if now >= stop_at:
                        return
                    body = bodies[int(rng.integers(0, len(bodies)))]
                    t0 = time.perf_counter()
                    try:
                        async with s.post(path, data=body, headers={"Content-Type": "application/json"}) as r:
                            await r.read()
                            ok = r.status == 200
                    except aiohttp.ClientError:
                        ok = False
                    t1 = time.perf_counter()
                    if t0 >= measure_from:
                        if ok:
                            lat.append(t1 - t0)
                        else:
                            errors += 1
            cpu0 = time.process_time()
            await asyncio.gather(*[one() for _ in range(concurrency)])
            cpu = time.process_time() - cpu0
        return {"lat": lat, "errors": errors, "client_cpu_s": cpu}

    return asyncio.run(run())


def load(system: System, bodies: list[bytes], path: str, concurrency: int, duration: float, warmup: float = 2.0) -> dict:
    procs = min(4, concurrency)
    per = [concurrency // procs + (1 if i < concurrency % procs else 0) for i in range(procs)]
    jobs = [(system.base, path, bodies, c, duration, warmup, 1000 + i) for i, c in enumerate(per)]
    load_before = os.getloadavg()[0]
    with mp.get_context("spawn").Pool(procs) as pool:
        parts = pool.map(_worker, jobs)
    lat = sorted(x for p in parts for x in p["lat"])
    errors = sum(p["errors"] for p in parts)
    client_cpu = sum(p["client_cpu_s"] for p in parts)
    n = len(lat)

    def pct(q: float) -> float:
        return lat[min(n - 1, int(q * n))] * 1000 if n else float("nan")

    return {
        "concurrency": concurrency,
        "requests": n,
        "errors": errors,
        "qps": n / duration,
        "p50_ms": pct(0.50), "p95_ms": pct(0.95), "p99_ms": pct(0.99),
        "mean_ms": statistics.fmean(lat) * 1000 if n else float("nan"),
        "client_cpu_fraction_of_one_core": client_cpu / (duration + warmup),
        "load1_before": load_before, "load1_after": os.getloadavg()[0],
        "server_rss_mb": system.rss_mb(),
    }


async def ingest(system: System, session: aiohttp.ClientSession, docs: np.ndarray, batch: int = 500) -> dict:
    t = time.time()
    for start in range(0, len(docs), batch):
        ids = list(range(start, min(start + batch, len(docs))))
        method, path, body = system.ingest_request(ids, docs[start : start + batch])
        async with session.request(method, system.base + path, data=body, headers={"Content-Type": "application/json"}) as r:
            if r.status != 200:
                raise RuntimeError(f"{system.name} ingest failed: {r.status} {(await r.text())[:200]}")
    seconds = time.time() - t
    return {"ingest_seconds": seconds, "ingest_docs_per_s": len(docs) / seconds}


async def recall(system: System, session: aiohttp.ClientSession, qs: np.ndarray, truth: np.ndarray, n: int) -> float:
    hits = total = 0
    for q, t in zip(qs[:n], truth[:n]):
        path, body = system.query_request(q.tolist())
        async with session.post(system.base + path, data=body, headers={"Content-Type": "application/json"}) as r:
            got = system.parse_ids(await r.json())
        hits += len(set(got[:TOP_K]) & set(int(x) for x in t[:TOP_K]))
        total += TOP_K
    return hits / total


def environment(args: argparse.Namespace) -> dict:
    return {
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": {
            "platform": platform.platform(), "cpu": sh("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor(),
            "cpus": os.cpu_count(), "memory_gib": round(psutil.virtual_memory().total / 2**30, 1),
        },
        "load_average_at_start": os.getloadavg(),
        "git_commit": sh("git", "-C", str(REPO), "rev-parse", "HEAD"),
        "git_dirty": bool(sh("git", "-C", str(REPO), "status", "--porcelain", "--", "src", "Cargo.toml", "deploy")),
        "hms_server_version": sh(args.hms_bin, "--version"),
        "qdrant_version": sh(args.qdrant_bin, "--version") if getattr(args, "qdrant_bin", None) else None,
        "python": sys.version.split()[0],
        "client": "aiohttp, closed loop, same machine as the servers",
    }


async def compare(args: argparse.Namespace) -> None:
    sizes = [int(x) for x in args.sizes.split(",")]
    levels = [int(x) for x in args.concurrency.split(",")]
    results: list[dict] = []
    for n in sizes:
        docs, qs = make_corpus(n, 600)
        truth = exact_topk(docs, qs, TOP_K)
        bodies_cache: dict[str, list[bytes]] = {}
        factories = [
            lambda w: Hms(w, args.hms_bin, False, args.hms_work),
            lambda w: Hms(w, args.hms_bin, True, args.hms_work),
            lambda w: Qdrant(w, args.qdrant_bin, False),
            lambda w: Qdrant(w, args.qdrant_bin, True),
            SqliteFastApi,
        ]
        for make in factories:
            with tempfile.TemporaryDirectory(prefix="hmsbench-") as tmp:
                system = make(Path(tmp))
                print(f"== N={n} {system.name} [{system.variant}]", flush=True)
                try:
                    await system.start()
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
                        await system.prepare(session)
                        row: dict = {"system": system.name, "variant": system.variant, "n": n}
                        row.update(await ingest(system, session, docs))
                        row.update(await system.after_ingest(session))
                        row["recall_at_10"] = await recall(system, session, qs, truth, 200)
                        row["rss_mb_after_ingest"] = system.rss_mb()
                    path, _ = system.query_request(qs[0].tolist())
                    bodies = [system.query_request(q.tolist())[1] for q in qs]
                    row["load"] = []
                    for c in levels:
                        r = load(system, bodies, path, c, args.duration)
                        print(f"   c={c:3d} qps={r['qps']:8.1f} p50={r['p50_ms']:.2f}ms p99={r['p99_ms']:.2f}ms err={r['errors']} rss={r['server_rss_mb']:.0f}MB load1={r['load1_after']:.1f}", flush=True)
                        row["load"].append(r)
                    results.append(row)
                except Exception as e:  # recorded, not hidden: a system that cannot run is a result
                    print(f"   FAILED: {e}", flush=True)
                    results.append({"system": system.name, "variant": system.variant, "n": n, "failed": str(e)})
                finally:
                    system.stop()
    out = {
        "schema": "hms-server-bench/1",
        "environment": environment(args),
        "method": {
            "corpus": f"synthetic Gaussian clusters, dim {DIM}, sizes {sizes}; 600 queries = documents plus noise; top_k {TOP_K}",
            "recall": "recall@10 over 200 queries against an exact numpy cosine scan",
            "load": f"closed loop, concurrency {levels}, {args.duration}s per level after a 2s warmup, "
                    "client processes <= 4, a fresh server per (system, size)",
            "hms_flags": f"--no-auth --max-concurrent-work {args.hms_work} --max-queue 4096 --queue-timeout-ms 60000 (defaults otherwise)",
            "qdrant": "single node, default storage config, HNSW m=16 ef_construct=100 with hnsw_ef=128, and exact=true",
            "sqlite_fastapi": "SQLite WAL + in-memory numpy matrix, exact scan, one uvicorn worker (deploy/bench/baseline_sqlite_fastapi.py)",
            "caveat": "client and servers share one machine whose load average is recorded per run; absolute numbers are indicative",
        },
        "results": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")


# ---------------------------------------------------------------------------------------------
# Soak
# ---------------------------------------------------------------------------------------------


async def soak(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(11)
    work = Path(tempfile.mkdtemp(prefix="hmssoak-"))
    port = free_port()
    cmd = [args.hms_bin, "--no-auth", "--bind", f"127.0.0.1:{port}", "--data-dir", str(work / "data"), "--input-dim", str(DIM),
           "--dim", "2048", "--max-concurrent-work", "8", "--max-queue", "256", "--queue-timeout-ms", "5000", "--max-tenants", "6",
           "--tenant-idle-seconds", "4", "--compaction-interval-seconds", "20", "--log-level", "warn"]
    base = f"http://127.0.0.1:{port}"

    server_log = open(work / "server.log", "ab")  # closed when the soak ends

    def launch() -> subprocess.Popen:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=server_log)

    proc = launch()
    proc.stdout.readline()
    ps = psutil.Process(proc.pid)
    tenants = [f"t{i}" for i in range(12)]
    model: dict[str, set[str]] = {t: set() for t in tenants}
    locks = {t: asyncio.Lock() for t in tenants}
    stats = {"ok": 0, "busy_503": 0, "tenant_limit_503": 0, "rate_429": 0, "other_errors": 0, "by_op": {}}
    samples: list[dict] = []
    t_start = time.time()
    deadline = t_start + args.minutes * 60

    def active() -> list[str]:
        phase = int((time.time() - t_start) // 25)
        return [tenants[(phase * 2 + k) % len(tenants)] for k in range(3)]

    def vec() -> list[float]:
        return np.round(rng.normal(size=DIM), 5).tolist()

    async def call(s: aiohttp.ClientSession, op: str, method: str, path: str, tenant: str, body: bytes | None = None) -> dict | None:
        try:
            async with s.request(method, base + path, data=body, headers={"X-Tenant-ID": tenant, "Content-Type": "application/json"}) as r:
                data = await r.json(content_type=None)
                if r.status == 200:
                    stats["ok"] += 1
                    stats["by_op"][op] = stats["by_op"].get(op, 0) + 1
                    return data
                code = (data or {}).get("code", "")
                key = {"server_busy": "busy_503", "tenant_limit": "tenant_limit_503", "rate_limited": "rate_429"}.get(code, "other_errors")
                stats[key] += 1
                if key == "other_errors":
                    print("unexpected", op, r.status, str(data)[:200], flush=True)
        except aiohttp.ClientError as e:
            stats["other_errors"] += 1
            print("client error", op, e, flush=True)
        return None

    async def worker(s: aiohttp.ClientSession) -> None:
        while time.time() < deadline:
            t = active()[int(rng.integers(0, 3))]
            roll = rng.random()
            if roll < 0.55:
                await call(s, "query", "POST", "/api/v1/query", t, json.dumps({"query_vector": vec(), "top_k": 10}).encode())
            elif roll < 0.80:
                async with locks[t]:
                    ids = [f"d{int(rng.integers(0, 400))}" for _ in range(20)]
                    docs = [{"id": i, "text": f"doc {i}", "vector": vec(), "metadata": {"n": 1}} for i in ids]
                    if await call(s, "upsert", "POST", "/api/v1/documents/batch", t, json.dumps(docs).encode()):
                        model[t].update(ids)
            elif roll < 0.90:
                async with locks[t]:
                    victim = f"d{int(rng.integers(0, 400))}"
                    if victim in model[t]:
                        if await call(s, "delete", "DELETE", f"/api/v1/documents/{victim}", t):
                            model[t].discard(victim)
            elif roll < 0.95:
                await call(s, "list", "GET", "/api/v1/documents?limit=50", t)
            else:
                await call(s, "stats", "GET", "/api/v1/stats", t)

    async def sampler(s: aiohttp.ClientSession) -> None:
        while time.time() < deadline:
            await asyncio.sleep(args.sample_seconds)
            try:
                async with s.get(base + "/metrics") as r:
                    text = await r.text()
                kv = {}
                for line in text.splitlines():
                    if line and not line.startswith("#") and "{" not in line:
                        k, v = line.rsplit(" ", 1)
                        kv[k] = float(v)
                samples.append({
                    "t": round(time.time() - t_start, 1),
                    "rss_mb": kv.get("process_resident_memory_bytes", 0) / 2**20,
                    "ps_rss_mb": ps.memory_info().rss / 2**20,
                    "fds": ps.num_fds(), "threads": ps.num_threads(),
                    "work_in_use": kv.get("hms_work_slots_in_use"), "queue": kv.get("hms_queue_depth"),
                    "tenants_open": kv.get("hms_tenants_open"), "connections": kv.get("hms_connections_open"),
                    "evictions": kv.get("hms_tenant_evictions_total"),
                })
            except aiohttp.ClientError:
                continue  # a missed sample is not an error

    conn = aiohttp.TCPConnector(limit=64)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=60)) as s:
        await asyncio.gather(sampler(s), *[worker(s) for _ in range(args.concurrency)])
        await asyncio.sleep(1)
        async with s.get(base + "/metrics") as r:
            final_metrics = await r.text()

        async def server_ids(t: str) -> set[str] | None:
            got: set[str] = set()
            after = None
            while True:
                path = "/api/v1/documents?limit=1000" + (f"&after={after}" if after else "")
                for _ in range(40):
                    page = await call(s, "verify", "GET", path, t)
                    if page is not None:
                        break
                    await asyncio.sleep(0.5)
                else:
                    return None
                got.update(d["id"] for d in page["documents"])
                after = page["next_cursor"]
                if not after:
                    return got

        mismatches = []
        for t in tenants:
            got = await server_ids(t)
            if got != model[t]:
                mismatches.append(t)
        slots_idle = "hms_work_slots_in_use 0" in final_metrics
        # Durability: SIGTERM, restart on the same data, verify again.
        proc.send_signal(signal.SIGTERM)
        exit_code = proc.wait(timeout=60)
        proc = launch()
        proc.stdout.readline()
        restart_mismatches = []
        for t in tenants:
            got = await server_ids(t)
            if got != model[t]:
                restart_mismatches.append(t)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=60)

    half = [x for x in samples if x["t"] >= samples[-1]["t"] / 2] if samples else []
    xs = np.array([x["t"] for x in half])
    ys = np.array([x["rss_mb"] for x in half])
    slope = float(np.polyfit(xs, ys, 1)[0] * 60) if len(half) > 2 else float("nan")
    out = {
        "schema": "hms-server-soak/1",
        "environment": environment(args),
        "config": {"minutes": args.minutes, "concurrency": args.concurrency, "flags": cmd[1:], "workload": "55% query, 25% upsert x20, 10% delete, 5% list, 5% stats; tenants rotate every 25 s through 12 ids with max-tenants 6 and idle eviction 4 s"},
        "requests": stats,
        "samples": samples,
        "rss_mb_first": samples[0]["rss_mb"] if samples else None,
        "rss_mb_max": max((x["rss_mb"] for x in samples), default=None),
        "rss_mb_last": samples[-1]["rss_mb"] if samples else None,
        "rss_slope_mb_per_minute_second_half": slope,
        "fds_first": samples[0]["fds"] if samples else None,
        "fds_max": max((x["fds"] for x in samples), default=None),
        "fds_last": samples[-1]["fds"] if samples else None,
        "work_slots_idle_at_end": slots_idle,
        "model_mismatch_tenants": mismatches,
        "restart_mismatch_tenants": restart_mismatches,
        "sigterm_exit_code": exit_code,
    }
    server_log.close()
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("samples", "environment")}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--hms-bin", required=True)
    c.add_argument("--qdrant-bin", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--sizes", default="1000,10000,50000")
    c.add_argument("--concurrency", default="1,4,16")
    c.add_argument("--duration", type=float, default=8.0)
    c.add_argument("--hms-work", type=int, default=10)
    s = sub.add_parser("soak")
    s.add_argument("--hms-bin", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--minutes", type=float, default=10)
    s.add_argument("--concurrency", type=int, default=16)
    s.add_argument("--sample-seconds", type=float, default=5)
    args = ap.parse_args()
    asyncio.run(compare(args) if args.cmd == "compare" else soak(args))


if __name__ == "__main__":
    main()
