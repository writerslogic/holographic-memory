"""Per-call client overhead of the SDK against a local hms-server, paired with raw http.client.

Starts the server binary given by ``--server``, loads ``--rows`` documents, then runs
``--rounds`` interleaved rounds. Each round times every variant on query, add, replace and
delete, rotating the variant order between rounds. Variants:

- ``raw``: one persistent ``http.client.HTTPConnection`` with a pre-encoded JSON body.
- ``sdk``: ``HolographicClient`` from the installed package (sync).
- ``sdk-async``: ``AsyncHolographicClient`` awaited sequentially in one event loop.

The SDK is whichever ``holographic_sdk`` is importable, so a before/after pair runs this script
once per SDK build (each in its own environment), interleaving the processes.

Writes a JSON summary with p50/p99 per variant and operation to ``--out``.
"""

import argparse
import asyncio
import http.client
import importlib.metadata
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time

API_KEY = "bench-key"


def vec(rng, dim):
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def pct(samples, q):
    s = sorted(samples)
    return s[min(len(s) - 1, int(q * len(s)))]


class Raw:
    def __init__(self, url):
        host, port = url.removeprefix("http://").split(":")
        self.conn = http.client.HTTPConnection(host, int(port))
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    def call(self, method, path, body=None):
        self.conn.request(method, path, body=body, headers=self.headers)
        res = self.conn.getresponse()
        data = res.read()
        if res.status >= 300:
            raise RuntimeError(f"{res.status} {data[:200]!r}")
        return data


def installed(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def make_ops(rng, dim, n, tag):
    """Per-variant inputs: queries, fresh docs to add, the same ids to replace, then delete."""
    queries = [vec(rng, dim) for _ in range(n["query"])]
    ids = [f"{tag}-{i}" for i in range(n["write"])]
    adds = [{"id": i, "text": i, "vector": vec(rng, dim)} for i in ids]
    reps = [{"id": i, "text": i, "vector": vec(rng, dim)} for i in ids]
    return queries, adds, reps, ids


def time_sync(variant, client, ops, top_k):
    queries, adds, reps, ids = ops
    out = {k: [] for k in ("query", "add", "replace", "delete")}
    clock = time.perf_counter_ns
    if variant == "raw":
        for q in queries:
            t = clock()
            json.loads(client.call("POST", "/api/v1/query", json.dumps({"query_vector": q, "top_k": top_k})))
            out["query"].append(clock() - t)
        for key, docs in (("add", adds), ("replace", reps)):
            for d in docs:
                t = clock()
                client.call("POST", "/api/v1/documents/batch", json.dumps([d]))
                out[key].append(clock() - t)
        for i in ids:
            t = clock()
            client.call("DELETE", f"/api/v1/documents/{i}")
            out["delete"].append(clock() - t)
        return out
    for q in queries:
        t = clock()
        client.query(q, top_k=top_k)
        out["query"].append(clock() - t)
    for key, docs in (("add", adds), ("replace", reps)):
        for d in docs:
            t = clock()
            client.add_documents([d])
            out[key].append(clock() - t)
    for i in ids:
        t = clock()
        client.delete_document(i)
        out["delete"].append(clock() - t)
    return out


async def time_async(client, ops, top_k):
    queries, adds, reps, ids = ops
    out = {k: [] for k in ("query", "add", "replace", "delete")}
    clock = time.perf_counter_ns
    for q in queries:
        t = clock()
        await client.query(q, top_k=top_k)
        out["query"].append(clock() - t)
    for key, docs in (("add", adds), ("replace", reps)):
        for d in docs:
            t = clock()
            await client.add_documents([d])
            out[key].append(clock() - t)
    for i in ids:
        t = clock()
        await client.delete_document(i)
        out["delete"].append(clock() - t)
    return out


def start_server(binary, dim, data_dir):
    env = {**os.environ, "HMS_API_KEY": API_KEY}
    proc = subprocess.Popen(
        [binary, "--bind", "127.0.0.1:0", "--input-dim", str(dim), "--data-dir", data_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env,
    )
    line = proc.stdout.readline().strip()
    if not line.startswith("hms-server listening on "):
        proc.terminate()
        raise RuntimeError(line)
    return proc, f"http://{line.rsplit(' ', 1)[1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--rows", type=int, default=10_000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=2000)
    ap.add_argument("--writes", type=int, default=500)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from holographic_sdk import AsyncHolographicClient, HolographicClient

    impls = {"sdk": (HolographicClient, AsyncHolographicClient)}

    rng = random.Random(7)
    n = {"query": args.queries, "write": args.writes}
    with tempfile.TemporaryDirectory() as data_dir:
        proc, url = start_server(args.server, args.dim, data_dir)
        try:
            raw = Raw(url)
            for start in range(0, args.rows, 1000):
                docs = [{"id": f"row-{i}", "text": f"row {i}", "vector": vec(rng, args.dim)} for i in range(start, min(args.rows, start + 1000))]
                raw.call("POST", "/api/v1/documents/batch", json.dumps(docs))
            raw.call("POST", "/api/v1/index/maintain", "{}")

            variants = ["raw"] + [v for name in impls for v in (name, f"{name}-async")]
            samples = {v: {k: [] for k in ("query", "add", "replace", "delete")} for v in variants}
            clients = {"raw": raw}
            for name, (sync_cls, _) in impls.items():
                clients[name] = sync_cls(url, api_key=API_KEY)
            loop = asyncio.new_event_loop()
            for name, (_, async_cls) in impls.items():
                clients[f"{name}-async"] = async_cls(url, api_key=API_KEY)

            # Warm every connection before the first timed call.
            warm = make_ops(rng, args.dim, {"query": 50, "write": 20}, "warm")
            for v in variants:
                if v.endswith("-async"):
                    loop.run_until_complete(time_async(clients[v], warm, args.top_k))
                else:
                    time_sync(v, clients[v], warm, args.top_k)
                warm = make_ops(rng, args.dim, {"query": 50, "write": 20}, f"warm-{v}")

            # Variants alternate in small blocks with a rotating order, so background load on a
            # shared machine lands on raw and SDK calls alike and the per-call difference stays paired.
            blocks = 20
            block = {"query": max(1, n["query"] // blocks), "write": max(1, n["write"] // blocks)}
            for r in range(args.rounds):
                for b in range(blocks):
                    s = (r + b) % len(variants)
                    for v in variants[s:] + variants[:s]:
                        ops = make_ops(rng, args.dim, block, f"r{r}-b{b}-{v}")
                        if v.endswith("-async"):
                            got = loop.run_until_complete(time_async(clients[v], ops, args.top_k))
                        else:
                            got = time_sync(v, clients[v], ops, args.top_k)
                        for k, xs in got.items():
                            samples[v][k].extend(xs)
        finally:
            proc.terminate()
            proc.wait(timeout=30)

    summary = {}
    for v, ops in samples.items():
        summary[v] = {}
        for k, xs in ops.items():
            p50 = pct(xs, 0.5) / 1e3
            p99 = pct(xs, 0.99) / 1e3
            raw50 = pct(samples["raw"][k], 0.5) / 1e3
            summary[v][k] = {"n": len(xs), "p50_us": round(p50, 1), "p99_us": round(p99, 1), "mean_us": round(statistics.fmean(xs) / 1e3, 1), "p50_over_raw_us": round(p50 - raw50, 1)}
    result = {
        "label": args.label,
        "config": {k: v for k, v in vars(args).items() if k not in ("server", "out", "label")},
        "python": sys.version.split()[0],
        "packages": {p: installed(p) for p in ("httpx", "httpcore", "sniffio", "orjson")},
        "machine": platform.machine(),
        "load_average": os.getloadavg(),
        "results": summary,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)



if __name__ == "__main__":
    main()
