#!/usr/bin/env python3
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""RSS breakdown of `hms-server` after ingesting N synthetic documents, per component.

Mechanism: one fresh server (fresh data directory, loopback, `--no-auth`) per arm; the same
deterministic corpus (seeded LCG, values uniform in (-1, 1), rounded to 6 decimals) is
ingested through `POST /api/v1/documents/batch`, and the server's resident set size is read
from `ps -o rss=` (KiB; macOS and Linux) as the median of `--samples` readings taken
`--settle` seconds apart. Components are differences between arms that toggle one thing:

  hypervectors   eager - lexical                      (`--ingest-encode`)
  text tables    lexical+long text - lexical          (terms, postings, text segment)
  snapshot       after maintain (no graph) - before   (`--graph-min-rows 4194304`)
  graph          maintain (graph) - maintain (no graph)
  embeddings     slope of RSS over --input-dim, per stored value, times dims times N
  residual       lexical after ingest - idle - embeddings (BTree tables, mmapped journal
                 pages, allocator retention)

Output: one JSON document (schema in RUN.md) with every arm's readings and the derived
components, plus a table on stdout. Standard library only; no project code is imported.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

NO_GRAPH_ROWS = 4_194_304


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rss_kib(pid: int) -> int:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False)
    text = out.stdout.strip()
    if out.returncode != 0 or not text:
        raise RuntimeError(f"ps found no process {pid}")
    return int(text)


def dir_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
    return total


@dataclass
class Arm:
    name: str
    encode: str
    dim: int
    text_bytes: int
    maintain: bool
    graph_min_rows: int | None
    env: dict = field(default_factory=dict)
    readings: dict = field(default_factory=dict)


class Server:
    def __init__(self, binary: str, arm: Arm, workdir: Path, extra: list[str], encode_flag: bool):
        self.binary = binary
        self.arm = arm
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.data_dir = workdir / "hms-data"
        self.stderr_path = workdir / "server.stderr"
        self.extra = extra
        self.encode_flag = encode_flag
        self.proc: subprocess.Popen | None = None

    def command(self) -> list[str]:
        cmd = [
            self.binary, "--no-auth", "--bind", f"127.0.0.1:{self.port}",
            "--data-dir", str(self.data_dir), "--input-dim", str(self.arm.dim),
            "--log-level", "warn",
            "--max-concurrent-work", "4", "--max-queue", "4096", "--queue-timeout-ms", "60000",
        ]
        if self.encode_flag:
            cmd += ["--ingest-encode", self.arm.encode]
        if self.arm.graph_min_rows is not None:
            cmd += ["--graph-min-rows", str(self.arm.graph_min_rows)]
        return cmd + self.extra

    def start(self, timeout: float = 60.0) -> None:
        env = {**os.environ, **self.arm.env}
        # The child inherits the descriptor; the parent's copy closes once it has started.
        with open(self.stderr_path, "wb") as stderr:
            self.proc = subprocess.Popen(self.command(), stdout=subprocess.DEVNULL, stderr=stderr, env=env)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.arm.name}: server exited {self.proc.returncode}; see {self.stderr_path}")
            try:
                with urllib.request.urlopen(self.base + "/readyz", timeout=2) as r:
                    if r.status < 500:
                        return
            except (urllib.error.URLError, urllib.error.HTTPError, OSError):
                time.sleep(0.2)
        raise RuntimeError(f"{self.arm.name}: server not ready after {timeout}s")

    def call(self, method: str, path: str, body: bytes | None = None, timeout: float = 600.0) -> dict:
        req = urllib.request.Request(self.base + path, data=body, method=method)
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read()
        return json.loads(payload) if payload else {}

    def rss_mb(self, samples: int, settle: float) -> float:
        assert self.proc is not None
        values = []
        for _ in range(samples):
            values.append(rss_kib(self.proc.pid) / 1024.0)
            time.sleep(settle)
        return statistics.median(values)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def make_text(rng: random.Random, i: int, text_bytes: int) -> str:
    base = f"document {i}"
    if text_bytes <= len(base):
        return base
    words = []
    while sum(len(w) + 1 for w in words) + len(base) < text_bytes:
        words.append("w%d" % rng.randrange(5000))
    return base + " " + " ".join(words)


def batches(n: int, dim: int, batch: int, seed: int, text_bytes: int):
    """Yields JSON bodies for the batch endpoint; the vector stream depends only on (seed, dim)."""
    vec_rng = random.Random(seed)
    text_rng = random.Random(seed + 1)
    for start in range(0, n, batch):
        docs = []
        for i in range(start, min(n, start + batch)):
            vector = [round(vec_rng.uniform(-1.0, 1.0), 6) for _ in range(dim)]
            docs.append({"id": f"d{i}", "text": make_text(text_rng, i, text_bytes), "vector": vector, "metadata": {"i": i}})
        yield json.dumps(docs).encode()


def run_arm(binary: str, arm: Arm, args: argparse.Namespace, root: Path) -> Arm:
    workdir = root / arm.name
    workdir.mkdir(parents=True, exist_ok=True)
    server = Server(binary, arm, workdir, args.extra, not args.no_encode_flag)
    print(f"[{arm.name}] starting: {' '.join(server.command())}", flush=True)
    server.start()
    try:
        r = arm.readings
        r["rss_idle_mb"] = server.rss_mb(args.samples, args.settle)
        t = time.time()
        for body in batches(args.n, arm.dim, args.batch, args.seed, arm.text_bytes):
            server.call("POST", "/api/v1/documents/batch", body)
        r["ingest_seconds"] = time.time() - t
        stats = server.call("GET", "/api/v1/stats")
        r["vectors"] = stats.get("vectors")
        r["rss_after_ingest_mb"] = server.rss_mb(args.samples, args.settle)
        r["data_dir_bytes_after_ingest"] = dir_bytes(server.data_dir)
        if arm.maintain:
            t = time.time()
            out = server.call("POST", "/api/v1/index/maintain", b"{}")
            r["maintain_seconds"] = time.time() - t
            r["vector_index"] = out.get("vectorIndex")
            r["rss_after_maintain_mb"] = server.rss_mb(args.samples, args.settle)
        print(f"[{arm.name}] " + json.dumps(r), flush=True)
    finally:
        server.stop()
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
    return arm


def arms_for(args: argparse.Namespace) -> list[Arm]:
    dim = args.dim
    text = args.text_bytes
    arms = [
        Arm("eager", "eager", dim, text, False, None),
        Arm("lexical", "lexical", dim, text, False, None),
        Arm("lexical_long_text", "lexical", dim, args.long_text_bytes, False, None),
        Arm("lexical_snapshot", "lexical", dim, text, True, NO_GRAPH_ROWS),
        Arm("lexical_graph", "lexical", dim, text, True, None),
        # Journal pages kept resident (the pre-release behaviour): the difference to
        # `lexical` is what page release returns to the OS.
        Arm("lexical_keep_pages", "lexical", dim, text, False, None, {"HMS_RELEASE_JOURNAL_PAGES": "0"}),
    ]
    for other in args.dims:
        if other != dim:
            arms.append(Arm(f"lexical_dim{other}", "lexical", other, text, False, None))
    base_env = dict(kv.split("=", 1) for kv in args.env)
    for arm in arms:
        arm.env = {**base_env, **arm.env}
    if args.no_encode_flag:
        # A binary without `--ingest-encode` (origin/main): every arm ingests eagerly, so the
        # hypervector difference is 0 by construction and the other components still hold.
        for arm in arms:
            arm.encode = "eager"
        arms = [a for a in arms if a.name != "lexical"]
    if args.arms:
        wanted = set(args.arms.split(","))
        arms = [a for a in arms if a.name in wanted]
    return arms


def derive(arms: dict[str, Arm], args: argparse.Namespace) -> dict:
    def reading(name: str, key: str) -> float | None:
        arm = arms.get(name)
        return None if arm is None else arm.readings.get(key)

    n = args.n
    out: dict = {"n": n, "dim": args.dim}
    eager = reading("eager", "rss_after_ingest_mb")
    base = "eager" if args.no_encode_flag else "lexical"
    lexical = reading(base, "rss_after_ingest_mb")
    idle = reading(base, "rss_idle_mb")
    if eager is not None and lexical is not None:
        out["hypervectors_mb"] = eager - lexical
    long_text = reading("lexical_long_text", "rss_after_ingest_mb")
    if long_text is not None and lexical is not None:
        extra = args.long_text_bytes - args.text_bytes
        out["text_tables_mb_per_extra_text_byte_x_n"] = (long_text - lexical) / max(1, extra * n) * 2**20
        out["text_tables_mb"] = long_text - lexical
    snap_before = reading("lexical_snapshot", "rss_after_ingest_mb")
    snap_after = reading("lexical_snapshot", "rss_after_maintain_mb")
    if snap_before is not None and snap_after is not None:
        out["snapshot_and_bbox_mb"] = snap_after - snap_before
    graph_after = reading("lexical_graph", "rss_after_maintain_mb")
    if graph_after is not None and snap_after is not None:
        out["graph_mb"] = graph_after - snap_after
    keep = reading("lexical_keep_pages", "rss_after_ingest_mb")
    if keep is not None and lexical is not None:
        out["journal_pages_released_mb"] = keep - lexical
    slopes = []
    for other in args.dims:
        if other == args.dim:
            continue
        rss_other = reading(f"lexical_dim{other}", "rss_after_ingest_mb")
        if rss_other is not None and lexical is not None:
            bytes_per_value = (lexical - rss_other) * 2**20 / (n * (args.dim - other))
            slopes.append({"other_dim": other, "bytes_per_value": bytes_per_value})
    if slopes:
        per_value = statistics.fmean(s["bytes_per_value"] for s in slopes)
        out["embedding_bytes_per_value_measured"] = per_value
        out["embedding_slopes"] = slopes
        out["embeddings_mb"] = per_value * n * args.dim / 2**20
        if idle is not None and lexical is not None:
            out["residual_mb"] = lexical - idle - out["embeddings_mb"]
    out["arithmetic"] = {
        "embeddings_f64_mb": n * args.dim * 8 / 2**20,
        "embeddings_f32_mb": n * args.dim * 4 / 2**20,
        "snapshot_matrix_copy_f32_mb_before_change": n * args.dim * 4 / 2**20,
        "snapshot_handles_mb_after_change": n * 32 / 2**20,
        "graph_estimate_mb": n * (args.dim + args.dim + 32 * 4 + 8) / 2**20,
        "bbox_nodes_f32_mb": (2 * ((n + 63) // 64) - 1) * args.dim * 8 / 2**20,
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hms-bin", required=True, help="hms-server binary (built with --features server)")
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--dims", type=lambda s: [int(x) for x in s.split(",")], default=[384, 128],
                        help="input dims whose RSS difference gives the bytes per stored value")
    parser.add_argument("--batch", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--text-bytes", type=int, default=12)
    parser.add_argument("--long-text-bytes", type=int, default=200)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--settle", type=float, default=0.5)
    parser.add_argument("--arms", default="", help="comma-separated subset of arm names")
    parser.add_argument("--label", default="", help="free text recorded in the output (e.g. the commit)")
    parser.add_argument("--keep", action="store_true", help="keep the data directories")
    parser.add_argument("--extra", nargs="*", default=[], help="extra hms-server flags for every arm")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="environment for every server (repeatable), e.g. HMS_RELEASE_JOURNAL_PAGES=0")
    parser.add_argument("--no-encode-flag", action="store_true",
                        help="the binary predates --ingest-encode (origin/main baseline): run every arm eagerly")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    binary = str(Path(args.hms_bin).resolve())
    if not os.access(binary, os.X_OK):
        print(f"not executable: {binary}", file=sys.stderr)
        return 2
    root = Path(tempfile.mkdtemp(prefix="hms-rss-"))
    results: dict[str, Arm] = {}
    try:
        for arm in arms_for(args):
            results[arm.name] = run_arm(binary, arm, args, root)
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
    derived = derive(results, args)
    document = {
        "schema": "hms-rss-breakdown/1",
        "label": args.label,
        "binary": binary,
        "platform": sys.platform,
        "method": {
            "n": args.n, "dim": args.dim, "dims": args.dims, "batch": args.batch, "seed": args.seed,
            "text_bytes": args.text_bytes, "long_text_bytes": args.long_text_bytes,
            "rss": f"median of {args.samples} ps -o rss= readings {args.settle}s apart",
            "extra_flags": args.extra,
        },
        "arms": {name: {"encode": a.encode, "dim": a.dim, "text_bytes": a.text_bytes, "maintain": a.maintain,
                        "graph_min_rows": a.graph_min_rows, "env": a.env, **a.readings} for name, a in results.items()},
        "components": derived,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=1, sort_keys=True)
    print(f"{'arm':22s} {'idle':>8s} {'ingest':>8s} {'maint':>8s} {'disk MB':>8s}")
    for name, arm in results.items():
        r = arm.readings
        maint = r.get("rss_after_maintain_mb")
        print(f"{name:22s} {r.get('rss_idle_mb', 0):8.1f} {r.get('rss_after_ingest_mb', 0):8.1f} "
              f"{(maint if maint is not None else float('nan')):8.1f} {r.get('data_dir_bytes_after_ingest', 0) / 2**20:8.1f}")
    for key, value in derived.items():
        if isinstance(value, (int, float)):
            print(f"  {key:45s} {value:12.3f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
