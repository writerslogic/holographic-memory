# SPDX-License-Identifier: Apache-2.0
"""Graph-parameter sweep for the vertex index on glove-100-angular, on one Modal x86 container.

  modal volume put hms-ann-confirm ~/.cache/hms-bench/glove-100-angular /glove-100-angular   # once
  uv run --with modal modal run --detach benchmarks/public/ann_modal_graph.py --rev HEAD --cap-usd 5

Every arm is the same `public-bench` binary (built from `git archive <rev>`) with the accepted
search configuration (4-bit traversal codes, 8-bit residual, aligned rows, BFS reorder, 3-byte
ids, rerank 64) and a different graph: degree and construction ef. Builds use
`--quantized-search`. Queries are a 2,000-vector holdout of the train split (tuning mode; the
test queries are not read). Round 1 builds and saves each graph; round 2 reloads it and times
the search again, so the paired numbers are the round-2 rows. The container's cores are shared
hardware threads: ratios between arms are the result, absolute QPS is indicative.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent.parent if len(Path(__file__).resolve().parents) > 2 else Path("/")
CPU, MEM_GIB = 8, 32
USD_PER_HOUR = CPU * 0.0473 + MEM_GIB * 0.008  # Modal's published rates, as in ann_modal_confirm.py
SEARCH = [
    "--index", "vertex", "--vertex-bits", "4", "--residual", "--residual-bits", "8",
    "--align-rows", "--reorder", "--id-bytes", "3", "--rerank", "64",
    "--ef", "64,96,128,192,256,320,384,512,736,1024", "--repeats", "5",
    "--holdout", "2000", "--max-load", "1000", "--max-wait-secs", "0", "--quantized-search",
]
# name -> extra arguments; an arm may name the graph cache it shares with another arm
# ("cache": the other arm's name) when its arguments do not change the graph structure.
ARMS = {
    "d32e200": {"args": ["--degree", "32", "--build-ef", "200"]},
    "d32e500": {"args": ["--degree", "32", "--build-ef", "500"]},
    "d48e400": {"args": ["--degree", "48", "--build-ef", "400"]},
    "d64e500": {"args": ["--degree", "64", "--build-ef", "500"]},
}

VOL = modal.Volume.from_name("hms-ann-confirm", create_if_missing=True)
app = modal.App("hms-ann-graph")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "pkg-config", "time")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.96.0")
)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kw)


@app.function(image=image, volumes={"/vol": VOL}, cpu=CPU, memory=MEM_GIB * 1024, timeout=6 * 3600)
def run(archive: bytes, lock: bytes, cap_usd: float, arms: dict) -> dict:
    started = time.time()
    env = {"PATH": "/root/.cargo/bin:/usr/bin:/bin", "HOME": "/root", "CARGO_BUILD_JOBS": str(CPU)}
    cpuinfo = Path("/proc/cpuinfo").read_text().splitlines()
    cpu_model = next((l.split(":", 1)[1].strip() for l in cpuinfo if l.startswith("model name")), "?")
    flags = next((l for l in cpuinfo if l.startswith("flags")), "")
    out = Path("/vol/graph-out")
    out.mkdir(exist_ok=True)

    def spent() -> float:
        return (time.time() - started) / 3600 * USD_PER_HOUR

    src = Path("/work/tip")
    src.mkdir(parents=True, exist_ok=True)
    (src / "src.tar").write_bytes(archive)
    sh(["tar", "xf", "src.tar"], cwd=src)
    (src / "Cargo.lock").write_bytes(lock)
    target = Path("/vol/target-graph")
    sh(["cargo", "build", "--release", "--bin", "public-bench"], cwd=src, env=env | {"CARGO_TARGET_DIR": str(target)})
    binary = target / "release" / "public-bench"
    VOL.commit()

    rows = []
    for rnd in (1, 2):
        for name, arm in arms.items():
            extra = arm["args"]
            if spent() > cap_usd * 0.9:
                rows.append({"arm": name, "round": rnd, "skipped": "spend cap"})
                continue
            cache = f"/vol/graph-cache-{arm.get('cache', name)}.graph"
            result = out / f"{name}-r{rnd}.json"
            cmd = ["/usr/bin/time", "-v", str(binary), "ann-qgraph", "--data", "/vol/glove-100-angular",
                   "--out", str(result), "--graph-cache", cache, *SEARCH, *extra]
            t0 = time.time()
            proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
            elapsed = time.time() - t0
            peak_kb = next((int(l.split(":")[1]) for l in proc.stderr.splitlines() if "Maximum resident" in l), None)
            rows.append({"arm": name, "round": rnd, "exit": proc.returncode, "wall_s": round(elapsed, 1),
                         "peak_rss_kib": peak_kb, "cold": rnd == 1, "stderr_tail": proc.stderr[-400:] if proc.returncode else "",
                         "result": json.loads(result.read_text()) if result.exists() else None})
            (out / "rows.json").write_text(json.dumps(rows))
            VOL.commit()
    report = {"cpu_model": cpu_model, "avx2": " avx2" in flags, "avx512f": " avx512f" in flags,
              "search": SEARCH, "arms": arms, "cpu": CPU, "mem_gib": MEM_GIB, "wall_s": round(time.time() - started, 1),
              "spent_usd_estimate": round(spent(), 2), "rows": rows}
    (out / "report.json").write_text(json.dumps(report))
    VOL.commit()
    return report


@app.local_entrypoint()
def main(rev: str = "HEAD", cap_usd: float = 5.0, out: str = "benchmarks/results/ann_modal_graph_v1.raw.json", arms: str = ""):
    """`arms`: a JSON object in the shape of ARMS to run instead of the default arms."""
    chosen = json.loads(arms) if arms else ARMS
    archive = subprocess.run(["git", "archive", "--format=tar", rev], cwd=REPO, check=True, capture_output=True).stdout
    report = run.remote(archive, (REPO / "Cargo.lock").read_bytes(), cap_usd, chosen)
    Path(out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("cpu_model", "avx2", "avx512f", "wall_s", "spent_usd_estimate")}))
    for row in report["rows"]:
        res = row.get("result")
        if not res:
            print(row["arm"], row["round"], row.get("skipped") or row.get("exit"))
            continue
        best = {}
        for sweep in res["sweep"]:
            for target in (0.9, 0.95, 0.99):
                if sweep["recall_at_10"] >= target and (target not in best or sweep["qps_single_thread"] > best[target][0]):
                    best[target] = (sweep["qps_single_thread"], sweep["recall_at_10"], sweep["mean_exact_evals_per_query"], sweep["params"]["ef"])
        print(row["arm"], f"r{row['round']}", f"build {res['build']['build_secs']:.0f}s", f"bytes {res['index_bytes']}",
              {t: f"{q:.0f} qps @ {r:.4f} ({e:.0f} evals, ef {ef})" for t, (q, r, e, ef) in best.items()})
