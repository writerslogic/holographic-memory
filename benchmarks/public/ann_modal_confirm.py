# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""x86 confirmation of the 1A kernel and build levers on Modal (1A, replaces the blocked GCE run).

  modal volume create hms-ann-confirm
  modal volume put hms-ann-confirm ~/.cache/hms-bench/glove-100-angular /glove-100-angular
  uv run --with modal modal run --detach benchmarks/public/ann_modal_confirm.py \
      --baseline a46108c --tip <sha> --cap-usd 10

Three arms on one container (8 reserved vCPU, 32 GiB, x86), each built from a `git archive` of
the named commit and run through `public-bench ann-qgraph` on glove-100-angular with
`--holdout 2000` (the 1.18M train vectors only; no test queries):

  A  the baseline commit (scalar x86 kernels, float-distance construction search)
  B  the tip commit without --quantized-search (AVX2 kernels, same build path as A)
  C  the tip commit with --quantized-search (the build lever)

The vertex index uses the configuration of the paired x86 glove rows (4-bit traversal codes,
8-bit residual, degree 32, build ef 200, aligned rows, BFS reorder, 3-byte ids, rerank 64). Build
seconds and peak RSS come from the cold run of each arm; search QPS from two interleaved rounds
(A B C A B C, graph cached on the volume for the second) of five repeats each. The container's
core count is reserved but its cores are shared hardware threads, so the row records the CPU
model and is compared only to the same-container arms, never to the accepted GCE numbers.
Spend is bounded by --cap-usd: the driver stops before a stage whose elapsed wall time at the
published rate would pass the cap.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent.parent if len(Path(__file__).resolve().parents) > 2 else Path("/")
CPU, MEM_GIB = 8, 32
USD_PER_HOUR = CPU * 0.0473 + MEM_GIB * 0.008  # Modal's published rates, as in longmemeval_modal.py
CONFIG = [
    "--index", "vertex", "--vertex-bits", "4", "--residual", "--residual-bits", "8",
    "--degree", "32", "--build-ef", "200", "--align-rows", "--reorder", "--id-bytes", "3",
    "--rerank", "64", "--ef", "128,192,256,320,384,512,736", "--repeats", "5",
    "--holdout", "2000", "--max-load", "1000", "--max-wait-secs", "0",
]

VOL = modal.Volume.from_name("hms-ann-confirm", create_if_missing=True)
app = modal.App("hms-ann-confirm")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "pkg-config", "time")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.96.0")
)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kw)


@app.function(image=image, volumes={"/vol": VOL}, cpu=CPU, memory=MEM_GIB * 1024, timeout=6 * 3600)
def run(archives: dict[str, bytes], lock: bytes, cap_usd: float) -> dict:
    started = time.time()
    env = {"PATH": "/root/.cargo/bin:/usr/bin:/bin", "HOME": "/root", "CARGO_BUILD_JOBS": str(CPU)}
    cpu_model = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    flags = next((l for l in open("/proc/cpuinfo") if l.startswith("flags")), "")
    out = Path("/vol/out")
    out.mkdir(exist_ok=True)

    def spent() -> float:
        return (time.time() - started) / 3600 * USD_PER_HOUR

    binaries = {}
    for tag, blob in archives.items():
        src = Path(f"/work/{tag}")
        src.mkdir(parents=True, exist_ok=True)
        (src / "src.tar").write_bytes(blob)
        sh(["tar", "xf", "src.tar"], cwd=src)
        (src / "Cargo.lock").write_bytes(lock)  # untracked: the resolution of the measuring checkout
        target = Path(f"/vol/target-{tag}")
        sh(["cargo", "build", "--release", "--bin", "public-bench"], cwd=src,
           env=env | {"CARGO_TARGET_DIR": str(target)})
        binaries[tag] = target / "release" / "public-bench"
        VOL.commit()

    arms = {
        "A": (binaries["baseline"], []),
        "B": (binaries["tip"], []),
        "C": (binaries["tip"], ["--quantized-search"]),
    }
    rows = []
    for rnd in (1, 2):
        for name, (binary, extra) in arms.items():
            if spent() > cap_usd * 0.9:
                rows.append({"arm": name, "round": rnd, "skipped": "spend cap"})
                continue
            cache = f"/vol/cache-{name}.graph"
            result = out / f"{name}-r{rnd}.json"
            cmd = ["/usr/bin/time", "-v", str(binary), "ann-qgraph", "--data", "/vol/glove-100-angular",
                   "--out", str(result), "--graph-cache", cache, *CONFIG, *extra]
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
              "config": CONFIG, "cpu": CPU, "mem_gib": MEM_GIB, "wall_s": round(time.time() - started, 1),
              "spent_usd_estimate": round(spent(), 2), "rows": rows}
    (out / "report.json").write_text(json.dumps(report))
    VOL.commit()
    return report


@app.local_entrypoint()
def main(baseline: str, tip: str, cap_usd: float = 10.0, out: str = "benchmarks/results/ann_modal_confirm_v1.raw.json"):
    archives = {}
    for tag, rev in (("baseline", baseline), ("tip", tip)):
        archives[tag] = subprocess.run(["git", "archive", "--format=tar", rev], cwd=REPO, check=True, capture_output=True).stdout
    report = run.remote(archives, (REPO / "Cargo.lock").read_bytes(), cap_usd)
    Path(out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("cpu_model", "avx2", "avx512f", "wall_s", "spent_usd_estimate")}))
