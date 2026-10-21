# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run an hms-bench suite (default heavy: full GloVe-100, SIFT-1M) on Modal, one container per shard.

  uv run --with modal modal run --detach benchmarks/competitive/modal_run.py --cap-usd 12
  uv run --with modal modal run --detach benchmarks/competitive/modal_run.py --cap-usd 30 \
      --shards 'ann:glove-100-angular;text:beir:scifact,beir:nfcorpus,memory'
  uv run --with modal modal run benchmarks/competitive/modal_run.py --only ann --datasets glove-100-angular

The working tree (tracked plus untracked, non-ignored files) is shipped to the container, built
there, and measured in place; the raw results come back and are written to
benchmarks/results/competitive/. Every row carries the hardware class `modal-x86_64-8vcpu-32g`
and is only ever compared with rows of that class. The container's cores are shared hardware
threads, so absolute numbers are indicative and ratios inside the file are the result.
Spend is bounded by --cap-usd: the run is killed when elapsed wall time at Modal's published
CPU and memory rates would pass the cap (split equally over the shards) or would reach the
platform timeout, and whatever finished is still returned. Each finished results file is also
kept on the volume under /results/<run>-<shard>/, with the shard's log under /logs/.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path("/")  # inside the container only the module file exists
CPU, MEM_GIB = 8, 32
USD_PER_HOUR = CPU * 0.0473 + MEM_GIB * 0.008  # published rates, as in benchmarks/public/ann_modal_confirm.py
HW_CLASS = "modal-x86_64-8vcpu-32g"

VOL = modal.Volume.from_name("hms-bench-run", create_if_missing=True)
app = modal.App("hms-bench-run")
image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("curl", "build-essential", "pkg-config", "time", "git", "openjdk-17-jdk-headless")
    .pip_install("uv==0.12.10")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.96.0")
)


def tree_tar() -> bytes:
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name in files:
            path = ROOT / name
            if path.is_file() and path.stat().st_size < 20 * 2**20 and not name.startswith(("benchmarks/results/", "docs/research/", "figures/")):
                tar.add(path, arcname=name)
    return buf.getvalue()


TIMEOUT_S = 23 * 3600


@app.function(image=image, volumes={"/vol": VOL}, cpu=CPU, memory=MEM_GIB * 1024, timeout=TIMEOUT_S)
def run(archive: bytes, suite: str, args: list[str], cap_usd: float, tag: str, revision: str) -> dict:
    started = time.time()
    work = Path("/work/src")
    work.mkdir(parents=True, exist_ok=True)
    (work / "src.tar").write_bytes(archive)
    subprocess.run(["tar", "xf", "src.tar"], cwd=work, check=True)
    # Results and the log live on the volume (committed in the background as they are written), so
    # a container lost to the platform timeout still leaves every finished file behind. Builds,
    # environments and work directories stay container-local: shards run in parallel, possibly on
    # different CPUs, and the source-built comparators are compiled for the CPU that runs them.
    results_dir = Path("/vol/results") / tag
    results_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PATH": f"/root/.cargo/bin:{os.environ['PATH']}", "HMS_BENCH_DATA": "/vol/data", "HMS_BENCH_HW_CLASS": HW_CLASS,
           "HMS_BENCH_WORK": "/tmp/hms-work", "HMS_BENCH_ENVS": "/tmp/hms-envs", "HMS_BENCH_RESULTS": str(results_dir),
           "HMS_BENCH_COMMIT": revision, "CARGO_TARGET_DIR": "/tmp/target", "CARGO_BUILD_JOBS": str(CPU), "PYTHONUNBUFFERED": "1"}
    # The profile the harness runs and measures (runner.HMS_PROFILE).
    subprocess.run(["cargo", "build", "--profile", "dist", "--bin", "public-bench", "--example", "competitive_text"], cwd=work, env=env, check=True)
    # The cap stops the run gracefully only if it falls before the platform's hard timeout.
    budget_s = min(cap_usd / USD_PER_HOUR * 3600, TIMEOUT_S - 1800) - (time.time() - started)
    log = Path("/vol/logs") / f"{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    status = "completed"
    with log.open("w") as handle:
        try:
            subprocess.run(["uv", "run", "--script", "benchmarks/competitive/hms-bench", "run", "--suite", suite, *args], cwd=work, env=env,
                           stdout=handle, stderr=subprocess.STDOUT, timeout=max(budget_s, 60), check=True)
        except subprocess.TimeoutExpired:
            status = f"stopped at the ${cap_usd:.2f} cap or the platform timeout"
        except subprocess.CalledProcessError as error:
            status = f"failed with exit {error.returncode}"
    VOL.commit()
    produced = [*results_dir.glob(f"{suite}-*.json"), *Path("/vol/data/competitive/smoke").glob("smoke-*.json")]
    results = {p.name: p.read_text() for p in produced}
    return {"tag": tag, "args": args, "status": status, "elapsed_s": time.time() - started,
            "usd": (time.time() - started) / 3600 * USD_PER_HOUR, "results": results, "log_tail": log.read_text()[-4000:]}


def shard_args(spec: str) -> list[str]:
    """`ann:glove-100-angular` or `text:beir:scifact,memory` -> hms-bench run arguments."""
    only, _, datasets = spec.partition(":")
    return ["--only", only] + (["--datasets", *datasets.split(",")] if datasets else [])


@app.local_entrypoint()
def main(suite: str = "heavy", cap_usd: float = 12.0, only: str = "both", datasets: str = "", shards: str = "", rounds: int = 0):
    """`rounds` overrides the suite's serve rounds. `shards`: `;`-separated `only:datasets` specs, each run on its own container in parallel with
    an equal share of the cap; without it, one container runs `--only`/`--datasets`."""
    specs = shards.split(";") if shards else [f"{only}:{datasets}"]
    run_id = time.strftime("%Y%m%dT%H%M%S")
    archive = tree_tar()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    revision = head + ("+uncommitted" if dirty else "")
    extra = ["--rounds", str(rounds)] if rounds else []
    calls = [(archive, suite, shard_args(spec) + extra, cap_usd / len(specs), f"{run_id}-{i}", revision) for i, spec in enumerate(specs)]
    dest = ROOT / "benchmarks/results/competitive"
    for out in run.starmap(calls, return_exceptions=True):
        if isinstance(out, BaseException):
            print("shard failed:", repr(out)[-400:])
            continue
        for name, text in out["results"].items():
            target = Path.home() / ".cache/hms-bench/competitive/smoke" if name.startswith("smoke-") else dest
            target.mkdir(parents=True, exist_ok=True)
            (target / name).write_text(text)
        print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
        print("wrote", sorted(out["results"]))
