# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""x86 kernel microbenchmark on Modal CPU containers, with a CPU-flag probe (platform area).

  uv run --with modal modal run benchmarks/public/platform_modal_avx512.py --probes 12 --cap-usd 2

Modal does not let a function pick its CPU model, so this builds the `kernels` test binary once
in the image (rustc, no target-cpu flags, so every kernel is chosen by runtime detection) and
fans `--probes` small containers out over it. Each probe records its /proc/cpuinfo model and
flags, runs the bit-exact equivalence tests for every kernel its CPU supports, then the
`dot_kernel_microbench` kill test (ns per call, cold and hot working sets, dim 128 and 256).
A probe whose CPU lacks AVX-512 still reports the AVX2 rows, so the table always has a
same-container baseline. Spend is bounded by --cap-usd at the published rate.
"""
from __future__ import annotations

import io
import json
import subprocess
import tarfile
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent.parent
CPU, MEM_GIB = 2, 4
USD_PER_HOUR = CPU * 0.0473 + MEM_GIB * 0.008


def source_tar() -> bytes:
    """The working tree as it is now (tracked files plus uncommitted edits), as a tar."""
    names = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPO, check=True, capture_output=True,
    ).stdout.decode().split("\0")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name in filter(None, names):
            path = REPO / name
            if path.is_file() and path.stat().st_size < 20_000_000:
                tar.add(path, arcname=name)
        lock = REPO / "Cargo.lock"
        if lock.exists():
            tar.add(lock, arcname="Cargo.lock")
    return buf.getvalue()


SRC = Path("/tmp/hms-platform-src.tar")
if modal.is_local():
    SRC.write_bytes(source_tar())

app = modal.App("hms-platform-avx512")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "pkg-config")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.96.0")
    .add_local_file(str(SRC), "/src.tar", copy=True)
    .run_commands(
        "mkdir -p /work /opt && tar xf /src.tar -C /work",
        "cd /work && /root/.cargo/bin/cargo test --release --lib --no-run --message-format=json 2>/dev/null"
        " | grep -o '\"executable\":\"[^\"]*\"' | head -1 | cut -d'\"' -f4 > /opt/bin.path",
        "cp $(cat /opt/bin.path) /opt/kbench && /opt/kbench --list | tail -3",
    )
)


@app.function(image=image, cpu=CPU, memory=MEM_GIB * 1024, timeout=1800, max_containers=12)
def probe(i: int) -> dict:
    started = time.time()
    info = Path("/proc/cpuinfo").read_text().splitlines()
    model = next((l.split(":", 1)[1].strip() for l in info if l.startswith("model name")), "?")
    flags = next((l.split(":", 1)[1].split() for l in info if l.startswith("flags")), [])
    wanted = [f for f in flags if f.startswith("avx") or f in ("bmi2", "fma", "f16c")]
    out = Path(f"/tmp/kb-{i}.json")
    tests = subprocess.run(
        ["/opt/kbench", "qgraph::kernels", "--test-threads=1"], text=True, capture_output=True
    )
    bench = subprocess.run(
        ["/opt/kbench", "dot_kernel_microbench", "--ignored", "--nocapture", "--test-threads=1"],
        text=True, capture_output=True, env={"HMS_PLATFORM_BENCH_OUT": str(out)},
    )
    return {
        "probe": i, "cpu_model": model, "flags": wanted,
        "avx512bw": "avx512bw" in flags, "avx512vnni": "avx512_vnni" in flags or "avx512vnni" in flags,
        "avx512vl": "avx512vl" in flags,
        "tests_exit": tests.returncode, "tests_tail": tests.stdout[-600:],
        "bench_exit": bench.returncode, "bench_stderr_tail": bench.stderr[-300:] if bench.returncode else "",
        "results": json.loads(out.read_text()) if out.exists() else None,
        "wall_s": round(time.time() - started, 1),
    }


@app.local_entrypoint()
def main(probes: int = 12, cap_usd: float = 2.0,
         out: str = "benchmarks/results/platform_x86_modal_microbench_v1.json"):
    # Worst case: every probe runs ten minutes. Refuse a fan-out that could pass the cap.
    worst = probes * (600 / 3600) * USD_PER_HOUR
    if worst > cap_usd:
        raise SystemExit(f"{probes} probes could cost ${worst:.2f} > cap ${cap_usd}")
    rows = list(probe.map(range(probes)))
    spent = sum(r["wall_s"] for r in rows) / 3600 * USD_PER_HOUR
    report = {
        "schema_version": 1, "category": "platform_kernels", "date": time.strftime("%Y-%m-%d"),
        "protocol": "dot_kernel_microbench (kernels.rs): 400k random rows, 64-row prefetched batches (cold) "
                    "and 64 L1-resident rows (hot); best of 7 passes of 2^20 calls; fn-pointer calls for all variants",
        "harness": "benchmarks/public/platform_modal_avx512.py", "cap_usd": cap_usd,
        "spend_usd_estimate": round(spent, 3), "probes": rows,
    }
    Path(out).write_text(json.dumps(report, indent=1) + "\n")
    for r in rows:
        print(r["probe"], r["cpu_model"], "avx512bw" if r["avx512bw"] else "-",
              "vnni" if r["avx512vnni"] else "-", "tests", r["tests_exit"], "bench", r["bench_exit"])
