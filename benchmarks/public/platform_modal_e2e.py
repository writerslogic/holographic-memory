# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""End to end AVX2 versus AVX512-VNNI kernel comparison on a Modal container (platform area).

  uv run --with modal modal run benchmarks/public/platform_modal_e2e.py --attempts 8 --cap-usd 3

Modal cannot pick a CPU model, so the function checks /proc/cpuinfo first and returns at once on
a container without AVX512-VNNI; the entrypoint retries up to --attempts times. On a capable
container it (1) runs the qgraph unit tests, which include the recall tests, once per kernel
tier through HMS_QGRAPH_KERNEL=scalar|avx2|vnni, (2) builds the glove-100-angular graph once
(cached on the volume, the config of ann_modal_confirm.py), and (3) runs
benchmarks/public/platform_paired.py with the arms `avx2` and `vnni` as the same binary under
two values of the kernel knob, at 1 and 4 search threads. The paired script fails the run when
the two arms disagree on recall or exact evaluations per query (the kernels are bit-exact, so
the searches must visit the same candidates). The volume `hms-ann-confirm` already holds
/glove-100-angular (see ann_modal_confirm.py).
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
CPU, MEM_GIB = 8, 32
USD_PER_HOUR = CPU * 0.0473 + MEM_GIB * 0.008
PROBE_USD = 0.02  # a skipped container: image pull plus a cpuinfo read


def source_tar() -> bytes:
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
        if (REPO / "Cargo.lock").exists():
            tar.add(REPO / "Cargo.lock", arcname="Cargo.lock")
    return buf.getvalue()


SRC = Path("/tmp/hms-platform-e2e-src.tar")
if modal.is_local():
    SRC.write_bytes(source_tar())

VOL = modal.Volume.from_name("hms-ann-confirm", create_if_missing=True)
app = modal.App("hms-platform-e2e")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential", "pkg-config")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.96.0")
    .add_local_file(str(SRC), "/src.tar", copy=True)
    .add_local_file(str(REPO / "benchmarks/public/platform_paired.py"), "/opt/platform_paired.py", copy=True)
    .run_commands(
        "mkdir -p /work /opt && tar xf /src.tar -C /work",
        "cd /work && /root/.cargo/bin/cargo build --release --features cli,mimalloc --bin public-bench && cp target/release/public-bench /opt/public-bench",
        "cd /work && /root/.cargo/bin/cargo test --release --lib --no-run --message-format=json 2>/dev/null"
        " | grep -o '\"executable\":\"[^\"]*\"' | head -1 | cut -d'\"' -f4 > /opt/bin.path",
        "cp $(cat /opt/bin.path) /opt/kbench && /opt/kbench --list | tail -1",
    )
)


@app.function(image=image, volumes={"/vol": VOL}, cpu=CPU, memory=MEM_GIB * 1024, timeout=3 * 3600)
def e2e(cap_usd: float, repeats: int) -> dict:
    started = time.time()
    info = Path("/proc/cpuinfo").read_text().splitlines()
    model = next((l.split(":", 1)[1].strip() for l in info if l.startswith("model name")), "?")
    flags = next((l.split(":", 1)[1].split() for l in info if l.startswith("flags")), [])
    have = {f: (f in flags) for f in ("avx2", "avx512f", "avx512bw", "avx512vl", "avx512_vnni")}
    if not (have["avx512_vnni"] and have["avx512vl"]):
        return {"skipped": True, "cpu_model": model, "have": have}
    spent = lambda: (time.time() - started) / 3600 * USD_PER_HOUR  # noqa: E731
    report: dict = {"skipped": False, "cpu_model": model, "have": have, "cpu": CPU, "mem_gib": MEM_GIB}
    gate = {}
    for tier in ("scalar", "avx2", "vnni"):
        p = subprocess.run(["/opt/kbench", "qgraph::", "--test-threads=4"], text=True, capture_output=True,
                           env={"HMS_QGRAPH_KERNEL": tier, "PATH": "/usr/bin:/bin"})
        summary = next((l for l in p.stdout.splitlines() if l.startswith("test result")), p.stdout[-300:])
        gate[tier] = {"exit": p.returncode, "summary": summary}
    report["qgraph_tests_by_tier"] = gate
    cache = "/vol/cache-platform.graph"
    data = "/vol/glove-100-angular"
    out = Path("/vol/out-platform")
    out.mkdir(exist_ok=True)
    # Build (or load) the graph once so the paired processes only read it.
    warm = subprocess.run(
        ["/opt/public-bench", "ann-qgraph", "--data", data, "--out", str(out / "warm.json"), "--graph-cache", cache,
         "--ef", "256", "--repeats", "1", "--index", "vertex", "--vertex-bits", "4", "--residual",
         "--residual-bits", "8", "--degree", "32", "--build-ef", "200", "--align-rows", "--reorder",
         "--id-bytes", "3", "--rerank", "64", "--holdout", "2000", "--max-load", "100000", "--max-wait-secs", "0"],
        text=True, capture_output=True)
    report["warm_exit"] = warm.returncode
    if warm.returncode:
        report["warm_stderr_tail"] = warm.stderr[-500:]
        return report
    VOL.commit()
    for threads in ("1", "4"):
        if spent() > cap_usd * 0.8:
            report[f"threads_{threads}"] = {"skipped": "spend cap"}
            continue
        dest = out / f"paired-t{threads}.json"
        p = subprocess.run(
            ["python3", "/opt/platform_paired.py", "--arm", "avx2=/opt/public-bench@HMS_QGRAPH_KERNEL=avx2",
             "--arm", "vnni=/opt/public-bench@HMS_QGRAPH_KERNEL=vnni", "--data", data, "--cache", cache,
             "--threads", threads, "--ef", "256,736", "--repeats", str(repeats), "--out", str(dest), "--tag", "modal"],
            text=True, capture_output=True)
        report[f"threads_{threads}"] = {
            "exit": p.returncode, "stderr_tail": p.stderr[-400:] if p.returncode else "",
            "result": json.loads(dest.read_text()) if dest.exists() else None,
        }
        VOL.commit()
    report["wall_s"] = round(time.time() - started, 1)
    report["spent_usd_estimate"] = round(spent(), 3)
    return report


@app.local_entrypoint()
def main(attempts: int = 8, cap_usd: float = 3.0, repeats: int = 21,
         out: str = "benchmarks/results/platform_x86_modal_e2e_v1.raw.json"):
    seen = []
    spent = 0.0
    for i in range(attempts):
        if spent + PROBE_USD > cap_usd:
            break
        r = e2e.remote(cap_usd - spent, repeats)
        spent += r.get("spent_usd_estimate", PROBE_USD)
        seen.append({"attempt": i, "cpu_model": r["cpu_model"], "skipped": r["skipped"]})
        if not r["skipped"]:
            r["attempts"] = seen
            r["driver_spend_usd_estimate"] = round(spent, 3)
            Path(out).write_text(json.dumps(r, indent=1) + "\n")
            print(json.dumps({k: r[k] for k in ("cpu_model", "qgraph_tests_by_tier", "spent_usd_estimate")}, indent=1))
            return
    Path(out).write_text(json.dumps({"skipped": True, "attempts": seen, "note": "no AVX512-VNNI container obtained"}, indent=1) + "\n")
    print("no AVX512-VNNI container in", len(seen), "attempts")
