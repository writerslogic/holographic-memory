"""Run the qgraph VNNI kernel tests on a Modal CPU container that has AVX512-VNNI.

The EVEX (`avx2,avx512f,avx512vl,avx512vnni`) kernels in src/core/qgraph/vnni.rs and the VNNI
tier of src/core/qgraph/kernels.rs only execute on a CPU with AVX512-VNNI. Modal CPU containers
land on varying hosts, so the container first reads /proc/cpuinfo and exits nonzero before
building when the flag is missing; the entrypoint then asks for a new container, at most three
times. On a host with the flag it builds the library tests in release (runtime feature
detection, no target-cpu flags) and runs the VNNI and kernel tests on the auto-detected tier
and on every pinned tier (HMS_QGRAPH_KERNEL), then the existing VNNI microbenchmark.

    modal run benchmarks/vibe/modal_vnni_check.py --out vnni-check.log
"""

import subprocess
import sys
import time
from pathlib import Path

import modal

# The repository root; only read on the local side (the container imports this file from /root).
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/")
SRC = "/work/hms"
CPU, MEM_GIB = 4, 8
USD_CORE_S, USD_GIB_S = 0.0000131, 0.00000222  # Modal published rates, as in modal_vibe.py
KEEP_DIRS = ("src/", "benches/", "tests/", "examples/", "proptest-regressions/")
KEEP_FILES = {"Cargo.toml", "Cargo.lock", "build.rs", "README.md"}


def ignore(path: Path) -> bool:
    """Ship only what `cargo test --lib` reads: the crate sources, its declared targets and
    the Rust files under benchmarks/ (one bin target lives there), never results or target/."""
    p = path.as_posix()
    if p in KEEP_FILES or p.startswith(KEEP_DIRS):
        return False
    if p.startswith("benchmarks/") and p.endswith(".rs"):
        return False
    # Keep directories that can hold kept files so the walk descends into them.
    return not any(k.startswith(p + "/") for k in KEEP_DIRS) and p not in ("benchmarks", "src", "")


image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install("build-essential", "curl", "ca-certificates", "pkg-config")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable")
    .add_local_dir(ROOT, SRC, ignore=ignore)
)
app = modal.App("hms-vnni-check")

STEPS = [
    ("vnni encodings, auto tier", {}, "qgraph::vnni"),
    ("kernels, auto tier", {}, "qgraph::kernels::tests"),
    ("kernels, pinned vnni", {"HMS_QGRAPH_KERNEL": "vnni"}, "qgraph::kernels::tests"),
    ("kernels, pinned avxvnni", {"HMS_QGRAPH_KERNEL": "avxvnni"}, "qgraph::kernels::tests"),
    ("kernels, pinned avx2", {"HMS_QGRAPH_KERNEL": "avx2"}, "qgraph::kernels::tests"),
    ("kernels, pinned scalar", {"HMS_QGRAPH_KERNEL": "scalar"}, "qgraph::kernels::tests"),
    ("construction kernels, auto tier", {}, "qgraph::build_kernels"),
]
BENCHES = [
    ("vnni microbench, auto tier", {}, "microbench_dot_kernels"),
    ("vnni microbench, pinned avx2", {"HMS_QGRAPH_KERNEL": "avx2"}, "microbench_dot_kernels"),
]


@app.function(image=image, cpu=CPU, memory=MEM_GIB * 1024, timeout=40 * 60)
def check() -> str:
    import os

    log: list[str] = []

    def say(s: str) -> None:
        print(s, flush=True)
        log.append(s)

    info = Path("/proc/cpuinfo").read_text().split("\n\n")[0]
    fields = dict(line.split(":", 1) for line in info.splitlines() if ":" in line)
    fields = {k.strip(): v.strip() for k, v in fields.items()}
    flags = set(fields.get("flags", "").split())
    say(f"cpu: {fields.get('model name')} (family {fields.get('cpu family')}, model {fields.get('model')}, "
        f"stepping {fields.get('stepping')}), {os.cpu_count()} logical CPUs")
    say("flags: " + " ".join(sorted(f for f in flags if "vnni" in f or f.startswith("avx512") or f == "avx2")))
    if "avx512_vnni" not in flags:
        say("avx512_vnni missing; not building")
        sys.exit(3)

    env = dict(os.environ, PATH=f"/root/.cargo/bin:{os.environ['PATH']}", CARGO_TERM_COLOR="never")
    say(subprocess.run(["rustc", "-V"], env=env, capture_output=True, text=True).stdout.strip())

    def cargo(args: list[str], extra: dict) -> int:
        r = subprocess.run(["cargo", *args], cwd=SRC, env={**env, **extra}, capture_output=True, text=True)
        out = "\n".join(line for line in (r.stdout + r.stderr).splitlines()
                        if not line.lstrip().startswith(("Compiling", "Downloaded", "Downloading")))
        say(out)
        return r.returncode

    t = time.time()
    rc = cargo(["test", "--release", "--lib", "--no-run"], {})
    say(f"build exit {rc} in {time.time() - t:.0f} s")
    if rc:
        return "\n".join(log)
    failed = []
    for name, extra, filt in STEPS:
        say(f"=== {name}: {filt} {extra}")
        if cargo(["test", "--release", "--lib", filt, "--", "--nocapture"], extra):
            failed.append(name)
    for name, extra, filt in BENCHES:
        say(f"=== {name} (development measurement): {filt} {extra}")
        cargo(["test", "--release", "--lib", filt, "--", "--ignored", "--nocapture", "--test-threads=1"], extra)
    say("FAILED: " + ", ".join(failed) if failed else "ALL STEPS PASSED")
    return "\n".join(log)


@app.local_entrypoint()
def main(out: str = "vnni-check.log", attempts: int = 3):
    spent = 0.0
    for attempt in range(1, attempts + 1):
        t = time.time()
        try:
            text = check.remote()
        except Exception as e:  # the container exits nonzero on a CPU without avx512_vnni
            text = None
            print(f"attempt {attempt}: {type(e).__name__}: {e}")
        secs = time.time() - t
        spent += secs * (CPU * USD_CORE_S + MEM_GIB * USD_GIB_S)
        print(f"attempt {attempt}: {secs:.0f} s, upper-bound spend so far ${spent:.3f}")
        if text is not None:
            Path(out).write_text(text + f"\nupper-bound spend ${spent:.3f} over {attempt} container(s)\n")
            print(f"log written to {out}")
            return
    print("no container with avx512_vnni")
    sys.exit(1)
