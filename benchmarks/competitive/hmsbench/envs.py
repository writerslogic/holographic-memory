# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""One pinned virtual environment per comparator, plus its install footprint.

`requirements/<name>.in` holds the pinned top-level packages; `requirements/<name>.txt` is the
fully resolved universal lock (`hms-bench lock`). The footprint (bytes and distribution count)
is measured by installing the lock alone into an empty target, so it is what a user adds, not
what the benchmark runtime (which also needs numpy) holds.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .common import CACHE, ROOT, dir_bytes, log

REQUIREMENTS = ROOT / "benchmarks/competitive/requirements"
# Source-built packages (hnswlib) compile with -march=native, so an environment is only valid on
# the CPU that built it; a fleet with mixed CPUs points this at machine-local storage.
ENV_ROOT = Path(os.environ["HMS_BENCH_ENVS"]) if os.environ.get("HMS_BENCH_ENVS") else CACHE / "competitive" / "envs"
PYTHON = "3.13"
# A package whose wheels do not cover CPython 3.13 gets its own interpreter; the version is
# recorded in every results file next to the package versions.
PYTHON_FOR = {"voyager": "3.12", "diskann": "3.11"}


def python_for(name):
    return PYTHON_FOR.get(name, PYTHON)

# Pins: library name -> (requirement lines, importable distributions that report the version).
PINS = {
    "hnswlib": ["hnswlib==0.8.0"],
    "faiss": ["faiss-cpu==1.12.0"],
    "usearch": ["usearch==2.26.5"],
    "annoy": ["annoy==1.17.3"],
    "sqlite-vec": ["sqlite-vec==0.1.9"],
    "lancedb": ["lancedb==0.40.0"],
    "chroma": ["chromadb==1.5.9"],
    "qdrant": ["qdrant-client==1.19.1"],
    "milvus": ["pymilvus==3.0.2", "milvus-lite==3.2.2"],
    "duckdb": ["duckdb==1.5.6"],
    "tantivy": ["tantivy==0.26.2"],
    "bm25s": ["bm25s==0.3.14", "PyStemmer==3.1.0"],
    "whoosh": ["whoosh-reloaded==2.7.5"],
    "stdlib": [],
    "scann": ["scann==1.4.2"],
    "ngt": ["ngt==2.3.12"],
    "pynndescent": ["pynndescent==0.6.0"],
    "voyager": ["voyager==2.2.2"],
    "diskann": ["diskannpy==0.7.0"],
}
BASE = ["numpy==2.5.4"]
# diskannpy 0.7.0 is built against NumPy 1.x.
BASE_FOR = {"diskann": ["numpy==1.25.0"]}
# Needed by the benchmark adapter only; excluded from the footprint.
RUNTIME_EXTRA = {"duckdb": ["pyarrow==21.0.0"]}


def write_inputs():
    REQUIREMENTS.mkdir(parents=True, exist_ok=True)
    for name, lines in PINS.items():
        (REQUIREMENTS / f"{name}.in").write_text("\n".join([*lines, *RUNTIME_EXTRA.get(name, []), *BASE_FOR.get(name, BASE)]) + "\n")
        (REQUIREMENTS / f"{name}.footprint.in").write_text("\n".join(lines) + "\n" if lines else "")


def lock(names=None):
    """Resolve the universal locks. Run on a networked machine; commit the .txt files."""
    write_inputs()
    for name in names or PINS:
        for suffix in ("", ".footprint"):
            src = REQUIREMENTS / f"{name}{suffix}.in"
            if not src.read_text().strip():
                (REQUIREMENTS / f"{name}{suffix}.txt").write_text("")
                continue
            subprocess.run(["uv", "pip", "compile", "--universal", "--python-version", python_for(name), "-q",
                            "--no-header", "-o", str(REQUIREMENTS / f"{name}{suffix}.txt"), str(src)], check=True)


def env_python(name):
    return ENV_ROOT / name / "bin" / "python"


def ensure(name):
    """Create the pinned environment for `name` if absent and return its interpreter path."""
    python = env_python(name)
    marker = ENV_ROOT / name / ".installed"
    lock_file = REQUIREMENTS / f"{name}.txt"
    wanted = (f"# python {PYTHON_FOR[name]}\n" if name in PYTHON_FOR else "") + (lock_file.read_text() if lock_file.exists() else "")
    if python.exists() and marker.exists() and marker.read_text() == wanted:
        return python
    log(f"creating environment {name}")
    shutil.rmtree(ENV_ROOT / name, ignore_errors=True)
    subprocess.run(["uv", "venv", "--python", python_for(name), "-q", str(ENV_ROOT / name)], check=True)
    if lock_file.exists() and lock_file.read_text().strip():
        subprocess.run(["uv", "pip", "install", "-q", "--python", str(python), "-r", str(lock_file)], check=True)
    marker.write_text(wanted)
    return python


def footprint(name):
    """Install size in bytes and number of distributions for `name` alone."""
    lock_file = REQUIREMENTS / f"{name}.footprint.txt"
    if not lock_file.exists() or not lock_file.read_text().strip():
        return {"bytes": 0, "distributions": 0, "ships_with_interpreter": True, "note": "standard library or already-installed dependency"}
    target = ENV_ROOT / f"{name}-footprint"
    shutil.rmtree(target, ignore_errors=True)
    subprocess.run(["uv", "pip", "install", "-q", "--python-version", python_for(name), "--target", str(target), "-r", str(lock_file)], check=True)
    dists = sorted(p.name for p in target.glob("*.dist-info"))
    size = dir_bytes(target)
    shutil.rmtree(target, ignore_errors=True)
    return {"bytes": size, "distributions": len(dists), "dists": [re.sub(r"\.dist-info$", "", d) for d in dists]}


def versions(name, packages):
    """Installed versions of `packages` as reported by the environment itself."""
    code = ("import importlib.metadata as m, json, platform, sys;"
            "print(json.dumps({**{p: m.version(p) for p in sys.argv[1:]}, 'python': platform.python_version()}))")
    out = subprocess.run([str(ensure(name)), "-I", "-c", code, *packages], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


if __name__ == "__main__":
    lock(sys.argv[1:] or None)
