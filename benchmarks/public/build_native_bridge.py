# /// script
# requires-python = "==3.13.*"
# ///
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def sources(root):
    bridge = root / "benchmarks/public/native_bridge"
    paths = [bridge / "Cargo.toml", *bridge.rglob("*.rs"), *root.joinpath("src/core/qgraph").glob("*.rs")]
    lock = bridge / "Cargo.lock"
    if lock.exists():
        paths.append(lock)
    return [artifact(p) for p in sorted(paths)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    target = Path(args.target_dir).resolve()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    compiler = subprocess.check_output(["rustc", "--version"], text=True).strip()
    if not compiler.startswith("rustc 1.96.0 ") or platform.python_version() != "3.13.9":
        raise ValueError("native compiler/Python version differs")
    before = sources(root)
    command = ["cargo", "build", "--locked", "--release", "--manifest-path", "benchmarks/public/native_bridge/Cargo.toml"]
    environment = {"CARGO_TARGET_DIR": str(target), "PYO3_PYTHON": sys.executable,
                   "PYO3_BUILD_EXTENSION_MODULE": "1", "RUSTFLAGS": "-C target-cpu=native"}
    with Path(args.log).open("w") as log:
        completed = subprocess.run(command, cwd=root, env=os.environ | environment, stdout=log, stderr=subprocess.STDOUT)
    after = sources(root)
    binary = target / "release" / ("libhms_native_bridge.dylib" if platform.system() == "Darwin" else "libhms_native_bridge.so")
    generated = [artifact(p) for p in sorted(target.joinpath("release/build").glob("hms-native-bridge-*/out/qgraph/*.rs"))]
    linked = []
    if completed.returncode == 0 and platform.system() == "Darwin":
        output = subprocess.check_output(["otool", "-L", str(binary)], text=True)
        for line in output.splitlines()[1:]:
            name = line.strip().split(" (", 1)[0]
            if Path(name).name == binary.name:
                continue
            entry = {"link_name": name}
            if Path(name).is_file():
                entry["artifact"] = artifact(name)
            elif name.startswith(("/usr/lib/", "/System/Library/")):
                entry.update(artifact=None, storage="macOS dyld shared cache", os=platform.platform())
            else:
                raise ValueError(f"unresolved non-OS bridge dependency: {name}")
            linked.append(entry)
    result = {"schema_version": 1, "command": command, "cwd": str(root), "environment": environment,
              "compiler": compiler, "architecture": platform.machine(), "exit_code": completed.returncode,
              "source_before": before, "source_after": after, "source_identity_verified": before == after,
              "generated_sources": generated, "binary": artifact(binary) if completed.returncode == 0 else None,
              "linked_libraries": linked, "build_log": artifact(args.log), "benchmark_timing": False}
    Path(args.out).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"receipt": artifact(args.out), "exit_code": completed.returncode, "source_identity_verified": before == after}, sort_keys=True))
    if completed.returncode or before != after:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
