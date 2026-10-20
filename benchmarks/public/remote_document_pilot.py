# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Run a frozen synthetic document pilot on an existing Linux SSH host."""

def remote_run(source, baseline, manifest, work_root):
    import hashlib
    import io
    import json
    import os
    from pathlib import Path
    import platform
    import shutil
    import signal
    import subprocess
    import tarfile
    import time

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    if sha(source) != manifest["payload_sha256"] or sha(baseline) != manifest["baseline_sha256"]:
        raise ValueError("payload hash mismatch")
    root = Path(work_root)
    work, evidence = root / "source", root / "evidence"
    work.mkdir()
    evidence.mkdir()
    receipt = {"status": "started", "started_unix": time.time(), "proposal": manifest,
               "machine": platform.machine(), "platform": platform.platform(),
               "python": platform.python_version(), "initial_load": os.getloadavg(),
               "allowed_cpus": sorted(os.sched_getaffinity(0))}

    def run(command, name, timeout, cwd=work, env=None):
        started = time.monotonic()
        with (evidence / f"{name}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise
        receipt.setdefault("commands", []).append({"argv": command, "exit_code": code,
                                                     "seconds": time.monotonic() - started})
        if code:
            raise RuntimeError(f"command failed: {name}, exit={code}")

    def source_hashes():
        return {name: sha((work / name).read_bytes()) for name in manifest["files"]}

    try:
        for name in ("/proc/cpuinfo", "/proc/meminfo", "/proc/self/cgroup",
                     "/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu.stat",
                     "/sys/fs/cgroup/memory.max"):
            path = Path(name)
            if path.is_file():
                (evidence / name.strip("/").replace("/", "_")).write_bytes(path.read_bytes())
        if receipt["machine"] != "x86_64" or receipt["python"] != "3.13.9":
            raise RuntimeError("unexpected architecture or Python version")
        if receipt["initial_load"][0] >= 3:
            raise RuntimeError("host unsuitable: initial load misses frozen <3 gate")
        with tarfile.open(fileobj=io.BytesIO(source), mode="r:gz") as archive:
            archive.extractall(work, filter="data")
        if source_hashes() != manifest["files"]:
            raise ValueError("extracted source mismatch")
        run(["rustc", "-Vv"], "rustc", 30)
        if not (evidence / "rustc.log").read_text().startswith("rustc 1.96.0 "):
            raise ValueError("wrong Rust version")
        env = dict(os.environ, CARGO_TARGET_DIR=str(root / "target"), CARGO_BUILD_JOBS="4",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="1")
        originals = {name: (work / name).read_bytes() for name in
                     ("src/core/documents.rs", "src/core/engine/mod.rs")}
        with tarfile.open(fileobj=io.BytesIO(baseline), mode="r:gz") as archive:
            if set(archive.getnames()) != set(originals):
                raise ValueError("unexpected baseline overlay")
            archive.extractall(work, filter="data")
        bins = {}
        for arm in ("baseline", "candidate"):
            if arm == "candidate":
                for name, data in originals.items():
                    (work / name).write_bytes(data)
            before = source_hashes()
            run(["cargo", "build", "--release", "--locked", "--bin", "document-search-bench"],
                f"build-{arm}", 1200, env=env)
            if before != source_hashes():
                raise ValueError("source changed while building")
            binary = root / arm
            shutil.copy2(root / "target/release/document-search-bench", binary)
            bins[arm] = str(binary)
            (evidence / f"build-{arm}.json").write_text(json.dumps({"source_sha256": before,
                "binary_sha256": sha(binary.read_bytes()), "rustc": (evidence / "rustc.log").read_text()}, indent=2))
        if source_hashes() != manifest["files"]:
            raise ValueError("candidate source was not restored")
        cooldown = []
        for _ in range(10):
            load = os.getloadavg()
            cooldown.append(load)
            if load[0] < 3:
                break
            time.sleep(10)
        receipt["post_build_load"] = cooldown
        if cooldown[-1][0] >= 3:
            raise RuntimeError("host unsuitable: post-build load misses frozen <3 gate")
        cpu = str(receipt["allowed_cpus"][0])
        run(["taskset", "-c", cpu, "python", "benchmarks/public/document_search_compare.py",
             "--baseline", bins["baseline"], "--candidate", bins["candidate"],
             "--out", str(evidence / "paired"), "--lock-path", "/tmp/hms-benchmark-timing.lock"],
            "paired", 900, env=env)
        run(["python", "benchmarks/public/check_document_search_compare.py", str(evidence / "paired")],
            "independent-check", 120, env=env)
        receipt["status"] = "checked diagnostic; no superiority claim"
    except Exception as error:
        receipt["status"] = "failed or unsuitable; no accepted timings"
        receipt["failure"] = f"{type(error).__name__}: {error}"
    receipt["finished_unix"] = time.time()
    receipt["final_load"] = os.getloadavg()
    (evidence / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    files = {str(path.relative_to(evidence)): sha(path.read_bytes())
             for path in evidence.rglob("*") if path.is_file()}
    (evidence / "hashes.json").write_text(json.dumps(files, indent=2) + "\n")
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for path in sorted(evidence.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(evidence)), recursive=False)
    if output.tell() > 64 * 1024 * 1024:
        raise ValueError("evidence exceeds 64 MiB return budget")
    return output.getvalue()

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to replace an existing run")
    args.work_root.mkdir(exist_ok=False)
    data = remote_run(args.source.read_bytes(), args.baseline.read_bytes(),
                      json.loads(args.manifest.read_text()), args.work_root)
    with args.out.open("xb") as output:
        output.write(data)
