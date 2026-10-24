import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import run_glove_local_diagnosis as base
import run_glove_prepared_diagnosis as monitor


ROOT = base.ROOT.with_name("vector-glove-batch-pool-checks-v1")
NATIVE = "src/core/qgraph/vertex.rs"


def main(expected):
    os.umask(0o077)
    ROOT.mkdir()
    monitor.ROOT = ROOT
    if base.sha(base.TREE / NATIVE) != expected:
        raise ValueError("candidate differs from reviewed source")
    original = json.loads((base.ROOT / "source-manifest.json").read_text())
    for name, digest in original["native"].items():
        if name != NATIVE and base.sha(base.TREE / name) != digest:
            raise ValueError(f"unreviewed native change: {name}")
    if base.sha(base.TREE / "Cargo.lock") != original["harness"]["Cargo.lock"]:
        raise ValueError("lockfile changed")
    env = {key: os.environ[key] for key in ["PATH", "HOME", "TMPDIR", "RUSTUP_HOME", "CARGO_HOME", "DEVELOPER_DIR", "SDKROOT"] if key in os.environ}
    env.update({"CARGO_TARGET_DIR": str(base.TREE / "target"), "CARGO_BUILD_JOBS": "1", "RAYON_NUM_THREADS": "1", "CARGO_PROFILE_TEST_OPT_LEVEL": "1", "CARGO_PROFILE_TEST_DEBUG": "0", "CARGO_PROFILE_DEV_OPT_LEVEL": "1", "CARGO_PROFILE_DEV_DEBUG": "0"})
    commands = {
        "tests": ["cargo", "test", "--offline", "--locked", "--lib", "--features", "experimental", "core::qgraph", "--", "--test-threads=1"],
        "clippy": ["cargo", "clippy", "--offline", "--locked", "--lib", "--features", "experimental", "--", "-D", "warnings"],
    }
    base.save(ROOT / "plan.json", {"scope": "Focused vector candidate regression and lint checks; no benchmark result", "base": base.SOURCE, "candidate_vertex_sha256": expected, "commands": commands, "limits": {"per_command_s": 600, "overall_s": 900, "sampled_process_tree_rss_gib": 4, "build_jobs": 1, "test_threads": 1}, "memory_scope": "Compiler/test combined child-tree RSS sampled approximately200ms; not strict peak or unique memory", "environment": {k: v for k, v in env.items() if k.startswith(("CARGO_", "RAYON_"))}, "lock": str(Path.home() / "hms-timing.lock")})
    for name in [NATIVE, "Cargo.toml", "Cargo.lock", "benchmarks/public/run_glove_candidate_checks.py", "benchmarks/public/run_glove_prepared_diagnosis.py"]:
        destination = ROOT / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base.TREE / name, destination)
    (ROOT / "candidate.patch").write_bytes(subprocess.check_output(["git", "diff", "--", NATIVE], cwd=base.TREE))
    lock = (Path.home() / "hms-timing.lock").open("a+")
    base.fcntl.flock(lock, base.fcntl.LOCK_EX | base.fcntl.LOCK_NB)
    try:
        deadline = time.monotonic() + 900
        with Path("/Volumes/A/hms-coordination/RUN-ALL.md").open("a") as coordination:
            coordination.write(f"\n{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} Vector parent acquired local timing flock for opt-in batch-pool candidate focused qgraph regression/lint; plan {ROOT / 'plan.json'},4GiB sampled child-tree RSS/600s command/900s total,one buildjob/testthread/Rayonthread. No competitor/final evaluation or remote action.\n")
        for name, command in commands.items():
            monitor.run(name, command, env, 600, deadline, 4)
        output = (ROOT / "tests.stdout").read_text()
        if "batch_pool_matches_sequential_insertion ... ok" not in output or "batch_pool_preserves_search_results_work_and_reopening ... ok" not in output:
            raise ValueError("required candidate regression tests missing")
        if base.sha(base.TREE / NATIVE) != expected:
            raise ValueError("source changed during checks")
        base.save(ROOT / "complete.json", {"status": "passed-focused-vector-checks", "candidate_vertex_sha256": expected, "tests_summary": [line for line in output.splitlines() if line.startswith("test result:")], "tests_receipt_sha256": base.sha(ROOT / "tests.receipt.json"), "clippy_receipt_sha256": base.sha(ROOT / "clippy.receipt.json"), "full_repository_ci": False})
        print(json.dumps({"status": "passed-focused-vector-checks", "root": str(ROOT)}))
    finally:
        base.fcntl.flock(lock, base.fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--attempt", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    ROOT = base.ROOT.with_name(f"vector-glove-batch-pool-checks-v{args.attempt}")
    if not re.fullmatch("[0-9a-f]{64}", args.candidate_sha256):
        parser.error("invalid SHA256")
    if ROOT.exists():
        raise SystemExit("Run directory exists; preserve it and declare a new attempt")
    try:
        main(args.candidate_sha256)
    except BaseException as error:
        if ROOT.exists():
            base.save(ROOT / "failure.json", {"error": f"{type(error).__name__}: {error}", "automatic_retry": False})
        raise
