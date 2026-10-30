# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Build isolated baseline/ablation probes; run the declared local comparison once."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import time

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OWNED = ROOT / "target/campaign-final-probes"
SOURCE = OWNED / "source"
BINARIES = OWNED / "binaries"
BASELINE = "a736af02"
TEST = "core::documents::search::campaign_probe::campaign_retrieval_probe"
TEMPORAL = "frozen_native_temporal_partitions_preserve_fixed_budgets_and_sources"
ENV = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / "target"), CARGO_BUILD_JOBS="4",
           CARGO_PROFILE_DEV_OPT_LEVEL="3", CARGO_PROFILE_TEST_OPT_LEVEL="3",
           CARGO_PROFILE_DEV_DEBUG="0", CARGO_PROFILE_TEST_DEBUG="0")


def archive(revision):
    if SOURCE.exists():
        shutil.rmtree(SOURCE)
    SOURCE.mkdir(parents=True)
    process = subprocess.Popen(["git", "archive", revision], cwd=ROOT, stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout, mode="r|") as stream:
        stream.extractall(SOURCE, filter="data")
    if process.wait():
        raise RuntimeError("git archive failed")
    if revision == "HEAD":
        changed = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=ROOT, text=True)
        for name in changed.splitlines():
            source, target = ROOT / name, SOURCE / name
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            elif target.is_file():
                target.unlink()
    shutil.copy2(HERE / "retrieval_probe.rs", SOURCE / "src/core/documents/campaign_probe.rs")
    search = SOURCE / "src/core/documents/search.rs"
    with search.open("a") as output:
        output.write('\n#[cfg(test)]\n#[path = "campaign_probe.rs"]\nmod campaign_probe;\n')
    # This diagnostic copy reports every frozen case, including known baseline failures.
    # Normal source tests retain their failure assertions and remain required gates.
    temporal = (ROOT / "tests/native_memory_workflow.rs").read_text()
    marker = '''    assert!(
        failures.is_empty(),
        "{} native evidence failures",
        failures.len()
    );'''
    assert temporal.count(marker) == 1
    temporal = temporal.replace(marker, "    // All failures are retained in the JSON evidence above.")
    (SOURCE / "tests/campaign_temporal_probe.rs").write_text(temporal)
    for fixture in ["native_memory_lifecycle_v1.json", "native_memory_evidence_v1.json", "memory_contract_cases_v1.json"]:
        shutil.copy2(ROOT / "tests/fixtures" / fixture, SOURCE / "tests/fixtures" / fixture)


def build(name, revision, disable_eligibility=False):
    archive(revision)
    if disable_eligibility:
        path = SOURCE / "src/core/documents/search.rs"
        source = path.read_text()
        old = "if slots.is_some() && dense.is_some() && documents.len() <= DENSE_SLOT_LIMIT {"
        assert source.count(old) == 1
        source = source.replace(old, "if false && slots.is_some() && dense.is_some() && documents.len() <= DENSE_SLOT_LIMIT {")
        old = "let slots = (store.slot_count <= DENSE_SLOT_LIMIT"
        assert source.count(old) == 1
        source = source.replace(old, "let slots = (false && store.slot_count <= DENSE_SLOT_LIMIT")
        path.write_text(source)
    command = ["cargo", "test", "--offline", "--locked", "--lib", "--test", "campaign_temporal_probe", "--no-run", "--message-format=json"]
    with (HERE / f"probe-build-{name}.log").open("w") as log:
        process = subprocess.Popen(command, cwd=SOURCE, env=ENV, stdout=subprocess.PIPE, stderr=log, text=True)
        executables = {}
        for line in process.stdout:
            record = json.loads(line)
            if record.get("reason") == "compiler-message":
                log.write(record["message"].get("rendered", line))
            if record.get("reason") == "compiler-artifact" and record.get("executable"):
                target_name = record["target"]["name"]
                if target_name == "campaign_temporal_probe":
                    executables["temporal"] = record["executable"]
                elif target_name == "holographic_memory" and record["profile"]["test"]:
                    executables["retrieval"] = record["executable"]
        if process.wait():
            raise RuntimeError(f"{name} compilation failed; see log")
    assert set(executables) == {"retrieval", "temporal"}, executables
    hashes = {}
    for target, executable in executables.items():
        destination = BINARIES / f"{name}-{target}"
        shutil.copy2(executable, destination)
        hashes[target] = hashlib.sha256(destination.read_bytes()).hexdigest()
    source_hash = hashlib.sha256()
    for path in sorted((SOURCE / "src").rglob("*.rs")) + [SOURCE / "Cargo.toml", SOURCE / "Cargo.lock"]:
        source_hash.update(str(path.relative_to(SOURCE)).encode())
        source_hash.update(b"\0")
        source_hash.update(path.read_bytes())
    return {"name": name, "revision": subprocess.check_output(["git", "rev-parse", revision], cwd=ROOT, text=True).strip(),
            "command": command, "binary_sha256": hashes, "source_sha256": source_hash.hexdigest(),
            "environment": {key:ENV[key] for key in ["CARGO_TARGET_DIR", "CARGO_BUILD_JOBS",
                "CARGO_PROFILE_DEV_OPT_LEVEL", "CARGO_PROFILE_TEST_OPT_LEVEL",
                "CARGO_PROFILE_DEV_DEBUG", "CARGO_PROFILE_TEST_DEBUG"]},
            "rustc":subprocess.check_output(["rustc","-Vv"],text=True),
            "search_sha256": hashlib.sha256((SOURCE / "src/core/documents/search.rs").read_bytes()).hexdigest()}


def execute(repair=False):
    destination = HERE / "repaired-retrieval" if repair else HERE
    if repair:
        destination.mkdir()
    ledger_path = destination / "probe-execution.json"
    if ledger_path.exists():
        raise RuntimeError("Execution ledger exists: no automatic benchmark retry")
    spent = json.loads((HERE / "probe-execution.json").read_text())["execution_seconds"] if repair else 0.0
    results = {"budget_seconds": 600, "previous_execution_seconds": spent, "execution_seconds": spent, "rounds": [], "completed": False}
    variants = ["baseline", "eligibility", "delta", "combined"]
    for round_number in range(3):
        order = variants[round_number:] + variants[:round_number]
        for variant in order:
            binary = "combined" if variant == "eligibility" else variant
            out = destination / f"probe-{variant}-{round_number + 1}.json"
            env = dict(ENV, HMS_PROBE_VARIANT=variant, HMS_PROBE_OUT=str(out), RAYON_NUM_THREADS="1")
            if variant == "eligibility":
                env["HMS_PROBE_DISABLE_DELTA"] = "1"
            command = [str(BINARIES / f"{binary}-retrieval"), "--exact", TEST, "--ignored", "--nocapture", "--test-threads=1"]
            run(command, env, destination / f"probe-{variant}-{round_number+1}.log", results, ledger_path)
        for variant in ([] if repair else ["baseline", "combined"]):
            command = [str(BINARIES / f"{variant}-temporal"), "--exact", TEMPORAL, "--nocapture", "--test-threads=1"]
            run(command, dict(ENV, RAYON_NUM_THREADS="1"), destination / f"temporal-{variant}-{round_number+1}.log", results, ledger_path)
    results["completed"] = True
    ledger_path.write_text(json.dumps(results, indent=2) + "\n")


def run(command, env, log_path, ledger, ledger_path):
    remaining = 600 - ledger["execution_seconds"]
    if remaining <= 0:
        raise RuntimeError("Local benchmark execution budget exhausted")
    start = time.monotonic()
    load_before = os.getloadavg()
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT)
        timeout = threading.Timer(remaining, process.kill)
        timeout.start()
        try:
            _, status, usage = os.wait4(process.pid, 0)
            code = os.waitstatus_to_exitcode(status)
            process.returncode = code
        finally:
            timeout.cancel()
            elapsed = time.monotonic() - start
            ledger["execution_seconds"] += elapsed
            ledger["rounds"].append({"command":command, "log":log_path.name, "seconds":elapsed,
                                     "exit_code":process.returncode, "peak_rss_platform_units":usage.ru_maxrss,
                                     "user_seconds":usage.ru_utime, "system_seconds":usage.ru_stime,
                                     "loadavg_before":load_before, "loadavg_after":os.getloadavg()})
            ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    if code:
        raise RuntimeError(f"Probe failed; see {log_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "run", "repair-build", "repair-run"])
    args = parser.parse_args()
    OWNED.mkdir(exist_ok=True)
    BINARIES.mkdir(exist_ok=True)
    if args.action in ("build", "repair-build"):
        baseline = build("baseline", BASELINE) if args.action == "build" else json.loads((HERE / "probe-builds-initial.json").read_text())[0]
        manifest = [baseline, build("combined", "HEAD"), build("delta", "HEAD", True)]
        (HERE / "probe-builds.json").write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        execute(args.action == "repair-run")
