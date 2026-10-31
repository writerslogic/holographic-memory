# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import tomllib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OWNED = ROOT / "target/retrieval-tail-20261031"
BASELINE = "f28095e024add5ce3964192f6af7b81457ad0b3d"
TEST = "core::documents::search::campaign_probe::campaign_retrieval_probe"
SETTINGS = {
    "CARGO_TARGET_DIR": str(Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")).resolve()),
    "CARGO_BUILD_JOBS": "4",
    "CARGO_PROFILE_DEV_OPT_LEVEL": "3",
    "CARGO_PROFILE_TEST_OPT_LEVEL": "3",
    "CARGO_PROFILE_DEV_DEBUG": "0",
    "CARGO_PROFILE_TEST_DEBUG": "0",
    "RAYON_NUM_THREADS": "1",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare_source(variant, source):
    if source.exists():
        shutil.rmtree(source)
    source.mkdir(parents=True)
    revision = BASELINE
    paths = ["Cargo.toml", "Cargo.lock", "build.rs", "src", "benches", "examples", "tests/golden/lexical-postings-v1"]
    package_spec = tomllib.loads(subprocess.check_output(["git", "show", f"{revision}:Cargo.toml"], cwd=ROOT, text=True))
    for family in ["bin", "bench", "example", "test"]:
        for target in package_spec.get(family, []):
            path = target.get("path")
            if path and not any(path == root or path.startswith(root + "/") for root in paths):
                paths.append(path)
    command = ["git", "archive", revision, *paths]
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE)
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        archive.extractall(source, filter="data")
    if process.wait():
        raise RuntimeError("Source archive failed")
    if variant == "candidate":
        patch_path = HERE / "candidate.patch"
        if not patch_path.exists():
            patch = subprocess.check_output(["git", "diff", "HEAD", "--", "src"], cwd=ROOT)
            if not patch:
                raise RuntimeError("Candidate patch is absent; preserve the measured patch when reproducing")
            patch_path.write_bytes(patch)
        subprocess.run(["git", "init", "--quiet"], cwd=source, check=True)
        subprocess.run(["git", "apply", "--check", str(patch_path)], cwd=source, check=True)
        subprocess.run(["git", "apply", str(patch_path)], cwd=source, check=True)
    shutil.copy2(HERE / "retrieval_probe.rs", source / "src/core/documents/campaign_probe.rs")
    with (source / "src/core/documents/search.rs").open("a") as output:
        output.write('\n#[cfg(test)]\n#[path = "campaign_probe.rs"]\nmod campaign_probe;\n')


def source_digest(source):
    result = hashlib.sha256()
    for path in sorted((source / "src").rglob("*.rs")) + [source / "Cargo.toml", source / "Cargo.lock"]:
        result.update(str(path.relative_to(source)).encode() + b"\0" + path.read_bytes())
    return result.hexdigest()


def verify_sources():
    for variant in ["baseline", "candidate"]:
        source = OWNED / f"verify-{variant}"
        prepare_source(variant, source)
        expected = json.loads((HERE / f"build-{variant}.json").read_text())["source_sha256"]
        actual = source_digest(source)
        if actual != expected:
            raise RuntimeError(f"{variant} source hash differs: {actual} != {expected}")
        print(f"{variant}: source hash verified ({actual})")


def build(variant):
    manifest = HERE / f"build-{variant}.json"
    if manifest.exists():
        raise RuntimeError("Build manifest already exists; retain it and choose a new experiment")
    source = OWNED / variant
    prepare_source(variant, source)
    command = ["cargo", "+1.99.0", "test", "--offline", "--locked", "--lib", "--no-run", "--message-format=json"]
    environment = dict(os.environ, **SETTINGS, TMPDIR=str(OWNED / "tmp"))
    executable = None
    with (HERE / f"build-{variant}.log").open("a") as log:
        process = subprocess.Popen(command, cwd=source, env=environment, stdout=subprocess.PIPE, stderr=log, text=True)
        for line in process.stdout:
            record = json.loads(line)
            if record.get("reason") == "compiler-message":
                log.write(record["message"].get("rendered", line))
            if record.get("reason") == "compiler-artifact" and record.get("executable") and record["target"]["name"] == "writerslogic_warm":
                executable = Path(record["executable"])
        if process.wait() or executable is None:
            raise RuntimeError(f"Build failed: build-{variant}.log")
    binary = OWNED / f"{variant}-probe"
    shutil.copy2(executable, binary)
    save(manifest, {
        "variant": variant,
        "revision": subprocess.check_output(["git", "rev-parse", BASELINE], cwd=ROOT, text=True).strip(),
        "source_sha256": source_digest(source),
        "binary": str(binary), "binary_sha256": digest(binary),
        "probe_sha256": digest(HERE / "retrieval_probe.rs"),
        "command": command, "settings": SETTINGS,
        "rustc": subprocess.check_output(["rustc", "+1.99.0", "-Vv"], text=True),
    })


def run():
    ledger_path = HERE / "execution.json"
    if ledger_path.exists():
        raise RuntimeError("Execution ledger exists; no automatic benchmark retry")
    builds = {name: json.loads((HERE / f"build-{name}.json").read_text()) for name in ["baseline", "candidate"]}
    for build in builds.values():
        if digest(Path(build["binary"])) != build["binary_sha256"]:
            raise RuntimeError("Probe executable changed")
    ledger = {"budget_seconds": 600, "execution_seconds": 0, "runs": [], "complete": False}
    save(ledger_path, ledger)
    for round_number in range(5):
        order = ["baseline", "candidate"] if round_number % 2 == 0 else ["candidate", "baseline"]
        for variant in order:
            remaining = 600 - ledger["execution_seconds"]
            if remaining <= 0:
                raise RuntimeError("Local execution budget exhausted")
            name = f"{variant}-{round_number + 1}"
            output = HERE / f"{name}.json"
            command = [builds[variant]["binary"], "--exact", TEST, "--ignored", "--nocapture", "--test-threads=1"]
            environment = dict(os.environ, **SETTINGS, TMPDIR=str(OWNED / "tmp"), WARM_PROBE_VARIANT=variant, WARM_PROBE_OUT=str(output))
            start = time.monotonic()
            load = os.getloadavg()
            status = None
            try:
                with (HERE / f"{name}.log").open("w") as log:
                    result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
                status = result.returncode
            finally:
                elapsed = time.monotonic() - start
                ledger["execution_seconds"] += elapsed
                ledger["runs"].append({"name": name, "seconds": elapsed, "status": status, "load_before": load, "load_after": os.getloadavg()})
                save(ledger_path, ledger)
            if status != 0:
                raise RuntimeError(f"Probe failed: {name}.log")
    ledger["complete"] = True
    save(ledger_path, ledger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["baseline", "candidate", "run", "verify-source"])
    arguments = parser.parse_args()
    (OWNED / "tmp").mkdir(parents=True, exist_ok=True)
    if arguments.action == "verify-source":
        verify_sources()
    elif arguments.action == "run":
        run()
    else:
        build(arguments.action)
