# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Measure tight replay buffers against the sealed shared-owner treatment."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from compaction_updates import execute, sha, write
from instrument_publication import instrument
from publication_serving import archive, manifest, run

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "benchmarks/results/superiority_v3/shared_document_publication_v1"
HARNESSES = [*json.loads((PARENT / "attempt-1/protocol.json").read_text())["harness_sha256"],
             "replay_capacity_publication.py"]


def prepare(folder):
    evidence = folder.parent
    before = json.loads((evidence / "before-manifest.json").read_text())
    if sha(PARENT / "acceptance.json") != json.loads((evidence / "design.json").read_text())["parent_acceptance_sha256"]:
        raise ValueError("parent acceptance changed")
    folder.mkdir(exist_ok=False)
    shutil.copy2(evidence / "before-source.tar.gz", folder / "before-source.tar.gz")
    write(folder / "before-manifest.json", before)
    for n in (128, 5182, 20728):
        shutil.copy2(PARENT / "attempt-1" / f"input-{n}.json", folder / f"input-{n}.json")
    for name in HARNESSES:
        shutil.copy2(ROOT / "benchmarks/public" / name, folder / name)
    protocol = json.loads((PARENT / "attempt-1/protocol.json").read_text())
    protocol.update(objective="Recover replay buffer capacity without restoring publication deep copies",
                    baseline="sealed shared-owner candidate before length-sized chunk deserialization",
                    source_parent_manifest_sha256=sha(PARENT / "attempt-1/candidate-origin-manifest.json"),
                    before_manifest_sha256=sha(folder / "before-manifest.json"),
                    before_archive_sha256=sha(folder / "before-source.tar.gz"),
                    parent_acceptance_sha256=sha(PARENT / "acceptance.json"),
                    harness_sha256={name: sha(folder / name) for name in HARNESSES},
                    current_adoption_rule="strict no-sacrifice amendment inherited; report every regression and failed load gate")
    write(folder / "protocol.json", protocol)


def build(folder, arm, env):
    protocol = json.loads((folder / "protocol.json").read_text())
    if any(sha(folder / name) != digest for name, digest in protocol["harness_sha256"].items()):
        raise ValueError("harness changed")
    before = json.loads((folder / "before-manifest.json").read_text())
    source = folder / f"{arm}-source"
    source.mkdir()
    if arm == "baseline":
        with tarfile.open(folder / "before-source.tar.gz") as stream:
            stream.extractall(source, filter="data")
        origin = before
    else:
        origin = manifest(ROOT, before)
        changed = sorted(name for name in before if before[name] != origin[name])
        allowed = ["src/core/documents.rs", "src/core/engine/mutation.rs"]
        if changed != allowed:
            raise ValueError(f"unexpected change scope: {changed}")
        write(folder / "candidate-changes.json", changed)
        write(folder / "candidate-origin-manifest.json", origin)
        archive(folder / "candidate-origin.tar.gz", ROOT, origin)
        with tarfile.open(folder / "candidate-origin.tar.gz") as stream:
            stream.extractall(source, filter="data")
    if manifest(source) != origin:
        raise ValueError("source capsule differs")
    instrument(source, arm, folder)
    touched = ["src/core/engine/mod.rs", "src/core/engine/mutation.rs", "src/core/engine/compaction.rs",
               "src/core/engine/diagnostic.rs", "examples/publication_serving.rs"]
    execute(["rustfmt", "--edition", "2021", "--config", "skip_children=true", *touched], folder,
            f"{arm}-format", 30, env, source)
    frozen = manifest(source)
    write(folder / f"{arm}-source-manifest.json", frozen)
    archive(folder / f"{arm}-source.tar.gz", source, frozen)
    execute(["cargo", "clippy", "--locked", "--release", "--example", "publication_serving", "--", "-D", "warnings"],
            folder, f"{arm}-clippy", 600, env, source)
    execute(["cargo", "build", "--locked", "--release", "--example", "publication_serving"], folder,
            f"{arm}-build", 600, env, source)
    if manifest(source) != frozen:
        raise ValueError("source changed during build")
    shutil.copy2(Path(env["CARGO_TARGET_DIR"]) / "release/examples/publication_serving", folder / f"{arm}-probe")
    write(folder / f"{arm}-binary.json", {"sha256": sha(folder / f"{arm}-probe"),
          "source_manifest_sha256": sha(folder / f"{arm}-source-manifest.json"),
          "rustc": subprocess.check_output(["rustc", "-Vv"], env=env, text=True)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "build", "run"])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--arm", choices=["baseline", "candidate"])
    args = parser.parse_args()
    env = os.environ | {"RUSTUP_TOOLCHAIN": "1.96.0", "CARGO_TARGET_DIR": "/Volumes/A/.hms-target",
                        "CARGO_BUILD_JOBS": "3", "RAYON_NUM_THREADS": "4"}
    folder = args.folder.resolve()
    if args.action == "prepare":
        prepare(folder)
    elif args.action == "build":
        if args.arm is None:
            parser.error("build requires --arm")
        build(folder, args.arm, env)
    else:
        run(folder, env)
