# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Bounded open-loop publication experiment, isolated tracing and frozen evidence."""

import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tarfile
import time

from check_compaction_updates import rows
from check_publication_serving import check
from compaction_updates import execute, sha, write
from instrument_publication import instrument

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "benchmarks/results/superiority_v3/reader_publication_v1"
HARNESSES = ("publication_serving.py", "publication_serving.rs", "publication_trace.rs",
             "instrument_publication.py", "check_publication_serving.py", "check_compaction_updates.py",
             "compaction_updates.py")


def manifest(root, names=None):
    if names is None:
        return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}
    return {name: sha(root / name) for name in sorted(names)}


def archive(path, root, names):
    with tarfile.open(path, "x:gz") as stream:
        for name in sorted(names):
            stream.add(root / name, arcname=name, recursive=False)


def prepare(folder):
    accepted = json.loads((PARENT / "gates-final-2/source-manifest.json").read_text())
    if manifest(ROOT, accepted) != accepted:
        raise ValueError("production source differs from accepted queued writer")
    folder.mkdir(parents=True, exist_ok=False)
    names = set(accepted) | set(json.loads((PARENT / "before-manifest.json").read_text()))
    before = manifest(ROOT, names)
    write(folder / "before-manifest.json", before)
    archive(folder / "before-source.tar.gz", ROOT, before)
    parent = ROOT / "benchmarks/results/superiority_v3/compaction_updates_v1/attempt-6/input-5182.json"
    originals = json.loads(parent.read_text())
    documents = []
    for number in range(20728):
        copy, index = divmod(number, 5182)
        doc = originals[index].copy()
        embedding = [0.0] * 128
        embedding[(number // 2) % 128] = 1.0
        doc.update(id=f"d{number:08}", sourceUri=f"{doc['sourceUri']}:copy{copy}",
                   metadata={"pair": number // 2, "revision": 0, "copy": copy, "source_index": index},
                   embeddings=[embedding])
        documents.append(doc)
    for n in (128, 5182, 20728):
        write(folder / f"input-{n}.json", documents[:n])
    for name in HARNESSES:
        shutil.copy2(ROOT / "benchmarks/public" / name, folder / name)
    write(folder / "protocol.json", {
        "schema_version": 1, "objective": "Measure and reduce live publication cost of document replacements",
        "baseline": "accepted FIFO writer and append/publication split, before posting replacement optimization",
        "source_parent_manifest_sha256": sha(PARENT / "gates-final-2/source-manifest.json"),
        "before_manifest_sha256": sha(folder / "before-manifest.json"),
        "before_archive_sha256": sha(folder / "before-source.tar.gz"),
        "corpus_parent": str(parent.relative_to(ROOT)), "corpus_parent_sha256": sha(parent),
        "input_sha256": {str(n): sha(folder / f"input-{n}.json") for n in (128, 5182, 20728)},
        "sizes": [5182, 20728], "batches": [2, 64], "rounds": 3,
        "order": "baseline/candidate on even rounds; reverse on odd rounds",
        "work": {"arrivals": 4800, "interval_ns": 1666667, "readers": 4, "queue_per_reader": 64,
                 "small_batch_writes": 256, "bulk_writes": 64, "hot_documents": 128,
                 "whole_bulk_request": "every sixteenth request in bulk cells captures all 64 documents",
                 "compactions": 1, "dimensions": 1024, "embedding_dimensions": 128},
        "arrival_contract": "Absolute monotonic scheduled times; nonblocking try_send; reject full queues; no retries. All scheduled arrivals and all accepted completions are checked. API-return response latency includes producer and queue delay; serialization between calls delays subsequent dequeue. No percentile imputations for drops.",
        "scope": "SciFact texts; 20728 is four copies with distinct IDs. One-hot f64 embeddings are synthetic controls, not learned embeddings or relevance evaluation. Two independent size/batch workloads; no cross-batch speed claim.",
        "tracing": "Isolated source capsules only. Monotonic stamps inside commits and final maintenance, fixed 1024-row in-memory collector after guards release. Exclusion begins just after acquisition and ends just after release: acquisition overhead is not attributed. Instrumentation overhead is uncalibrated and common to arms. No storage flush-stage/power-loss attribution.",
        "memory": "Separate seeding processes once per arm/size; fresh verified copies for each serving process. Only 128 input documents loaded by server; static begins after replay/warmup; eight external ps samples precede serving. OS page cache not cleared. RSS is sampled, not continuous peak.",
        "acceptance": {"zero_oracle_failures": True, "bulk_median_exclusion_p99_ratio_max": .8,
                       "service_and_scheduled_response_p99_ratio_max": 1.1, "additional_drops": 0,
                       "load_gate": "all sampled and boundary one-minute loads <3; failed timing gates remain descriptive"},
        "timeouts_seconds": {"build": 600, "seed": 300, "serve": 120, "campaign": 1800},
        "platform": platform.platform(), "harness_sha256": {name: sha(folder / name) for name in HARNESSES},
        "limits": "Local ARM only. Broad ANN/QA quarantine, model downloads, remote/paid compute, publishing and git operations remain excluded. Competitive superiority requires matched primary evidence, not this experiment.",
    })


def build(folder, arm, env):
    protocol = json.loads((folder / "protocol.json").read_text())
    if any(sha(folder / name) != digest for name, digest in protocol["harness_sha256"].items()):
        raise ValueError("frozen harness changed")
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
        if any(name not in ("src/core/documents/index.rs", "src/core/documents.rs",
                            "src/core/engine/mutation.rs") for name in changed):
            raise ValueError(f"unexpected production change: {changed}")
        write(folder / "candidate-changes.json", changed)
        write(folder / "candidate-origin-manifest.json", origin)
        archive(folder / "candidate-origin.tar.gz", ROOT, origin)
        with tarfile.open(folder / "candidate-origin.tar.gz") as stream:
            stream.extractall(source, filter="data")
    if manifest(source) != origin:
        raise ValueError("origin source capsule mismatch")
    instrument(source, arm, folder)
    touched = ["src/core/engine/mod.rs", "src/core/engine/mutation.rs", "src/core/engine/compaction.rs",
               "src/core/engine/diagnostic.rs", "examples/publication_serving.rs"]
    execute(["rustfmt", "--edition", "2021", "--config", "skip_children=true", *touched],
            folder, f"{arm}-format", 30, env, source)
    frozen = manifest(source)
    write(folder / f"{arm}-source-manifest.json", frozen)
    archive(folder / f"{arm}-source.tar.gz", source, frozen)
    execute(["cargo", "clippy", "--locked", "--release", "--example", "publication_serving", "--", "-D", "warnings"],
            folder, f"{arm}-clippy", 600, env, source)
    execute(["cargo", "build", "--locked", "--release", "--example", "publication_serving"],
            folder, f"{arm}-build", 600, env, source)
    if manifest(source) != frozen:
        raise ValueError("source changed during build")
    probe = folder / f"{arm}-probe"
    shutil.copy2(Path(env["CARGO_TARGET_DIR"]) / "release/examples/publication_serving", probe)
    write(folder / f"{arm}-binary.json", {"sha256": sha(probe), "source_manifest_sha256": sha(folder / f"{arm}-source-manifest.json"),
                                         "rustc": subprocess.check_output(["rustc", "-Vv"], env=env, text=True)})


def resources(receipt):
    events, samples = receipt["events"], receipt["samples"]
    if [row["phase"] for row in events] != ["static", "serving", "verification", "done"]:
        raise ValueError("bad observer phases")
    if not all(a["epoch_ns"] < b["epoch_ns"] for a, b in zip(events, events[1:])):
        raise ValueError("phase clock regressed")
    phases = {}
    for left, right in zip(events, events[1:]):
        values = [row["rss_bytes"] for row in samples if row["begin_epoch_ns"] >= left["epoch_ns"]
                  and row["end_epoch_ns"] <= right["epoch_ns"]]
        phases[left["phase"]] = {"sample_count": len(values), "max_rss_bytes": max(values, default=None),
                                 "median_rss_bytes": statistics.median(values) if values else None}
    if phases["static"]["sample_count"] < 8:
        raise ValueError("static observation missing")
    return {"phases": phases, "load_gate_met": bool(samples) and all(load[0] < 3 for load in
            [receipt["load_before"], receipt["load_after"], *[row["load"] for row in samples]])}


def seed(folder, arm, n, env):
    store = folder / f"{n}-{arm}-seed-store"
    execute([str(folder / f"{arm}-probe"), "--input", str(folder / f"input-{n}.json"),
             "--store", str(store), "--documents", str(n), "--batch", "2", "--seed",
             "--out", str(folder / f"{n}-{arm}-seed")], folder, f"{n}-{arm}-seed", 300, env)
    frozen = manifest(store)
    write(folder / f"{n}-{arm}-seed-manifest.json", frozen)
    return store, frozen


def cell(folder, arm, n, batch, repetition, seed_store, frozen_seed, env):
    label = f"{n}-{batch}-{repetition}-{arm}"
    store = folder / f"{label}-store"
    shutil.copytree(seed_store, store)
    if manifest(store) != frozen_seed:
        raise ValueError("seed copy changed")
    write(folder / f"{label}-seed-copy.json", {"verified": True, "seed_manifest_sha256": sha(folder / f"{n}-{arm}-seed-manifest.json")})
    output = folder / label
    execute([str(folder / f"{arm}-probe"), "--input", str(folder / "input-128.json"), "--documents", str(n),
             "--batch", str(batch), "--store", str(store), "--out", str(output)],
            folder, label, 120, env, sample=True)
    summary = check(output, folder / f"input-{n}.json", arm)
    write(output / "checked.json", summary)
    receipt = json.loads((folder / f"{label}-receipt.json").read_text())
    resource = resources(receipt)
    for path in output.glob("*.jsonl"):
        with path.open("rb") as src, gzip.open(str(path) + ".gz", "xb") as dst:
            shutil.copyfileobj(src, dst)
        path.unlink()
    return {"label": label, "n": n, "batch": batch, "round": repetition, "arm": arm,
            "checked": summary, "resources": resource}


def run(folder, env, smoke=False):
    protocol = json.loads((folder / "protocol.json").read_text())
    for name, digest in protocol["harness_sha256"].items():
        if sha(folder / name) != digest:
            raise ValueError("frozen harness changed")
    for n, digest in protocol["input_sha256"].items():
        if sha(folder / f"input-{n}.json") != digest:
            raise ValueError("frozen input changed")
    for arm in (["baseline"] if smoke else ["baseline", "candidate"]):
        if sha(folder / f"{arm}-probe") != json.loads((folder / f"{arm}-binary.json").read_text())["sha256"]:
            raise ValueError("frozen binary changed")
    measurements, pairs = [], []
    with Path(env["CARGO_TARGET_DIR"]).joinpath("timing.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        sizes = [5182] if smoke else protocol["sizes"]
        for n in sizes:
            seeds = {arm: seed(folder, arm, n, env) for arm in (["baseline"] if smoke else ["baseline", "candidate"])}
            for batch in ([64] if smoke else protocol["batches"]):
                for repetition in range(1 if smoke else protocol["rounds"]):
                    order = ["baseline"] if smoke else (["baseline", "candidate"] if repetition % 2 == 0 else ["candidate", "baseline"])
                    for arm in order:
                        if time.monotonic() - started > 1680:
                            raise TimeoutError("campaign budget exhausted")
                        row = cell(folder, arm, n, batch, repetition, *seeds[arm], env)
                        measurements.append(row)
                        print(json.dumps({"label": row["label"], "checked": True,
                                          "response_p99_ms": row["checked"]["scheduled_response"]["p99_ns"] / 1e6,
                                          "exclusion_p99_ms": row["checked"]["reader_exclusion"]["p99_ns"] / 1e6}), flush=True)
                    if not smoke:
                        count = 0
                        for name in ("before-restart.jsonl", "after-restart.jsonl"):
                            left = rows(folder / f"{n}-{batch}-{repetition}-baseline", name)
                            right = rows(folder / f"{n}-{batch}-{repetition}-candidate", name)
                            if [r["results"] for r in left] != [r["results"] for r in right]:
                                raise ValueError("cross-arm final state changed")
                            count += len(left)
                        pairs.append({"n": n, "batch": batch, "round": repetition, "exact_result_pairs": count})
            if time.monotonic() - started > 1800:
                raise TimeoutError("campaign budget exhausted")
    write(folder / "measurements.json", {"runs": measurements, "cross_arm": pairs,
                                          "seconds": time.monotonic() - started, "smoke": smoke})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "build", "smoke", "run"])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--arm", choices=["baseline", "candidate"])
    args = parser.parse_args()
    folder = args.folder.resolve()
    env = os.environ | {"RUSTUP_TOOLCHAIN": "1.96.0", "CARGO_TARGET_DIR": "/Volumes/A/.hms-target",
                        "CARGO_BUILD_JOBS": "3", "RAYON_NUM_THREADS": "4"}
    if args.action == "prepare":
        prepare(folder)
    elif args.action == "build":
        if args.arm is None:
            parser.error("build requires --arm")
        build(folder, args.arm, env)
    else:
        run(folder, env, smoke=args.action == "smoke")


if __name__ == "__main__":
    main()
