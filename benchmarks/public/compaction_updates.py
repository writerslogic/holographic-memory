# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Bounded local compaction diagnostic with fixed work and separate seed processes."""

import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import selectors
import shutil
import signal
import statistics
import subprocess
import tarfile
import time

from check_compaction_updates import check, rows

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "benchmarks/results/superiority_v3/compaction_replay_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=False)
    corpus = Path.home() / ".cache/hms-bench/downloads/scifact/corpus.jsonl"
    records = [json.loads(line) for line in corpus.open()]
    documents = []
    for number, row in enumerate(records[:5182]):
        text = (row.get("title", "") + "\n" + row["text"]).strip() + " hmsprobe"
        if len(text.encode()) > 64000 or len(text.split()) > 4095:
            raise ValueError("fixture requires one complete chunk per document")
        documents.append({"id": f"d{number:08}", "text": text, "sourceUri": f"scifact:{row['_id']}",
                          "version": "0", "metadata": {"pair": number // 2, "revision": 0},
                          "chunkWords": 4096, "overlapWords": 0, "storeText": True})
    if len(documents) != 5182:
        raise ValueError("incomplete cached SciFact corpus")
    for n in (32, 1024, 5182):
        write(folder / f"input-{n}.json", documents[:n])
    parent_source = json.loads((PARENT / "gates-final-4/source-manifest.json").read_text())
    if any(sha(ROOT / name) != digest for name, digest in parent_source.items()):
        raise ValueError("production source differs from accepted compaction parent")
    source = parent_source | {"examples/compaction_updates.rs": sha(ROOT / "examples/compaction_updates.rs")}
    source.update({str(path.relative_to(ROOT)): sha(path) for path in (ROOT / "benches").rglob("*.rs")})
    write(folder / "source-manifest.json", source)
    with tarfile.open(folder / "source.tar.gz", "w:gz") as archive:
        for name in sorted(source):
            archive.add(ROOT / name, arcname=name, recursive=False)
    harness = {}
    for name in ("compaction_updates.py", "check_compaction_updates.py", "test_compaction_updates.py"):
        path = ROOT / "benchmarks/public" / name
        shutil.copy2(path, folder / name)
        harness[name] = sha(path)
    write(folder / "protocol.json", {
        "scope": "local ARM functional and descriptive workload diagnostic; no broad ANN or competitor campaign",
        "corpus": {"path": str(corpus), "sha256": sha(corpus), "rows": len(records),
                   "used": 5182, "selection": "first rows in existing corpus order; no queries or qrels read"},
        "sizes": [1024, 5182], "rounds": 3, "order": "baseline/candidate on even rounds, reversed on odd rounds",
        "work": {"readers": 4, "static_queries_per_reader": 256, "update_queries_per_reader": 1024,
                 "two_document_batches": 256, "single_deletions": 64, "compactions": 1,
                 "hot_pairs": 16, "dimensions": 1024, "embeddings": None,
                 "reader_pacing_ms": 8, "reader_start": "compaction begins after first write acknowledgment"},
        "memory": "fresh serving child reopens seeded store; ingestion excluded; samples bracket each ps read; "
                  "phase extrema use only reads wholly inside phase boundaries; fixed file buffers and 32 fixture "
                  "documents are included; static phase waits for eight observer samples before fixed queries, "
                  "individual query clocks exclude that wait; sampled maxima can miss peaks; no learned embeddings",
        "gate": "every serving load sample plus before/after must have one-minute load < 3",
        "limits": "synthetic update/query pattern on original abstracts; lexical filtered queries over hot pairs; "
                  "no throughput, relevance, primary-envelope or competitor claim; final lock pause not instrumented",
        "timeouts_seconds": {"seed": 300, "serve": 180, "timing_lock": 1500},
        "baseline_archive_sha256": sha(PARENT / "before-source.tar.gz"),
        "baseline_manifest_sha256": sha(PARENT / "before-manifest.json"),
        "candidate_archive_sha256": sha(folder / "source.tar.gz"),
        "source_manifest_sha256": sha(folder / "source-manifest.json"),
        "harness_sha256": harness,
        "input_sha256": {str(n): sha(folder / f"input-{n}.json") for n in (32, 1024, 5182)},
    })


def execute(command, folder, label, timeout, env, cwd=ROOT, sample=False):
    started, before = time.monotonic(), os.getloadavg()
    events, samples, buffer = [], [], b""
    signaled = False
    if sample:
        command = [*command, "--observe"]
    with (folder / f"{label}.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE if sample else log,
                                stdin=subprocess.PIPE if sample else subprocess.DEVNULL,
                                stderr=log, start_new_session=True)
        try:
            if sample:
                with selectors.DefaultSelector() as selector:
                    selector.register(proc.stdout, selectors.EVENT_READ)
                    while selector.get_map() or proc.poll() is None:
                        if time.monotonic() - started > timeout:
                            raise TimeoutError(label)
                        for key, _ in selector.select(.02):
                            chunk = os.read(key.fd, 4096)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            buffer += chunk
                            while b"\n" in buffer:
                                line, buffer = buffer.split(b"\n", 1)
                                events.append(json.loads(line))
                                log.write(line.decode() + "\n")
                        if proc.poll() is None:
                            begin = time.time_ns()
                            counter = subprocess.run(["ps", "-o", "rss=", "-p", str(proc.pid)],
                                                     capture_output=True, text=True, timeout=2)
                            end = time.time_ns()
                            if counter.returncode == 0 and counter.stdout.strip():
                                rss = int(counter.stdout.strip()) * 1024
                                if rss > 0:
                                    samples.append({"begin_epoch_ns": begin, "end_epoch_ns": end,
                                                    "rss_bytes": rss, "load": os.getloadavg()})
                                    if events and not signaled and sum(row["begin_epoch_ns"] >= events[0]["epoch_ns"]
                                                                     for row in samples) >= 8:
                                        proc.stdin.write(b"start\n")
                                        proc.stdin.flush()
                                        signaled = True
                            elif proc.poll() is None:
                                raise RuntimeError("RSS observer failed for live child")
                    if buffer:
                        raise ValueError("truncated phase event")
            code = proc.wait(timeout=max(.1, timeout - (time.monotonic() - started)))
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            if sample:
                proc.stdout.close()
                proc.stdin.close()
            write(folder / f"{label}-receipt.json", {
                "command": command, "exit_code": proc.returncode, "seconds": time.monotonic() - started,
                "load_before": before, "load_after": os.getloadavg(), "events": events,
                "samples": samples, "environment": {key: env[key] for key in
                    ("RUSTUP_TOOLCHAIN", "CARGO_TARGET_DIR", "RAYON_NUM_THREADS")},
            })
        if code:
            raise subprocess.CalledProcessError(code, command)


def resources(receipt):
    events, samples = receipt["events"], receipt["samples"]
    phases = ["static", "updates", "compacting", "updates-tail", "verification", "done"]
    if [row["phase"] for row in events] != phases:
        raise ValueError("missing serving phase events")
    if not all(a["epoch_ns"] < b["epoch_ns"] for a, b in zip(events, events[1:])):
        raise ValueError("phase clock regressed")
    metrics = {}
    for index, phase in enumerate(phases[:-1]):
        start, end = events[index]["epoch_ns"], events[index + 1]["epoch_ns"]
        values = [row["rss_bytes"] for row in samples
                  if row["begin_epoch_ns"] >= start and row["end_epoch_ns"] <= end]
        metrics[phase] = {"sample_count": len(values), "sampled_max_rss_bytes": max(values, default=None),
                          "sampled_median_rss_bytes": statistics.median(values) if values else None}
    return {"phases": metrics, "load_gate_met": bool(samples) and
            all(load[0] < 3 for load in [receipt["load_before"], receipt["load_after"],
                                        *[row["load"] for row in samples]])}


def build(folder, env):
    protocol = json.loads((folder / "protocol.json").read_text())
    if sha(PARENT / "before-source.tar.gz") != protocol["baseline_archive_sha256"]:
        raise ValueError("baseline archive changed")
    source = folder / "baseline-source"
    source.mkdir()
    with tarfile.open(folder / "source.tar.gz") as archive:
        archive.extractall(source, filter="data")
    with tarfile.open(PARENT / "before-source.tar.gz") as archive:
        archive.extractall(source, filter="data")
    manifest = json.loads((PARENT / "before-manifest.json").read_text())
    for name, digest in manifest.items():
        if sha(source / name) != digest:
            raise ValueError(f"baseline archive mismatch: {name}")
    write(folder / "baseline-source-manifest.json", {
        str(path.relative_to(source)): sha(path) for path in sorted(source.rglob("*")) if path.is_file()})
    for arm, cwd in [("baseline", source), ("candidate", ROOT)]:
        execute(["cargo", "build", "--locked", "--release", "--example", "compaction_updates"],
                folder, f"{arm}-build", 600, env, cwd)
        shutil.copy2(Path(env["CARGO_TARGET_DIR"]) / "release/examples/compaction_updates", folder / f"{arm}-probe")
    write(folder / "binaries.json", {arm: sha(folder / f"{arm}-probe") for arm in ("baseline", "candidate")})


def run(folder, env):
    protocol = json.loads((folder / "protocol.json").read_text())
    frozen = json.loads((folder / "source-manifest.json").read_text())
    if any(sha(ROOT / name) != digest for name, digest in frozen.items()):
        raise ValueError("candidate source changed")
    if any(sha(ROOT / "benchmarks/public" / name) != digest
           for name, digest in protocol["harness_sha256"].items()):
        raise ValueError("harness changed")
    binaries = json.loads((folder / "binaries.json").read_text())
    if any(sha(folder / f"{arm}-probe") != digest for arm, digest in binaries.items()):
        raise ValueError("binary changed")
    for n, digest in protocol["input_sha256"].items():
        if sha(folder / f"input-{n}.json") != digest:
            raise ValueError("input changed")
    measurements = []
    with Path("/Volumes/A/.hms-target/timing.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        for n in protocol["sizes"]:
            for repetition in range(protocol["rounds"]):
                order = ["baseline", "candidate"] if repetition % 2 == 0 else ["candidate", "baseline"]
                for arm in order:
                    if time.monotonic() - started > 1200:
                        raise TimeoutError("bounded campaign exhausted before next cell")
                    label = f"{n}-{repetition}-{arm}"
                    store = folder / f"{label}-store"
                    output = folder / label
                    command = [str(folder / f"{arm}-probe"), "--store", str(store)]
                    remaining = 1500 - (time.monotonic() - started)
                    execute([*command, "--input", str(folder / f"input-{n}.json"),
                             "--seed", "--out", str(folder / f"{label}-seed")],
                            folder, f"{label}-seed", min(300, remaining), env)
                    remaining = 1500 - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("campaign exhausted after seed")
                    execute([*command, "--input", str(folder / "input-32.json"), "--documents", str(n),
                             "--out", str(output)], folder, label, min(180, remaining), env, sample=True)
                    summary = check(output, folder / f"input-{n}.json")
                    receipt = json.loads((folder / f"{label}-receipt.json").read_text())
                    write(output / "checked.json", summary)
                    measurements.append({"n": n, "round": repetition, "arm": arm, "order": order,
                                         "label": label, "checked": summary, "resources": resources(receipt)})
                    for path in output.glob("*.jsonl"):
                        with path.open("rb") as src, gzip.open(str(path) + ".gz", "xb") as dest:
                            shutil.copyfileobj(src, dest)
                        path.unlink()
                    print(label, "independently checked", flush=True)
                    write(folder / f"{label}-completed.json", measurements[-1])
                baseline = folder / f"{n}-{repetition}-baseline"
                candidate = folder / f"{n}-{repetition}-candidate"
                for name in ["before-restart.jsonl", "after-restart.jsonl",
                             *[f"static-{reader}.jsonl" for reader in range(4)]]:
                    left, right = rows(baseline, name), rows(candidate, name)
                    if [row["results"] for row in left] != [row["results"] for row in right]:
                        raise ValueError(f"paired exact result mismatch: {name}")
                write(folder / f"{n}-{repetition}-paired.json", {"exact": True,
                      "static_query_pairs": 1024, "initial_final_restart_pairs": n * 2})
    if any(sha(ROOT / name) != digest for name, digest in frozen.items()):
        raise ValueError("source changed during measurement")
    write(folder / "measurements.json", {"platform": platform.platform(), "runs": measurements,
                                        "timing_claim_eligible": False, "competitor_claim": False})


def main():
    def interrupted(_signum, _frame):
        raise SystemExit("diagnostic interrupted; active child group cleaned up")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "build", "run"])
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    env = dict(os.environ, RUSTUP_TOOLCHAIN="1.96.0", CARGO_TARGET_DIR="/Volumes/A/.hms-target",
               CARGO_BUILD_JOBS="4", RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if args.action == "prepare":
        prepare(folder)
    elif args.action == "build":
        build(folder, env)
    else:
        run(folder, env)


if __name__ == "__main__":
    main()
