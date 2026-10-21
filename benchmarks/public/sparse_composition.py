# /// script
# requires-python = "==3.13.*"
# dependencies = []
# ///
"""Frozen synthetic input preparation and bounded native witness collection."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import selectors
import subprocess
import tarfile
import time
import threading

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "benchmarks/results/superiority_v2/sparse_composition_run"
SEEDS = (11, 23, 47, 89, 179)
LENGTHS = (1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 512)
DIMS = (1024, 4096, 16384, 65536)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inputs():
    rows = []
    rejected = 0
    for seed in SEEDS:
        rng = random.Random(seed)
        for length in LENGTHS:
            bases = []
            seen = set()
            for family in range(8):
                while True:
                    ids = (rng.sample(range(4096), length) if family < 4 else
                           rng.choices(range(256), weights=[1 / (i + 1) for i in range(256)], k=length))
                    text = " ".join(f"word{i:04}" for i in ids)
                    if text not in seen:
                        seen.add(text)
                        bases.append(text)
                        break
                    rejected += 1
            for family, text in enumerate(bases):
                words = text.split()
                pos = length // 2
                replace = words.copy()
                replace[pos] = "word4095" if words[pos] != "word4095" else "word4094"
                swap = words.copy()
                if length > 1:
                    swap[0], swap[1] = swap[1], swap[0]
                half = words.copy()
                for i in range(max(1, length // 2)):
                    half[i] = "word4095" if words[i] != "word4095" else "word4094"
                rare = words.copy()
                rare[pos] = f"rare{family:04}"
                other = rare.copy()
                other[pos] = f"rare{(family + 1) % 64:04}"
                variants = {
                    "base": text, "duplicate": text, "character": "x" + text[1:],
                    "replace": " ".join(replace), "swap": " ".join(swap),
                    "prefix": "word4095 " + text, "suffix": text + " word4095",
                    "delete": " ".join(words[:pos] + words[pos + 1:]),
                    "half": " ".join(half), "rare": " ".join(rare),
                    "rare_other": " ".join(other), "unrelated": bases[(family + 1) % 4 + (family // 4) * 4],
                }
                for variant, value in variants.items():
                    rows.append({"id": f"s{seed}-n{length}-f{family}-{variant}", "seed": seed,
                                 "family": family, "length": length, "variant": variant,
                                 "corpus": "uniform" if family < 4 else "repeated",
                                 "text": value, "rare_position": pos,
                                 "rare_id": family if variant == "rare" else (family + 1) % 64 if variant == "rare_other" else None,
                                 "applicable": variant in ("base", "duplicate", "unrelated") or value != text})
    for seed in SEEDS:
        for repeats in (1, 2, 3, 7, 8, 9, 31, 32, 33, 127, 128, 129, 511, 512, 513):
            identity = (seed + repeats) % 64
            for variant, tail in (("rare", f"rare{identity:04}"), ("absent", "word0001"),
                                  ("nonce", "x"), ("phrase", f"rare{identity:04} word0001"),
                                  ("reverse", f"word0001 rare{identity:04}")):
                rows.append({"id": f"dedicated-s{seed}-r{repeats}-{variant}", "seed": seed,
                             "text": "word0000 " * repeats + tail, "rare_position": repeats,
                             "rare_id": identity if variant in ("rare", "phrase") else None,
                             "variant": variant, "corpus": "dedicated", "length": repeats,
                             "family": identity, "applicable": True})
    parent = json.loads((BASE.parent / "compaction_comprehensive/reconstructive-witness-report.json").read_text())
    for corpus in ("template", "varied"):
        witness = next(w for w in parent["witnesses"] if w["dimensions"] == 1024 and w["corpus"] == corpus)
        for i, text in enumerate(witness["texts"]):
            rows.append({"id": f"parent-{corpus}-{i}", "text": text, "seed": 11,
                         "rare_position": 0, "rare_id": None, "variant": "base", "corpus": corpus,
                         "family": i, "length": len(text.split()), "applicable": True})
    for i, text in enumerate(("", "   ", "x", "é", "e\u0301", "東京の記憶", "🧠🔐", "a" * 1024)):
        rows.append({"id": f"edge-{i}", "text": text, "seed": 11, "rare_position": 0,
                     "rare_id": None, "variant": "base", "corpus": "edge", "family": i,
                     "length": len(text.split()), "applicable": True})
    return rows, rejected


def prepare(attempt, binary):
    attempt.mkdir(exist_ok=False)
    rows, rejected = inputs()
    input_file = attempt / "inputs.jsonl"
    with input_file.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard",
                                    "src", "tests", "examples", "Cargo.toml", "Cargo.lock", "build.rs"], cwd=ROOT, text=True).splitlines()
    paths += ["benchmarks/public/sparse_composition.py", "benchmarks/public/check_sparse_composition.py",
              "benchmarks/public/test_sparse_composition.py",
              "benchmarks/public/sparse_composition.py.lock",
              "benchmarks/public/check_sparse_composition.py.lock",
              "benchmarks/public/test_sparse_composition.py.lock",
              "benchmarks/results/superiority_v2/compaction_comprehensive/invention-register.json",
              "benchmarks/results/superiority_v2/compaction_comprehensive/reconstructive-witness-report.json",
              "benchmarks/results/superiority_v2/sparse_composition_run/amendment.md",
              "benchmarks/results/superiority_v2/sparse_composition_design/protocol.md"]
    paths = sorted(set(paths))
    manifest = {p: digest(ROOT / p) for p in paths}
    with tarfile.open(attempt / "source.tar.gz", "w:gz") as archive:
        for p in paths:
            archive.add(ROOT / p, arcname=p)
    requests = sum(2 if r["corpus"] in ("template", "varied") else 4 for r in rows)
    freeze = {"schema": 1, "status": "frozen before execution", "sources": manifest,
              "input_sha256": digest(input_file), "source_capsule_sha256": digest(attempt / "source.tar.gz"),
              "binary": str(binary.resolve()), "binary_sha256": digest(binary),
              "dimensions": DIMS, "inputs": len(rows), "expected_requests": requests,
              "duplicate_base_rejections": rejected, "deadline_seconds": 21600,
              "output_cap_bytes": 20 * 1024**3, "request_timeout_seconds": 600,
              "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
              "python": os.sys.version, "host": list(os.uname()),
              "command": f"uv run --script benchmarks/public/sparse_composition.py run {attempt.relative_to(ROOT)}"}
    (attempt / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps({"prepared_inputs": len(rows), "requests": requests, "attempt": str(attempt)}))


def run(attempt):
    freeze = json.loads((attempt / "freeze.json").read_text())
    assert digest(attempt / "inputs.jsonl") == freeze["input_sha256"]
    assert digest(Path(freeze["binary"])) == freeze["binary_sha256"]
    for path, expected in freeze["sources"].items():
        assert digest(ROOT / path) == expected, path
    rows = [json.loads(line) for line in (attempt / "inputs.jsonl").read_text().splitlines()]
    started = time.monotonic()
    receipt = {"complete": False, "rows": 0, "outputs": {}, "load_before": os.getloadavg(),
               "timing_gate_met": False, "timing_scope": "descriptive components in reused process"}
    log = (attempt / "producer.log").open("x")
    operator = subprocess.run([freeze["binary"], "--operators"], capture_output=True, check=True, timeout=600)
    with gzip.open(attempt / "operators.jsonl.gz", "xb") as stream:
        stream.write(operator.stdout)
    receipt["outputs"]["operators.jsonl.gz"] = digest(attempt / "operators.jsonl.gz")
    process = subprocess.Popen([freeze["binary"]], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log)
    done = threading.Event()
    def monitor():
        while not done.wait(1):
            if process.poll() is not None:
                return
            observed = subprocess.run(["ps", "-o", "rss=", "-p", str(process.pid)], capture_output=True, text=True, timeout=5)
            if observed.returncode != 0:
                if process.poll() is not None:
                    return
                receipt["resource_error"] = "RSS observation failed"
                process.terminate()
                return
            rss = int(observed.stdout.strip()) * 1024
            receipt["max_observed_worker_rss_bytes"] = max(rss, receipt.get("max_observed_worker_rss_bytes", 0))
            if rss > 8 * 1024**3:
                receipt["resource_error"] = "8GiB observed RSS cap"
                process.terminate()
                return
    observer = threading.Thread(target=monitor, daemon=True)
    observer.start()
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        for dim in freeze["dimensions"]:
            path = attempt / f"d{dim}.jsonl.gz"
            with gzip.open(path, "xb") as stream:
                for row in rows:
                    if row["corpus"] in ("template", "varied") and dim not in (1024, 16384):
                        continue
                    assert time.monotonic() - started < freeze["deadline_seconds"], "campaign deadline"
                    assert sum(p.stat().st_size for p in attempt.glob("*.gz")) < freeze["output_cap_bytes"], "output cap"
                    request = {k: row[k] for k in ("id", "text", "seed", "rare_position")}
                    request["dim"] = dim
                    process.stdin.write((json.dumps(request) + "\n").encode())
                    process.stdin.flush()
                    line = bytearray()
                    request_deadline = time.monotonic() + freeze["request_timeout_seconds"]
                    while not line.endswith(b"\n"):
                        remaining = min(request_deadline, started + freeze["deadline_seconds"]) - time.monotonic()
                        assert remaining > 0 and selector.select(remaining), "request timeout"
                        chunk = os.read(process.stdout.fileno(), 65536)
                        assert chunk, "producer exited"
                        line.extend(chunk)
                        assert len(line) < 16 * 1024**2, "response cap"
                    assert line and len(line) < 16 * 1024**2, "producer exit or oversized response"
                    result = json.loads(line)
                    assert result["id"] == row["id"] and result["dim"] == dim
                    stream.write(line)
                    receipt["rows"] += 1
                    if receipt["rows"] % 100 == 0:
                        print(json.dumps({"rows": receipt["rows"], "dim": dim, "elapsed_s": round(time.monotonic() - started)}), flush=True)
            receipt["outputs"][path.name] = digest(path)
        process.stdin.close()
        assert process.wait(timeout=30) == 0
        assert receipt["rows"] == freeze["expected_requests"]
        receipt["complete"] = True
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        done.set()
        observer.join(timeout=10)
        selector.close()
        log.close()
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["load_after"] = os.getloadavg()
        (attempt / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("./.hms-target/release/examples/sparse_composition"))
    args = parser.parse_args()
    args.attempt = args.attempt.resolve()
    if args.mode == "prepare":
        prepare(args.attempt, args.binary)
    else:
        run(args.attempt)
