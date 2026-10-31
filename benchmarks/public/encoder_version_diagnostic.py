# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = "==3.13.*"
# dependencies = []
# ///
"""Freeze and run a bounded train-only capped-union compatibility diagnostic."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from document_search_compare import peak_rss_bytes, run_timed, time_arguments


ROOT = Path(__file__).resolve().parents[2]
ENCODERS = ("multiscale-words-v1", "multiscale-capped-union-v1")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.write("\n")


def prepare(folder, dataset):
    train = dataset / "qrels/train.tsv"
    relevant = {}
    with train.open() as source:
        for row in csv.DictReader(source, delimiter="\t"):
            if int(row["score"]) > 0:
                relevant.setdefault(row["query-id"], []).append(row["corpus-id"])
    qids = sorted(relevant, key=int)[:16]
    assert len(qids) == 16
    corpus = {}
    with (dataset / "corpus.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            corpus[row["_id"]] = row
    selected = {did for qid in qids for did in relevant[qid]}
    for did in sorted(corpus, key=lambda value: hashlib.sha256(value.encode()).hexdigest()):
        if len(selected) == 64:
            break
        selected.add(did)
    assert len(selected) == 64 and selected <= corpus.keys()
    documents = []
    for did in sorted(selected):
        row = corpus[did]
        documents.append({
            "id": did, "text": f"{row.get('title', '')}\n{row['text']}".strip(),
            "sourceUri": f"scifact:corpus:{did}", "metadata": {"dataset": "scifact-train"},
            "chunkWords": 4096, "overlapWords": 0,
        })
    queries = {}
    with (dataset / "queries.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            if row["_id"] in qids:
                queries[row["_id"]] = {
                    "id": row["_id"], "text": row["text"],
                    "relevantIds": sorted(relevant[row["_id"]]),
                }
    assert set(queries) == set(qids)
    write(folder / "input.json", {"documents": documents, "queries": [queries[qid] for qid in qids]})
    write(folder / "protocol.json", {
        "schema": 1, "purpose": "experimental version/real-text diagnostic; no adoption or superiority claim",
        "dataset": "SciFact train only; combined query file filtered by train qrels before selection",
        "dataset_hashes": {name: sha(dataset / name) for name in ("qrels/train.tsv", "corpus.jsonl", "queries.jsonl")},
        "input_sha256": sha(folder / "input.json"), "dimensions": [1024, 4096, 16384],
        "encoders": list(ENCODERS), "rounds": 5, "queries": 16, "documents": 64,
        "budget_seconds": 1200, "per_run_seconds": 120, "load_limit": 3,
        "threads": 1, "ordering": "alternate arm order each repetition; dimensions fixed ascending",
        "quality": "independently checked sparse Jaccard nDCG@10/Recall@10; lexical document API must match",
        "costs": "ingest, encode+query, compaction, restart; actual store file lengths and logical frames; OS process peak RSS",
        "limits": "small train subset; no budget tuning/equal-byte frontier; RSS includes validation work; no migration crash injection",
        "decision": "retain experimental opt-in only even if quality improves; keep every loss and D1024",
    })
    print(json.dumps({"prepared": str(folder), "documents": 64, "queries": 16}))


def run(folder, binary):
    protocol = json.loads((folder / "protocol.json").read_text())
    assert sha(folder / "input.json") == protocol["input_sha256"]
    sources = list((ROOT / "src").rglob("*.rs")) + [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    sources += [ROOT / name for name in (
        "examples/encoder_version_probe.rs", "benchmarks/public/encoder_version_diagnostic.py",
        "benchmarks/public/check_encoder_version_diagnostic.py", "benchmarks/public/check_sparse_composition.py",
        "benchmarks/public/document_search_compare.py", "benchmarks/public/test_encoder_version_diagnostic.py",
        "tests/encoder_versions.rs", "tests/fixtures/text-encoder-v1.json",
        "tests/fixtures/text-encoder-majority-tie.json",
    )]
    write(folder / "source-freeze.json", {
        "sources": {str(path.relative_to(ROOT)): sha(path) for path in sources},
        "binary": str(binary), "binary_sha256": sha(binary),
        "protocol_sha256": sha(folder / "protocol.json"), "platform": platform.platform(),
        "baseline_witnesses_sha256": sha(folder / "baseline-witnesses.json"),
    })
    rows = []
    started = time.monotonic()
    previous = os.environ.get("RAYON_NUM_THREADS")
    os.environ["RAYON_NUM_THREADS"] = "1"
    try:
        with Path("./.warm-target/timing.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for dim in protocol["dimensions"]:
                for repetition in range(protocol["rounds"]):
                    order = ENCODERS if repetition % 2 == 0 else ENCODERS[::-1]
                    for encoder in order:
                        remaining = protocol["budget_seconds"] - (time.monotonic() - started)
                        if remaining <= 0:
                            raise TimeoutError("diagnostic budget exhausted")
                        tag = f"{dim}-{repetition}-{encoder}"
                        report = folder / f"{tag}.json"
                        store = folder / f"{tag}.store"
                        log = folder / f"{tag}.log"
                        load_before = os.getloadavg()
                        command = [*time_arguments(platform.system()), str(binary), "--input", str(folder / "input.json"),
                                   "--output", str(report), "--store", str(store), "--dimensions", str(dim), "--encoder", encoder]
                        with log.open("x") as output:
                            run_timed(command, output, min(protocol["per_run_seconds"], remaining))
                        load_after = os.getloadavg()
                        rows.append({"dimensions": dim, "round": repetition, "encoder": encoder,
                                     "report": report.name, "report_sha256": sha(report), "log": log.name,
                                     "log_sha256": sha(log), "peak_rss_bytes": peak_rss_bytes(log.read_text(), platform.system()),
                                     "load_before": load_before, "load_after": load_after,
                                     "load_gate_met": max(load_before[0], load_after[0]) < protocol["load_limit"]})
                        write(folder / f"{tag}.receipt.json", rows[-1])
                        print(json.dumps({"finished": tag, "load_gate_met": rows[-1]["load_gate_met"]}), flush=True)
        write(folder / "runs.json", {"rows": rows, "elapsed_seconds": time.monotonic() - started})
    finally:
        if previous is None:
            os.environ.pop("RAYON_NUM_THREADS", None)
        else:
            os.environ["RAYON_NUM_THREADS"] = previous


def baseline(folder, binary):
    provenance = json.loads((folder / "golden-provenance.json").read_text())
    assert sha(binary) == provenance["binary_sha256"], "baseline binary differs from prechange goldens"
    inputs = json.loads((folder / "input.json").read_text())
    protocol = json.loads((folder / "protocol.json").read_text())
    requests = [dict(id=f"{dim}-{i}", text=row["text"], dim=dim, seed=0, rare_position=0)
                for dim in protocol["dimensions"] for i, row in enumerate(inputs["documents"] + inputs["queries"])]
    completed = subprocess.run([str(binary.resolve())], input="".join(json.dumps(row) + "\n" for row in requests),
                               text=True, capture_output=True, check=True, timeout=300)
    outputs = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(outputs) == len(requests)
    witnesses = []
    for expected, row in zip(requests, outputs):
        assert all(row[key] == expected[key] for key in expected)
        indices = next(value["indices"] for value in row["rows"] if value["arm"] == "P")
        witnesses.append({"text": row["text"], "dim": row["dim"], "majority": row["majority"], "indices": indices})
    write(folder / "baseline-witnesses.json", witnesses)
    print(json.dumps({"baseline_witnesses": len(witnesses)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "baseline", "run"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--binary", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.dataset is None:
            parser.error("--dataset is required")
        prepare(args.folder, args.dataset)
    else:
        if args.binary is None:
            parser.error("--binary is required")
        if args.mode == "baseline":
            baseline(args.folder, args.binary.resolve())
        else:
            run(args.folder, args.binary.resolve())


if __name__ == "__main__":
    main()
