# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Paired public train/dev lexical retrieval against SQLite FTS5 and previous HMS."""

import argparse
import fcntl
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import sys
import time

from document_search_compare import peak_rss_bytes, quantile, run_timed, time_arguments


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(folder):
    folder.mkdir(parents=True, exist_ok=True)
    cache = Path.home() / ".cache/hms-bench/downloads"
    for dataset, split in [("scifact", "train"), ("nfcorpus", "dev")]:
        source = cache / dataset
        qrels = {}
        with (source / "qrels" / f"{split}.tsv").open() as stream:
            next(stream)
            for line in stream:
                qid, did, grade = line.rstrip().split("\t")
                qrels.setdefault(qid, {})[did] = int(grade)
        documents = []
        for line in (source / "corpus.jsonl").open():
            row = json.loads(line)
            text = (row.get("title", "") + "\n" + row["text"]).strip()
            if len(text.split()) > 4096:
                raise ValueError("this matched protocol requires one complete chunk per document")
            bucket = int(row["_id"]) % 17 if row["_id"].isdigit() else len(row["_id"]) % 17
            documents.append({"id": row["_id"], "text": text, "metadata": {"bucket": bucket},
                              "chunkWords": 4096, "overlapWords": 0, "storeText": False})
        queries = [{"id": row["_id"], "text": row["text"],
                    "options": {"k": 100, "candidateLimit": 100, "lexicalWeight": 1, "semanticWeight": 0}}
                   for row in map(json.loads, (source / "queries.jsonl").open()) if row["_id"] in qrels]
        (folder / f"{dataset}-{split}-input.json").write_text(json.dumps({"documents": documents, "queries": queries}) + "\n")
        (folder / f"{dataset}-{split}-qrels.json").write_text(json.dumps(qrels) + "\n")


def sqlite_run(input_path, store, output):
    data = json.loads(input_path.read_text())
    ingest = not store.exists()
    start = time.perf_counter_ns()
    connection = sqlite3.connect(store)
    if ingest:
        connection.execute("CREATE VIRTUAL TABLE docs USING fts5(body, content='', tokenize='porter unicode61')")
        connection.execute("CREATE TABLE identities (rowid INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL)")
        with connection:
            for i, document in enumerate(data["documents"], 1):
                connection.execute("INSERT INTO docs(rowid, body) VALUES (?, ?)", (i, document["text"]))
                connection.execute("INSERT INTO identities(rowid, id) VALUES (?, ?)", (i, document["id"]))
        connection.execute("INSERT INTO docs(docs) VALUES ('optimize')")
        connection.commit()
    open_ns = time.perf_counter_ns() - start

    def search(query):
        words = sorted({"".join(chars) for kind, chars in itertools.groupby(query["text"].lower(), str.isalnum) if kind})
        expression = " OR ".join('"' + word + '"' for word in words)
        if not expression:
            return []
        return connection.execute(
            "SELECT identities.id, bm25(docs) FROM docs JOIN identities ON docs.rowid=identities.rowid "
            "WHERE docs MATCH ? ORDER BY 2, identities.id LIMIT ?",
            (expression, query["options"]["k"]),
        ).fetchall()

    for query in data["queries"][:16]:
        search(query)
    rows = []
    for query in data["queries"]:
        start = time.perf_counter_ns()
        results = search(query)
        elapsed_ns = time.perf_counter_ns() - start
        rows.append({"id": query["id"], "elapsed_ns": elapsed_ns,
                     "results": [{"documentId": key, "lexicalScore": -score} for key, score in results]})
    connection.close()
    write(output, {"documents": len(data["documents"]), "ingest": ingest, "open_ns": open_ns,
                   "sqlite_version": sqlite3.sqlite_version, "rows": rows})


def quality(rows, qrels):
    ndcg, recall = [], []
    for row in rows:
        relevant = qrels[row["id"]]
        found = [hit["documentId"] for hit in row["results"]]
        gains = lambda ids: sum(relevant.get(key, 0) / math.log2(rank + 2) for rank, key in enumerate(ids))
        ideal = sum(grade / math.log2(rank + 2) for rank, grade in enumerate(sorted(relevant.values(), reverse=True)[:10]))
        ndcg.append(gains(found[:10]) / ideal if ideal else 0.0)
        positive = {key for key, grade in relevant.items() if grade > 0}
        recall.append(len(set(found) & positive) / len(positive) if positive else 0.0)
    return {"ndcg@10": statistics.mean(ndcg), "recall@100": statistics.mean(recall)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--sqlite-input", type=Path)
    parser.add_argument("--sqlite-store", type=Path)
    parser.add_argument("--sqlite-output", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.out)
        return
    if args.sqlite_input:
        sqlite_run(args.sqlite_input, args.sqlite_store, args.sqlite_output)
        return
    if not (args.baseline and args.candidate and 1 <= args.rounds <= 5):
        raise ValueError("supply both binaries and 1..=5 rounds")
    folder = args.out.resolve()
    binaries = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    protocol = {
        "datasets": {"scifact": "train", "nfcorpus": "dev"}, "rounds": args.rounds,
        "k": 100, "chunk_words": 4096, "overlap_words": 0, "retained_source_text": False,
        "binary_sha256": {arm: sha(path) for arm, path in binaries.items()},
        "sqlite_version": sqlite3.sqlite_version, "sqlite_primary_docs": "https://www.sqlite.org/fts5.html",
        "sqlite_tokenizer": "Porter1 over unicode61 (default diacritic folding)",
        "hms_tokenizer": "English Snowball Porter2 over lowercase Unicode alphanumeric tokens",
        "idf_difference": "SQLite log((N-df+.5)/(df+.5)) floored at 1e-6; HMS log(1+(N-df+.5)/(df+.5))",
        "matched": "same complete corpora, questions, top-100, persistent index, no retained source text; no model stages",
        "limits": "single local ARM host; lexical-only SQLite lacks HMS vectors; load<3 required for timing claims",
    }
    write(folder / "protocol.json", protocol)
    measurements, summary = [], {}
    started = time.monotonic()
    with Path("/Volumes/A/.hms-target/timing.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for dataset, split in protocol["datasets"].items():
            input_path = folder / f"{dataset}-{split}-input.json"
            qrels_path = folder / f"{dataset}-{split}-qrels.json"
            qrels = json.loads(qrels_path.read_text())
            reference = None
            cells = {arm: [] for arm in ["baseline", "candidate", "sqlite"]}
            qualities = {}
            for repetition in range(args.rounds):
                arms = ["baseline", "candidate", "sqlite"] if repetition % 2 == 0 else ["sqlite", "candidate", "baseline"]
                for arm in arms:
                    remaining = 900 - (time.monotonic() - started)
                    if remaining <= 0:
                        raise RuntimeError("900-second bounded campaign exhausted")
                    output = folder / f"{dataset}-{repetition}-{arm}.json"
                    log_path = output.with_suffix(".log")
                    store = folder / f"{dataset}-{arm}-store"
                    if arm == "sqlite":
                        command = [sys.executable, str(Path(__file__).resolve()), "--out", str(folder),
                                   "--sqlite-input", str(input_path), "--sqlite-store", str(store), "--sqlite-output", str(output)]
                    else:
                        command = [str(binaries[arm]), "--input", str(input_path), "--store", str(store), "--out", str(output)]
                    load_before = os.getloadavg()
                    with log_path.open("w") as log:
                        run_timed([*time_arguments(platform.system()), *command], log, min(240, remaining))
                    load_after = os.getloadavg()
                    raw = json.loads(output.read_text())
                    actual = {row["id"]: row["results"] for row in raw["rows"]}
                    if set(actual) != set(qrels):
                        raise ValueError("query identity mismatch")
                    if arm in binaries:
                        if reference is None:
                            reference = actual
                        if actual != reference:
                            raise ValueError("HMS exact result parity failed")
                    qualities[arm] = quality(raw["rows"], qrels)
                    cells[arm].append([row["elapsed_ns"] for row in raw["rows"]])
                    peak = peak_rss_bytes(log_path.read_text(), platform.system())
                    if peak > 8 * 1024 ** 3:
                        raise ValueError("8 GiB per-process memory budget exceeded")
                    record = {"dataset": dataset, "round": repetition, "arm": arm, "order": arms,
                              "load_before": load_before, "load_after": load_after,
                              "load_gate_met": load_before[0] < 3 and load_after[0] < 3,
                              "peak_process_rss_bytes": peak, "open_ns": raw["open_ns"], "ingest": raw["ingest"]}
                    with gzip.open(output.with_suffix(".json.gz"), "wb") as compressed:
                        compressed.write(output.read_bytes())
                    output.unlink()
                    record.update(artifact=output.with_suffix(".json.gz").name, sha256=sha(output.with_suffix(".json.gz")))
                    measurements.append(record)
                    write(folder / "measurements.json", measurements)
            latency = {arm: {"mean_ns": statistics.mean(itertools.chain.from_iterable(rounds)),
                             "p95_ns": quantile(list(itertools.chain.from_iterable(rounds)), .95),
                             "p99_ns": quantile(list(itertools.chain.from_iterable(rounds)), .99)} for arm, rounds in cells.items()}
            summary[dataset] = {"queries": len(qrels), "quality": qualities, "latency": latency,
                                "exact_hms_query_pairs": len(qrels) * args.rounds,
                                "baseline_over_candidate_mean": latency["baseline"]["mean_ns"] / latency["candidate"]["mean_ns"],
                                "sqlite_over_candidate_mean": latency["sqlite"]["mean_ns"] / latency["candidate"]["mean_ns"]}
            write(folder / "summary.json", {"results": summary, "load_gate_met": all(row["load_gate_met"] for row in measurements)})
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
