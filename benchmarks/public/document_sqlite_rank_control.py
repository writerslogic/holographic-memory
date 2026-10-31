# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Compare WARM with SQLite's documented native top-k rank path on the same indexes."""

import argparse
import fcntl
import gzip
import itertools
import json
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import sys
import time

from document_competition import quality, sha, write
from document_search_compare import peak_rss_bytes, quantile, run_timed, time_arguments


def sqlite_rank(input_path, store, output):
    data = json.loads(input_path.read_text())
    start = time.perf_counter_ns()
    connection = sqlite3.connect(store)
    connection.execute("PRAGMA cache_size=-65536")
    identifiers = dict(connection.execute("SELECT rowid, id FROM identities"))
    open_ns = time.perf_counter_ns() - start

    def search(query):
        words = sorted({"".join(chars) for kind, chars in itertools.groupby(query["text"].lower(), str.isalnum) if kind})
        expression = " OR ".join('"' + word + '"' for word in words)
        if not expression:
            return []
        results = connection.execute(
            "SELECT rowid, rank FROM docs WHERE docs MATCH ? ORDER BY rank LIMIT ?",
            (expression, query["options"]["k"]),
        ).fetchall()
        return [{"documentId": identifiers[rowid], "lexicalScore": -score} for rowid, score in results]

    for query in data["queries"][:16]:
        search(query)
    rows = []
    for query in data["queries"]:
        start = time.perf_counter_ns()
        results = search(query)
        rows.append({"id": query["id"], "elapsed_ns": time.perf_counter_ns() - start, "results": results})
    connection.close()
    write(output, {"documents": len(data["documents"]), "open_ns": open_ns, "ingest": False,
                   "sqlite_version": sqlite3.sqlite_version, "rows": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--sqlite-input", type=Path)
    parser.add_argument("--sqlite-store", type=Path)
    parser.add_argument("--sqlite-output", type=Path)
    args = parser.parse_args()
    if args.sqlite_input:
        sqlite_rank(args.sqlite_input, args.sqlite_store, args.sqlite_output)
        return
    if not 1 <= args.rounds <= 3:
        raise ValueError("rounds outside bounded budget")
    campaign = args.campaign.resolve()
    folder = campaign / "rank_optimized"
    folder.mkdir(exist_ok=False)
    protocol = {"rounds": args.rounds, "sqlite_version": sqlite3.sqlite_version,
                "query": "SELECT rowid, rank FROM docs WHERE docs MATCH ? ORDER BY rank LIMIT ?",
                "sqlite_primary_docs": "https://www.sqlite.org/fts5.html#sorting_by_auxiliary_function_results",
                "cache_size_kib": 65536, "tie_order": "SQLite native rank traversal",
                "index_source": "same optimized persistent contentless FTS5 indexes from initial campaign",
                "candidate_binary_sha256": sha(campaign / "candidate-probe")}
    write(folder / "protocol.json", protocol)
    records, summary = [], {}
    with Path("./.warm-target/timing.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.monotonic()
        for dataset, split in [("scifact", "train"), ("nfcorpus", "dev")]:
            input_path = campaign / f"{dataset}-{split}-input.json"
            qrels = json.loads((campaign / f"{dataset}-{split}-qrels.json").read_text())
            baseline = json.loads(gzip.decompress((campaign / f"{dataset}-0-baseline.json.gz").read_bytes()))
            reference = {row["id"]: row["results"] for row in baseline["rows"]}
            cells, qualities = {"candidate": [], "sqlite_rank": []}, {}
            for repetition in range(args.rounds):
                order = ["candidate", "sqlite_rank"] if repetition % 2 == 0 else ["sqlite_rank", "candidate"]
                for arm in order:
                    output = folder / f"{dataset}-{repetition}-{arm}.json"
                    if arm == "candidate":
                        command = [str(campaign / "candidate-probe"), "--input", str(input_path),
                                   "--store", str(campaign / f"{dataset}-candidate-store"), "--out", str(output)]
                    else:
                        command = [sys.executable, str(Path(__file__).resolve()), "--campaign", str(campaign),
                                   "--sqlite-input", str(input_path), "--sqlite-store", str(campaign / f"{dataset}-sqlite-store"),
                                   "--sqlite-output", str(output)]
                    remaining = 360 - (time.monotonic() - started)
                    if remaining <= 0:
                        raise RuntimeError("360-second follow-up budget exceeded")
                    log_path = output.with_suffix(".log")
                    before = os.getloadavg()
                    with log_path.open("w") as log:
                        run_timed([*time_arguments(platform.system()), *command], log, min(90, remaining))
                    after = os.getloadavg()
                    raw = json.loads(output.read_text())
                    actual = {row["id"]: row["results"] for row in raw["rows"]}
                    assert set(actual) == set(qrels)
                    if arm == "candidate":
                        assert actual == reference
                    qualities[arm] = quality(raw["rows"], qrels)
                    cells[arm].append([row["elapsed_ns"] for row in raw["rows"]])
                    peak = peak_rss_bytes(log_path.read_text(), platform.system())
                    assert peak <= 8 * 1024 ** 3
                    with gzip.open(output.with_suffix(".json.gz"), "wb") as compressed:
                        compressed.write(output.read_bytes())
                    output.unlink()
                    records.append({"dataset": dataset, "round": repetition, "arm": arm, "order": order,
                                    "load_before": before, "load_after": after,
                                    "load_gate_met": before[0] < 3 and after[0] < 3,
                                    "peak_process_rss_bytes": peak, "open_ns": raw["open_ns"],
                                    "artifact": output.with_suffix(".json.gz").name, "sha256": sha(output.with_suffix(".json.gz"))})
                    write(folder / "measurements.json", records)
            latency = {arm: {"mean_ns": statistics.mean(itertools.chain.from_iterable(rounds)),
                             "p95_ns": quantile(list(itertools.chain.from_iterable(rounds)), .95),
                             "p99_ns": quantile(list(itertools.chain.from_iterable(rounds)), .99)} for arm, rounds in cells.items()}
            summary[dataset] = {"queries": len(qrels), "quality": qualities, "latency": latency,
                                "sqlite_over_candidate_mean": latency["sqlite_rank"]["mean_ns"] / latency["candidate"]["mean_ns"]}
            write(folder / "summary.json", {"results": summary, "load_gate_met": all(row["load_gate_met"] for row in records)})
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
