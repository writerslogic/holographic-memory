# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Independent acknowledged-update, atomic-batch and restart oracle."""

import argparse
import gzip
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(folder, name):
    path = folder / name
    if path.exists():
        with path.open() as stream:
            return [json.loads(line) for line in stream]
    with gzip.open(str(path) + ".gz", "rt") as stream:
        return [json.loads(line) for line in stream]


def quantile(values, fraction):
    values = sorted(values)
    require(bool(values), "empty measurements")
    return values[math.ceil(len(values) * fraction) - 1]


def expected(document, revision):
    return {
        "version": str(revision),
        "text": document["text"] + (f" revision{revision}" if revision else ""),
        "sourceUri": document["sourceUri"] + (f":revision{revision}" if revision else ""),
        "metadata": document["metadata"] | {"revision": revision},
    }


def result(row, documents):
    identity = row["documentId"]
    require(identity.startswith("d") and identity[1:].isdigit(), "invalid document ID")
    number = int(identity[1:])
    require(0 <= number < len(documents) and identity == documents[number]["id"], "foreign ID")
    require(row["id"] == f"hms:chunk:{len(identity)}:{identity}:0", "wrong chunk ID")
    revision = int(row["version"])
    require(0 <= revision <= 256 and str(revision) == row["version"], "invalid revision")
    for field, value in expected(documents[number], revision).items():
        require(row[field] == value, f"torn {field}")
    require(row["startByte"] == 0 and row["endByte"] == len(row["text"].encode()), "wrong offsets")
    require(all(type(row[k]) in (int, float) and math.isfinite(row[k])
                for k in ("score", "lexicalScore")), "invalid score")
    require(row["semanticScore"] is None, "unexpected semantic score")
    return number, revision


def states(operations, pair, start, end):
    state = {pair * 2: 0, pair * 2 + 1: 0}
    candidates = []
    for operation in operations:
        if operation["pair"] != pair:
            continue
        if operation["start_ns"] >= end:
            break
        if operation["end_ns"] > start:
            candidates.append(state.copy())
        if operation["kind"] == "replace":
            state = {pair * 2: operation["revision"], pair * 2 + 1: operation["revision"]}
        else:
            state.pop(pair * 2)
    candidates.append(state)
    return candidates


def check_queries(queries, documents, operations, reader, count, compact=None):
    require(len(queries) == count, "query count changed")
    previous_end = 0
    latencies, overlapping = [], []
    for sequence, query in enumerate(queries):
        pair = (reader + sequence) % 16
        require((query["reader"], query["sequence"], query["pair"]) == (reader, sequence, pair),
                "query sequence changed")
        start, end = query["start_ns"], query["end_ns"]
        require(type(start) is int and type(end) is int and previous_end <= start < end,
                "invalid query interval")
        previous_end = end
        found = [result(row, documents) for row in query["results"]]
        require(len(dict(found)) == len(found), "duplicate result")
        require(dict(found) in states(operations, pair, start, end), "stale or torn query snapshot")
        latencies.append(end - start)
        if compact and start < compact["end_ns"] and end > compact["start_ns"]:
            overlapping.append(end - start)
    return {"queries": count, "p50_ns": quantile(latencies, .5),
            "p99_ns": quantile(latencies, .99), "max_ns": max(latencies),
            "overlapping_queries": len(overlapping), "overlapping_max_ns": max(overlapping, default=0)}


def check(folder, input_path):
    documents = json.loads(input_path.read_text())
    report = json.loads((folder / "report.json").read_text())
    require(report["schema_version"] == 1 and report["documents"] == len(documents), "wrong report")
    require((report["readers"], report["queries_per_reader"], report["updates"]) == (4, 1024, 256),
            "work budget changed")
    require(report["synthetic_updates"] is True and report["retrieval_quality_measured"] is False,
            "invalid scope")
    operations = rows(folder, "operations.jsonl")
    require(len(operations) == 320, "operation count changed")
    previous_end, index = 0, 0
    for revision in range(1, 257):
        for kind in (["replace", "delete"] if revision % 4 == 0 else ["replace"]):
            operation = operations[index]
            require((operation["kind"], operation["pair"], operation["revision"]) ==
                    (kind, (revision - 1) % 16, revision), "operation order changed")
            require(previous_end <= operation["start_ns"] < operation["end_ns"], "invalid operation interval")
            previous_end = operation["end_ns"]
            index += 1
    compact = report["compact"]
    require(operations[0]["end_ns"] <= compact["start_ns"] < compact["end_ns"], "invalid compaction")
    summary = {}
    for phase, count, journal in [("static", 256, []), ("updates", 1024, operations)]:
        summary[phase] = [check_queries(rows(folder, f"{phase}-{reader}.jsonl"), documents,
                                       journal, reader, count, compact if journal else None)
                          for reader in range(4)]
    require(sum(row["overlapping_queries"] for row in summary["updates"]) > 0,
            "no queries overlapped compaction")
    before, after = rows(folder, "before-restart.jsonl"), rows(folder, "after-restart.jsonl")
    require(len(before) == len(after) == len(documents), "snapshot count changed")
    for number, (left, right) in enumerate(zip(before, after)):
        require(left["id"] == right["id"] == number, "snapshot ID changed")
        require(left["start_ns"] >= previous_end and right["start_ns"] > left["end_ns"],
                "snapshot preceded acknowledgment")
        require(left["results"] == right["results"], "restart result changed")
        found = [result(row, documents) for row in left["results"]]
        state = states(operations, number // 2, previous_end + 1, previous_end + 2)[-1]
        require(dict(found) == {k: v for k, v in state.items() if k == number},
                "acknowledged final state lost")
    durations = [row["end_ns"] - row["start_ns"] for row in operations]
    inside = [row for row in operations if row["start_ns"] >= compact["start_ns"]
              and row["end_ns"] <= compact["end_ns"]]
    return {"checked": True, "documents": len(documents), "queries_checked": 5120,
            "exact_restart_pairs": len(documents), "operations_checked": len(operations),
            "compaction_ns": compact["end_ns"] - compact["start_ns"],
            "fully_overlapping_operations": len(inside), "operations": {
                "p50_ns": quantile(durations, .5), "p99_ns": quantile(durations, .99),
                "max_ns": max(durations)}, "phases": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.out.open("x") as stream:
        stream.write(json.dumps(check(args.folder, args.input), indent=2) + "\n")


if __name__ == "__main__":
    main()
