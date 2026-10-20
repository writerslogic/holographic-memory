# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Independent arrival accounting, phase, atomic source/embedding and restart oracle."""

import json
import math
from pathlib import Path

from check_compaction_updates import expected, quantile, require, rows


def distribution(values):
    return {"count": len(values), "p50_ns": quantile(values, .5),
            "p99_ns": quantile(values, .99), "max_ns": max(values)} if values else {"count": 0}


def result(row, documents, query_pair):
    identity = row["documentId"]
    require(isinstance(identity, str) and identity.startswith("d") and identity[1:].isdigit(), "bad ID")
    number = int(identity[1:])
    require(0 <= number < len(documents) and documents[number]["id"] == identity, "foreign ID")
    require(row["id"] == f"hms:chunk:{len(identity)}:{identity}:0", "bad chunk ID")
    revision = int(row["version"])
    require(0 <= revision <= 256 and row["version"] == str(revision), "bad revision")
    for field, value in expected(documents[number], revision).items():
        require(row[field] == value, f"torn {field}")
    require(row["startByte"] == 0 and row["endByte"] == len(row["text"].encode()), "bad offsets")
    require(all(type(row[key]) in (int, float) and math.isfinite(row[key])
                for key in ("score", "lexicalScore", "semanticScore")), "nonfinite score")
    semantic = float((number // 2 + revision) % 128 == query_pair % 128)
    require(row["semanticScore"] == semantic, "source/embedding revision torn")
    return number, revision


def possible_states(operations, selected, start, end):
    state = dict.fromkeys(selected, 0)
    candidates = []
    for op in operations:
        if op["start_ns"] >= end:
            break
        affected = set(range(op["first"], op["first"] + op["count"])) & selected
        if not affected:
            continue
        if op["end_ns"] > start:
            candidates.append(state.copy())
        state.update(dict.fromkeys(affected, op["revision"]))
    candidates.append(state)
    return candidates


def load(folder):
    return {"report": json.loads((folder / "report.json").read_text()),
            "traces": json.loads((folder / "traces.json").read_text()),
            "arrivals": rows(folder, "arrivals.jsonl"), "operations": rows(folder, "operations.jsonl"),
            "readers": [rows(folder, f"reader-{reader}.jsonl") for reader in range(4)],
            "before": rows(folder, "before-restart.jsonl"), "after": rows(folder, "after-restart.jsonl")}


def check_data(data, documents, arm):
    report = data["report"]
    n, batch = report["documents"], report["batch"]
    require(n == len(documents) and n in (5182, 20728) and batch in (2, 64), "bad workload")
    operation_count = 256 if batch == 2 else 64
    require(report["schema_version"] == 1 and report["operations"] == operation_count, "bad report")
    require((report["readers"], report["arrivals"], report["interval_ns"], report["queue_capacity"])
            == (4, 4800, 1666667, 64), "arrival or queue budget changed")
    require(report["synthetic_embeddings"] is True and report["retrieval_quality_measured"] is False,
            "bad scope")
    operations = data["operations"]
    require(len(operations) == operation_count, "operation count changed")
    previous = 0
    for revision, op in enumerate(operations, 1):
        require((op["revision"], op["first"], op["count"]) ==
                (revision, ((revision - 1) * batch) % 128, batch), "operation order changed")
        require(type(op["start_ns"]) is int and type(op["end_ns"]) is int
                and previous <= op["start_ns"] < op["end_ns"], "bad operation interval")
        previous = op["end_ns"]
    compact = report["compact"]
    require(operations[0]["end_ns"] <= compact["start_ns"] < compact["end_ns"], "bad compaction")
    commits = sorted([row for row in data["traces"] if row["kind"] == "commit"],
                     key=lambda row: row["wait_ns"])
    finals = [row for row in data["traces"] if row["kind"] == "compact-final"]
    require(len(data["traces"]) == operation_count + 1 and len(commits) == operation_count
            and len(finals) == 1, "missing or foreign trace")
    for op, trace in zip(operations, commits):
        require(trace["count"] == batch, "trace batch changed")
        chain = [op["start_ns"], trace["wait_ns"], trace["acquired_ns"], trace["append_start_ns"],
                 trace["append_end_ns"], trace["publication_end_ns"], trace["released_ns"], op["end_ns"]]
        require(all(type(value) is int for value in chain) and all(a <= b for a, b in zip(chain, chain[1:])),
                "trace order changed")
        require(trace["append_start_ns"] < trace["append_end_ns"] and
                trace["exclusion_start_ns"] < trace["publication_end_ns"], "empty trace interval")
        if arm == "legacy-reader-lock":
            require(trace["exclusion_start_ns"] == trace["acquired_ns"], "baseline exclusion changed")
        else:
            require(trace["append_end_ns"] <= trace["exclusion_start_ns"] <= trace["publication_end_ns"],
                    "candidate exclusion overlaps append")
    final = finals[0]
    require(final["count"] == final["append_start_ns"] == final["append_end_ns"] == final["exclusion_start_ns"] == 0,
            "maintenance misclassified as reader exclusion")
    require(compact["start_ns"] <= final["wait_ns"] <= final["acquired_ns"]
            < final["publication_end_ns"] <= final["released_ns"] <= compact["end_ns"], "bad final lease")
    arrivals = data["arrivals"]
    require(len(arrivals) == 4800, "arrival count changed")
    previous = 0
    for sequence, row in enumerate(arrivals):
        require(row["sequence"] == sequence and row["scheduled_ns"] ==
                report["scheduled_base_ns"] + sequence * 1666667, "arrival schedule changed")
        require(type(row["accepted"]) is bool and previous <= row["offered_ns"]
                and row["scheduled_ns"] <= row["offered_ns"] <= row["attempt_end_ns"], "bad arrival clock")
        previous = row["attempt_end_ns"]
    accepted = {row["sequence"] for row in arrivals if row["accepted"]}
    found, queries = set(), []
    for reader, records in enumerate(data["readers"]):
        previous = 0
        previous_sequence = -1
        for query in records:
            sequence, pair = query["sequence"], query["pair"]
            require(type(sequence) is int and 0 <= sequence < 4800 and sequence > previous_sequence
                    and sequence % 4 == reader and query["reader"] == reader
                    and pair == (sequence // 4 + reader) % 64, "query sequence changed")
            require(sequence in accepted and sequence not in found, "unaccepted or duplicate query")
            arrival = arrivals[sequence]
            require(query["scheduled_ns"] == arrival["scheduled_ns"] and
                    max(arrival["offered_ns"], previous) <= query["dequeued_ns"]
                    <= query["start_ns"] < query["end_ns"], "bad query clock")
            count = 64 if batch == 64 and sequence % 16 == 0 else 2
            first = (pair // 32) * 64 if count == 64 else pair * 2
            selected = set(range(first, first + count))
            results = [result(row, documents, pair) for row in query["results"]]
            require(len(results) == len(dict(results)) == count, "missing or duplicate result")
            require(dict(results) in possible_states(operations, selected, query["start_ns"], query["end_ns"]),
                    "stale or torn whole batch")
            previous, previous_sequence = query["end_ns"], sequence
            found.add(sequence)
            queries.append(query)
    require(found == accepted, "accepted request lost")
    require(bool(queries) and any(q["start_ns"] < compact["end_ns"] and q["end_ns"] > compact["start_ns"]
                                 for q in queries), "no compaction overlap")
    require(all(any(q["reader"] == reader and q["start_ns"] > compact["end_ns"] for q in queries)
                for reader in range(4)), "reader did not extend beyond compaction")
    before, after = data["before"], data["after"]
    require(len(before) == len(after) == n, "restart count changed")
    last = max(previous, compact["end_ns"], operations[-1]["end_ns"], max(q["end_ns"] for q in queries))
    final_state = possible_states(operations, set(range(128)), last + 1, last + 2)[-1]
    for number, (left, right) in enumerate(zip(before, after)):
        require(left["id"] == right["id"] == number and last <= left["start_ns"] < left["end_ns"]
                < right["start_ns"] < right["end_ns"], "bad restart clock or ID")
        require(left["results"] == right["results"], "restart result changed")
        require(len(left["results"]) == 1 and result(left["results"][0], documents, number // 2)
                == (number, final_state.get(number, 0)), "acknowledged state lost")
    service = [q["end_ns"] - q["start_ns"] for q in queries]
    response = [q["end_ns"] - q["scheduled_ns"] for q in queries]
    queue = [q["start_ns"] - arrivals[q["sequence"]]["offered_ns"] for q in queries]
    producer = [row["offered_ns"] - row["scheduled_ns"] for row in arrivals]
    overlap = lambda q, start, end: q["start_ns"] < end and q["end_ns"] > start
    append_only = [q for q in queries if any(t["append_start_ns"] <= q["start_ns"]
                   < q["end_ns"] <= t["append_end_ns"] for t in commits)]
    exclusion_overlap = [q for q in queries if any(overlap(q, t["exclusion_start_ns"], t["released_ns"])
                                                 for t in commits)]
    return {"checked": True, "documents": n, "batch": batch, "offered": 4800,
            "accepted": len(queries), "dropped": 4800 - len(queries), "exact_restart_pairs": n,
            "operations_checked": operation_count, "whole_bulk_queries": sum(len(q["results"]) == 64 for q in queries),
            "service": distribution(service), "scheduled_response": distribution(response),
            "queue_delay": distribution(queue), "producer_lag": distribution(producer),
            "writes": distribution([op["end_ns"] - op["start_ns"] for op in operations]),
            "append": distribution([t["append_end_ns"] - t["append_start_ns"] for t in commits]),
            "reader_exclusion": distribution([t["released_ns"] - t["exclusion_start_ns"] for t in commits]),
            "writer_lease_wait": distribution([t["acquired_ns"] - t["wait_ns"] for t in commits]),
            "publication_wait": distribution([t["exclusion_start_ns"] - t["append_end_ns"] for t in commits])
                if arm != "legacy-reader-lock" else None,
            "completed_wholly_during_append": len(append_only), "exclusion_overlapping_queries": len(exclusion_overlap),
            "exclusion_overlapping_service": distribution([q["end_ns"] - q["start_ns"] for q in exclusion_overlap]),
            "compaction_ns": compact["end_ns"] - compact["start_ns"],
            "final_maintenance_lease_ns": final["released_ns"] - final["acquired_ns"],
            "final_maintenance_wait_ns": final["acquired_ns"] - final["wait_ns"]}


def check(folder: Path, input_path: Path, arm: str):
    return check_data(load(folder), json.loads(input_path.read_text()), arm)
