# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///
"""Independent synthetic lifecycle and acknowledged-update oracle."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def quantile(values, fraction):
    values = sorted(values)
    return values[math.ceil(fraction * len(values)) - 1]


def result(row, n, versions, group=None):
    document = row["documentId"]
    require(isinstance(document, str) and document.startswith("d"), "bad document ID")
    number = int(document[1:])
    require(0 <= number < n and document == f"d{number:08}", "foreign document ID")
    require(row["id"] == f"hms:chunk:{len(document)}:{document}:0", "chunk ID mismatch")
    version = row["version"]
    require(version in versions(number), "stale or impossible version")
    require(row["text"] == f"common token{number} revision{version}", "torn text/version")
    require(row["metadata"] == {"group": number % 8}, "metadata mismatch")
    require(row["sourceUri"] == f"fixture:source:{number % 8}", "source mismatch")
    require(group is None or number % 8 == group, "filter violation")
    require(row["startByte"] == 0 and row["endByte"] == len(row["text"]), "offset mismatch")
    require(all(math.isfinite(row[key]) for key in ("score", "lexicalScore")), "nonfinite score")
    return number


def check(raw):
    n, readers = raw["documents"], raw["readers"]
    require(type(n) is int and 1023 <= n <= 10000, "bad corpus size")
    require(type(readers) is int and 1 <= readers <= 8, "bad reader count")
    require(raw["timing_claim_eligible"] is False, "invalid timing claim")
    require([p["phase"] for p in raw["phases"]] == ["replace", "delete", "compact"], "missing phases")
    phases = [{"phase": "static", "queries": raw["static_queries"], "operations": []}, *raw["phases"]]
    summary = []
    previous_end = 0
    for phase in phases:
        name, queries, operations = (phase[k] for k in ("phase", "queries", "operations"))
        count = {"static": 0, "replace": 64, "delete": n // 2 + 1, "compact": 1}[name]
        require(len(operations) == count, "missing operations")
        for i, operation in enumerate(operations):
            require(operation["id"] == i, "operation order mismatch")
            require(previous_end <= operation["start_ns"] < operation["end_ns"], "invalid operation interval")
            previous_end = operation["end_ns"]
        by_id = {o["id"]: o for o in operations}
        observed, overlaps, latencies = {}, 0, []
        for query in queries:
            reader, sequence = query["reader"], query["sequence"]
            require(type(reader) is int and 0 <= reader < readers, "invalid reader")
            prior = observed.get(reader, (-1, 0))
            require(sequence == prior[0] + 1, "missing query sequence")
            start, end = query["start_ns"], query["end_ns"]
            require(prior[1] <= start < end, "invalid query interval")
            observed[reader] = (sequence, end)
            group = (reader + sequence) % 8
            require(query["group"] == group, "query filter changed")

            def versions(number):
                if name == "replace" and number < 64:
                    operation = by_id[number]
                    if start >= operation["end_ns"]:
                        return {"2"}
                    if end <= operation["start_ns"]:
                        return {"1"}
                    return {"1", "2"}
                return {"2" if name != "static" and number < 64 else "1"}

            ids = [result(r, n, versions, group) for r in query["results"]]
            require(len(ids) == 10 and len(set(ids)) == 10, "missing or duplicate search results")
            for number in ids:
                if name == "delete" and number in by_id:
                    require(start < by_id[number]["end_ns"], "acknowledged deletion returned")
                if name == "compact":
                    require(number > n // 2, "deleted result during compaction")
            overlaps += any(start < op["end_ns"] and end > op["start_ns"] for op in operations)
            latencies.append(end - start)
        if name == "static":
            require(len(queries) == 512, "missing static queries")
        else:
            require(len(observed) == readers and all(v[0] >= 127 for v in observed.values()), "reader coverage missing")
            require(overlaps > 0, "no observed query/mutation overlap")
        summary.append({"phase": name, "queries": len(queries), "overlapping_queries": overlaps,
                        "query_p95_ns": quantile(latencies, .95), "query_p99_ns": quantile(latencies, .99),
                        "operation_max_ns": max((o["end_ns"] - o["start_ns"] for o in operations), default=0)})
    require(raw["before_restart"] == raw["after_restart"], "restart output changed")
    for snapshot in (raw["before_restart"], raw["after_restart"]):
        require(len(snapshot) == n, "incomplete snapshot")
        for number, row in enumerate(snapshot):
            require(row["id"] == number, "snapshot ID changed")
            require(len(row["results"]) == (number > n // 2), "deleted or missing snapshot record")
            for found in row["results"]:
                require(result(found, n, lambda _: {"1"}) == number, "wrong snapshot document")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.report.read_text())
    summary = check(raw)
    rejected = []
    for fault in ("version", "filter", "source", "restart", "missing_operation"):
        bad = copy.deepcopy(raw)
        row = bad["static_queries"][0]["results"][0]
        if fault == "version":
            row["version"] = "999"
        elif fault == "filter":
            bad["static_queries"][0]["group"] = 7
        elif fault == "source":
            row["sourceUri"] = "wrong"
        elif fault == "restart":
            bad["after_restart"].pop()
        else:
            bad["phases"][0]["operations"].pop()
        try:
            check(bad)
        except ValueError:
            rejected.append(fault)
        else:
            raise ValueError(f"checker accepted {fault}")
    args.out.write_text(json.dumps({"checked": True, "report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": summary, "corruptions_rejected": rejected,
        "scope": "synthetic lexical diagnostic; no interrupted transaction or corruption recovery test; no timing superiority claim"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
