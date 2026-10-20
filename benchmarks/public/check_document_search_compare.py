# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = "==3.13.9"
# dependencies = []
# ///

import argparse
import copy
import gzip
import hashlib
import itertools
import json
from pathlib import Path


def validate_report(report, n):
    expected = set(itertools.product(("lexical", "semantic", "hybrid"), (False, True), range(32), range(3)))
    actual = {}
    if report["documents"] != n or report["dimensions"] != 384 or report["synthetic"] is not True:
        raise ValueError("wrong corpus")
    for row in report["rows"]:
        key = row["mode"], row["filtered"], row["query"], row["repetition"]
        if key not in expected or key in actual or type(row["elapsed_ns"]) is not int or row["elapsed_ns"] <= 0:
            raise ValueError("invalid query identity/timing")
        seen = set()
        for hit in row["results"]:
            doc_id = hit["documentId"]
            if doc_id in seen or not doc_id.startswith("d"):
                raise ValueError("invalid result identity")
            seen.add(doc_id)
            number = int(doc_id[1:])
            if not 0 <= number < n or doc_id != f"d{number:08}" or hit["version"] != "1":
                raise ValueError("result outside source corpus/version")
            if hit["metadata"] != {"group":number % 100}:
                raise ValueError("source metadata differs")
            if row["filtered"] and number % 100 != row["query"] % 100:
                raise ValueError("filter violation")
            text = f"common text running topic{number%100} rare{number} alpha beta gamma delta epsilon"
            if hit["text"] != text or hit["startByte"] != 0 or hit["endByte"] != len(text):
                raise ValueError("source text/span differs")
        actual[key] = row["results"]
    if set(actual) != expected:
        raise ValueError("incomplete query grid")
    return actual


def main():
    parser = argparse.ArgumentParser(description="Independently check all paired source outputs, hashes and deliberate corruptions.")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.directory / "summary.json").read_text())
    reports = {}
    for run in summary["runs"]:
        path = args.directory / run["artifact"]
        if path.parent != args.directory or hashlib.sha256(path.read_bytes()).hexdigest() != run["sha256"]:
            raise ValueError("raw artifact path/hash mismatch")
        with gzip.open(path,"rt") as handle:
            report = json.load(handle)
        key = run["n"], run["round"], run["arm"]
        if key in reports:
            raise ValueError("duplicate run")
        reports[key] = validate_report(report, run["n"])
    expected = set(itertools.product((1000,10000),range(summary["rounds"]),("baseline","candidate")))
    if set(reports) != expected:
        raise ValueError("missing paired run")
    for n, repetition in itertools.product((1000,10000),range(summary["rounds"])):
        if reports[n,repetition,"baseline"] != reports[n,repetition,"candidate"]:
            raise ValueError("candidate changes exact results")
    corruptions = []
    for kind in ("missing_query","duplicate_query","filter","source","version"):
        changed = copy.deepcopy(report)
        if kind == "missing_query":
            changed["rows"].pop()
        elif kind == "duplicate_query":
            changed["rows"].append(changed["rows"][0])
        elif kind == "source":
            changed["rows"][0]["results"][0]["text"] = "tampered"
        elif kind == "version":
            changed["rows"][0]["results"][0]["version"] = "2"
        else:
            row = next(row for row in changed["rows"] if row["filtered"])
            row["query"] = (row["query"] + 1) % 32
        try:
            validate_report(changed, report["documents"])
        except ValueError:
            corruptions.append(kind)
        else:
            raise ValueError(f"accepted corruption: {kind}")
    result = {"exact_results_and_source_fields_verified":True,"query_pairs":576*2*summary["rounds"],
              "artifact_count":len(reports),"tamper_rejections":corruptions,
              "summary_sha256":hashlib.sha256((args.directory/"summary.json").read_bytes()).hexdigest(),
              "checker_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "timing_claim_established":False,"serving_memory_claim_established":False}
    (args.directory/"independent_check.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
