# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
from pathlib import Path

from summarize import summarize


HERE = Path(__file__).resolve().parent


def check(directory):
    for name in ("pre-run-seal.json", "hybrid-pre-run-seal.json"):
        seal = json.loads((directory / name).read_text())
        for filename, expected in seal["files_sha256"].items():
            actual = hashlib.sha256((HERE / filename).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"sealed source changed: {filename}")
    if summarize(directory) != json.loads((directory / "aggregate.json").read_text()):
        raise ValueError("aggregate or negative-comparison mismatch")
    for name, source, plan in [("run-synthetic-v1", "probe.py", "PLAN.json"),
                               ("run-hybrid-v1", "hybrid_probe.py", "hybrid_PLAN.json")]:
        summary = json.loads((directory / name / "summary.json").read_text())
        for key, path in [("source_sha256", HERE / source), ("plan_sha256", HERE / plan),
                          ("cases_sha256", directory / name / "cases.jsonl")]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != summary[key]:
                raise ValueError(f"result hash mismatch: {name}/{key}")
    return {"sealed_sources_unchanged": True, "result_hashes_match": True,
            "all_aggregates_and_negative_comparisons_recomputed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    print(json.dumps(check(parser.parse_args().directory), indent=2))
