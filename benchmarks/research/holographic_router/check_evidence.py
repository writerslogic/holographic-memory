# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
from pathlib import Path

from summarize import summarize
from recorded_source import verify_source


HERE = Path(__file__).resolve().parent


def check(directory):
    for name in ("pre-run-seal.json", "hybrid-pre-run-seal.json"):
        seal = json.loads((directory / name).read_text())
        for filename, expected in seal["files_sha256"].items():
            verify_source(HERE / filename, expected)
    if summarize(directory) != json.loads((directory / "aggregate.json").read_text()):
        raise ValueError("aggregate or negative-comparison mismatch")
    for name, source, plan in [("run-synthetic-v1", "probe.py", "PLAN.json"),
                               ("run-hybrid-v1", "hybrid_probe.py", "hybrid_PLAN.json")]:
        summary = json.loads((directory / name / "summary.json").read_text())
        verify_source(HERE / source, summary["source_sha256"])
        for key, path in [("plan_sha256", HERE / plan),
                          ("cases_sha256", directory / name / "cases.jsonl")]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != summary[key]:
                raise ValueError(f"result hash mismatch: {name}/{key}")
    return {"sealed_source_artifacts_unchanged": True, "result_hashes_match": True,
            "active_hybrid_source_change": "Remove unused coverage import only; exact original archived",
            "all_aggregates_and_negative_comparisons_recomputed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    print(json.dumps(check(parser.parse_args().directory), indent=2))
