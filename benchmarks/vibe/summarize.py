# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarize the measured diagnostic without treating missing methods as wins."""

import argparse
import hashlib
import json
from pathlib import Path


def summarize(root):
    validation = json.loads((root / "diagnostic/validation.json").read_text())
    checked = {record["file"]: record for record in validation["results"] if record["passed"]}
    result = {"schema_version": 1, "scope": "100000-row prefixes, 256 original queries; exploratory, not an official VIBE result",
              "quality": "VIBE distance-tolerant recall and strict neighbor-ID recall are distinct",
              "timing": "descriptive only: non-interleaved runs, concurrent builds, different hardware from VIBE",
              "laion_adapters": {}, "paired_native": {},
              "unmeasured": {
                  "pag": "blocked: pinned backend requires AVX-512; host lacks it",
                  "ngt-qg": "not run: required native CLI and dependencies are not installed",
                  "lorann": "failed after build: shell exit 132 (SIGILL); cause not established",
                  "pynndescent": "failed validation: duplicate returned IDs; this trial does not validate or invalidate published ranks",
              }}
    for path in sorted((root / "diagnostic").glob("laion-*.json")):
        payload = path.read_bytes()
        if path.name not in checked or hashlib.sha256(payload).hexdigest() != checked[path.name]["sha256"]:
            raise ValueError(f"result not independently checked: {path.name}")
        report = json.loads(payload)
        result["laion_adapters"][report["method"]] = {
            "source": str(path.relative_to(root)), "build": report["protocol"]["build"],
            "maximum_vibe_recall": max(row["recall"] for row in report["rows"]),
            "maximum_strict_id_recall": max(row["strict_id_recall"] for row in report["rows"]),
            "targets": report["targets"],
        }
    for dataset in ("arxiv", "laion", "inaturalist"):
        reports = [json.loads((root / f"native/{dataset}-native-{bits}.json").read_text()) for bits in (4, 8)]
        first, second = reports
        if first["k"] != 100 or second["k"] != 100 or first["n_queries"] != second["n_queries"]:
            raise ValueError("native runs differ in query count or result count")
        if first["build"]["graph_hash"] != second["build"]["graph_hash"]:
            raise ValueError("native ablation used different graphs")
        if first["dataset"] != second["dataset"]:
            raise ValueError("native ablation used different input data")
        configs = [{(row["params"]["ef"], row["params"]["rerank"]): row for row in report["sweep"]} for report in reports]
        if configs[0].keys() != configs[1].keys():
            raise ValueError("native ablation used different search configurations")
        result["paired_native"][dataset] = {
            "graph_hash": first["build"]["graph_hash"],
            "dimension": first["dataset"]["dim"], "rows": first["dataset"]["n_train"],
            "build_threads": first["build"]["threads"], "query_threads": 1,
            "index_bytes": {str(bits): report["index_bytes"] for bits, report in zip((4, 8), reports)},
            "maximum_strict_recall": {str(bits): max(row["recall_at_100"] for row in report["sweep"])
                                      for bits, report in zip((4, 8), reports)},
            "same_configuration": [
                {"ef": ef, "rerank": rerank, "strict_recall_4": configs[0][(ef, rerank)]["recall_at_100"],
                 "strict_recall_8": configs[1][(ef, rerank)]["recall_at_100"]}
                for ef, rerank in sorted(configs[0])],
            "timing_gate": "native ablation predates Linux load reporting fix; unknown load, no timing verdict",
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(summarize(args.results), indent=2) + "\n")
