# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.10"
# dependencies = ["aiohttp>=3.9", "tiktoken>=0.7"]
# ///
"""Audit saved LoCoMo retrievals and prepare an independently scored token-matched copy.

The original predictions remain untouched. Selection uses only ranked memory text
and the fixed token budget, never questions, reference answers, or existing grades.
Run the pinned official harness with --evaluate-only against the prepared directory.
Grade reuse requires the caller to verify identical answerer, judge, and scorer
settings; the receipt establishes retrieval identity, not those external settings.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import tempfile
from pathlib import Path

from mem0_protocol.grade_validation import complete_grade
from mem0_protocol.shim import fit_budget


def distribution(values: list[int]) -> dict:
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
    }


def cap_prediction(data: dict, encoding, budget: int) -> tuple[dict, int, int]:
    if type(budget) is not int or not 1 <= budget <= 100_000:
        raise ValueError("budget must be an integer in 1..100000")
    retrieval = data.get("retrieval")
    if not isinstance(retrieval, dict) or not isinstance(retrieval.get("search_results"), list):
        raise ValueError("prediction must contain retrieval.search_results")
    hits = retrieval["search_results"]
    if len(hits) > 200:
        raise ValueError("prediction exceeds the fixed top-200 protocol")
    if any(not isinstance(hit, dict) or not isinstance(hit.get("memory"), str) for hit in hits):
        raise ValueError("every retrieval result must contain a memory string")
    costs = [len(encoding.encode(hit["memory"])) for hit in hits]
    kept = fit_budget(hits, costs, budget)
    result = copy.deepcopy(data)
    result.pop("cutoff_results", None)
    result["retrieval"]["search_results"] = copy.deepcopy(kept)
    result["retrieval"]["total_results"] = len(kept)
    return result, sum(costs), sum(len(encoding.encode(hit["memory"])) for hit in kept)


def prepare(source: Path, destination: Path | None, encoding, budget: int, expected: int, reuse_unchanged: bool = False) -> dict:
    if type(expected) is not int or not 1 <= expected <= 1540:
        raise ValueError("expected question count must be in 1..1540")
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("source must be a prediction directory")
    if destination is not None:
        destination = destination.resolve()
        if destination == source or source in destination.parents or destination in source.parents:
            raise ValueError("source and destination must not overlap")
        if destination.exists():
            raise FileExistsError("destination already exists; preserve previous runs")
    paths = sorted(source.glob("conv*_q*.json"))
    if len(paths) != expected:
        raise ValueError(f"expected {expected} prediction files, found {len(paths)}")
    prepared, hashes, ids = {}, {}, set()
    before, after, counts_before, counts_after = [], [], [], []
    changed = empty = reused = 0
    for path in paths:
        if path.is_symlink() or path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("prediction must be a regular file of at most 16 MiB")
        raw = path.read_bytes()
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("question_id") != path.stem:
            raise ValueError(f"question id does not match filename: {path.name}")
        if data["question_id"] in ids:
            raise ValueError("duplicate question id")
        ids.add(data["question_id"])
        capped, old_tokens, new_tokens = cap_prediction(data, encoding, budget)
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        original_hits = data["retrieval"]["search_results"]
        kept_hits = capped["retrieval"]["search_results"]
        if reuse_unchanged and original_hits == kept_hits and data.get("cutoff_results"):
            capped["cutoff_results"] = copy.deepcopy(complete_grade(data, len(kept_hits)))
            reused += 1
        prepared[path.name] = json.dumps(capped, ensure_ascii=False, indent=2) + "\n"
        before.append(old_tokens)
        after.append(new_tokens)
        counts_before.append(len(original_hits))
        counts_after.append(len(kept_hits))
        changed += original_hits != kept_hits
        empty += not kept_hits
    receipt = {
        "schema_version": 1,
        "source": str(source),
        "destination": str(destination) if destination else None,
        "expected_questions": expected,
        "actual_questions": len(paths),
        "budget_tokens": budget,
        "encoding": encoding.name,
        "selection": "exact-S fit_budget: rank order, skip overflow",
        "reuse_unchanged_grades": reuse_unchanged,
        "reused_grades": reused,
        "reuse_rule": "only identical ranked memories with a complete top-200 grade; independent of correctness",
        "changed_retrievals": changed,
        "empty_retrievals_after": empty,
        "tokens_before": distribution(before),
        "tokens_after": distribution(after),
        "memory_counts_before": distribution(counts_before),
        "memory_counts_after": distribution(counts_after),
        "source_sha256": hashes,
        "prepared_sha256": {
            name: hashlib.sha256(content.encode()).hexdigest() for name, content in prepared.items()
        },
    }
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".locomo-budget-", dir=destination.parent) as staging:
            stage = Path(staging) / "predictions"
            stage.mkdir()
            for name, content in prepared.items():
                (stage / name).write_text(content, encoding="utf-8")
            (stage / "_token_budget_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            stage.rename(destination)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=7000)
    parser.add_argument("--expected", type=int, default=1540)
    parser.add_argument("--reuse-unchanged", action="store_true", help="Reuse complete grades only when token selection leaves retrieval identical")
    args = parser.parse_args()
    import tiktoken

    report = prepare(args.source, args.destination, tiktoken.get_encoding("o200k_base"), args.budget, args.expected, args.reuse_unchanged)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if not key.endswith("sha256")}, indent=2))


if __name__ == "__main__":
    main()
