#!/usr/bin/env python3
# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Derive phrase candidate populations from saved, paired Boolean counts."""
import json
from pathlib import Path
import statistics


def main():
    root = Path(__file__).parent
    queries = None
    repetitions = []
    for path in sorted(root.glob("COUNT-*.json")):
        record = json.loads(path.read_text())
        rows = record["samples"]
        counts = {row["query"]: row["count"] for row in rows}
        if queries is None:
            queries = counts
        if queries != counts or len(counts) != 962:
            raise ValueError("query counts changed between repetitions")
        repetitions.append(record["repetition"])
    if len(set(repetitions)) != 3:
        raise ValueError("three independent count receipts are required")
    results = []
    for row in rows:
        if row["tags"][0] != "phrase":
            continue
        words = row["query"].strip('"').split()
        union = queries[" ".join(words)]
        conjunction = queries[" ".join("+" + word for word in words)]
        results.append(dict(
            phrase=row["query"], union_candidates=union,
            conjunction_candidates=conjunction, phrase_matches=row["count"],
            union_over_conjunction=union / conjunction if conjunction else None,
        ))
    ratios = [row["union_over_conjunction"] for row in results if row["union_over_conjunction"] is not None]
    union = sum(row["union_candidates"] for row in results)
    conjunction = sum(row["conjunction_candidates"] for row in results)
    output = dict(
        engine_sha="46d20abeecece9d2e46686170d6ba4c12fae741a",
        adapter_revision="0aef786", input_articles=600000, indexed_articles=548246,
        corpus_scope="fixed first-600000 lines, not the full benchmark corpus",
        query_sha256="a8e8f2c1a4bede53063de22fb13be83b585683da1c28208da4921de73ce254bd",
        count_receipts=[path.name for path in sorted(root.glob("COUNT-*.json"))],
        paired_phrases=len(results), sum_union_candidates=union,
        sum_conjunction_candidates=conjunction,
        aggregate_candidate_ratio=union / conjunction,
        median_nonzero_candidate_ratio=statistics.median(ratios),
        maximum_nonzero_candidate_ratio=max(ratios),
        interpretation="Measured match populations from saved count receipts. Source inspection identifies union enumeration in the old single optional phrase path. Restricting candidates to a term conjunction is exact. These ratios are not measured executor counters or predictions of elapsed speedup.",
        queries=results,
    )
    (root / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "queries"}))


if __name__ == "__main__":
    main()
