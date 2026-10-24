# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics

import numpy as np

from probe import PhaseSketch, generate


def audit(directory):
    plan = json.loads(Path(__file__).with_name("PLAN.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    cases = [json.loads(line) for line in (directory / "cases.jsonl").read_text().splitlines()]
    expected = list(itertools.product(plan["scenarios"], plan["rows_per_block"], plan["seeds"]["data"]))
    assert [(c["scenario"], c["nominal_rows"], c["seed"]) for c in cases] == expected
    assert summary["cases_sha256"] == hashlib.sha256((directory / "cases.jsonl").read_bytes()).hexdigest()
    assert summary["plan_sha256"] == hashlib.sha256(Path(__file__).with_name("PLAN.json").read_bytes()).hexdigest()
    assert summary["source_sha256"] == hashlib.sha256(Path(__file__).with_name("probe.py").read_bytes()).hexdigest()
    for case in cases:
        banks, queries = generate(case["scenario"], case["nominal_rows"], case["seed"])
        source = np.concatenate(banks)
        assert hashlib.sha256(source.tobytes()).hexdigest() == case["source_sha256"]
        assert hashlib.sha256(queries.tobytes()).hexdigest() == case["query_sha256"]
        labels = np.repeat(np.arange(len(banks)), [len(b) for b in banks])
        truth = []
        for start in range(0, len(queries), 8):
            distance = ((queries[start:start + 8, None] - source[None]) ** 2).sum(axis=2)
            truth.extend(np.argsort(distance, axis=1, kind="stable")[:, :10].tolist())
        assert truth == case["top10_ids"]
        hits = np.array([np.bincount(labels[row], minlength=8) for row in truth])
        assert hits.tolist() == case["block_top10_hits"]
        assert len(case["methods"]) == 47
        for row in case["methods"]:
            order = np.array(row["order"])
            assert order.shape == (32, 8)
            assert np.array_equal(np.sort(order, axis=1), np.tile(np.arange(8), (32, 1)))
            for count in plan["fanouts"]:
                retained = sum(hits[q, order[q, :count]].sum() for q in range(32))
                assert row["coverage"][str(count)] == retained / 320
    eligible = []
    for width, sigma, method in itertools.product(plan["phase_widths"], plan["bandwidths"], ["phase_sum", "phase_mean"]):
        rows = [(c, r) for c in cases for r in c["methods"] if r["method"] == method
                and r["width"] == width and r["bandwidth"] == sigma]
        minimum = min(r["coverage"]["4"] for c, r in rows if c["scenario"] != "mixed")
        gain = statistics.mean(r["coverage"]["4"] - next(x["coverage"]["4"] for x in c["methods"] if x["method"] == "centroid")
                               for c, r in rows if c["scenario"] == "multimodal")
        eligible.append(minimum >= .9975 and gain >= .10)
    assert eligible == [r["eligible_for_separate_development_plan"] for r in summary["decisions"]]
    return {"cases": len(cases), "validated_method_records": sum(len(c["methods"]) for c in cases),
            "coordinatewise_truth_matches": True, "all_routing_coverages_recomputed": True,
            "frozen_decisions_match": True, "eligible_configurations": sum(eligible)}


def mutation_error():
    banks, query = generate("coherent", 2048, 17)
    live = {i: (block, row.copy()) for i, (block, row) in enumerate(
        (b, row) for b, bank in enumerate(banks) for row in bank)}
    omega = np.random.default_rng(101).normal(size=(32, 256))
    sketch = PhaseSketch(omega, 8, 1.5)
    for block, bank in enumerate(banks):
        sketch.change(block, bank)
    snapshots = []
    def check(name):
        reference = PhaseSketch(omega, 8, 1.5)
        for block in range(8):
            reference.change(block, np.array([row for owner, row in live.values() if owner == block]).reshape(-1, 32))
        assert np.array_equal(sketch.counts, reference.counts)
        values = {"cos_max_abs_error": float(np.abs(sketch.cos - reference.cos).max()),
                  "sin_max_abs_error": float(np.abs(sketch.sin - reference.sin).max()),
                  "query_max_abs_error": float(np.abs(sketch.score(query) - reference.score(query)).max())}
        assert max(values.values()) < 1e-9
        snapshots.append({"state": name, "live": len(live), **values})
    check("initial")
    deleted = {}
    for identifier in [0, 255, 256, 2047, 2048, 2049]:
        block, old = live.pop(identifier)
        deleted[identifier] = block
        sketch.change(block, old[None], -1)
    check("deleted")
    rng = np.random.default_rng(1217)
    for identifier in [1, 511, 2046, 2559, 2560]:
        block, old = live[identifier]
        new = rng.normal(size=32)
        sketch.change(block, old[None], -1)
        sketch.change(block, new[None])
        live[identifier] = (block, new)
    check("replaced")
    for identifier, block in deleted.items():
        new = rng.normal(size=32)
        sketch.change(block, new[None])
        live[identifier] = (block, new)
    check("reused")
    return snapshots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    result["mutation_roundoff"] = mutation_error()
    print(json.dumps(result, indent=2))
