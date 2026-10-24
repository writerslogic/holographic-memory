# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics

import numpy as np

from hybrid_probe import make_case
from recorded_source import verify_source


HERE = Path(__file__).resolve().parent


def independent_witnesses(data, count):
    selected = [int(np.argmin(((data - data.mean(axis=0)) ** 2).sum(axis=1)))]
    while len(selected) < count:
        nearest = np.full(len(data), np.inf)
        for identifier in selected:
            nearest = np.minimum(nearest, ((data - data[identifier]) ** 2).sum(axis=1))
        nearest[selected] = -np.inf
        selected.append(int(np.argmax(nearest)))
    return selected


def audit(directory):
    plan = json.loads((HERE / "hybrid_PLAN.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    cases = [json.loads(line) for line in (directory / "cases.jsonl").read_text().splitlines()]
    verify_source(HERE / "hybrid_probe.py", summary["source_sha256"])
    for key, path in [("plan_sha256", HERE / "hybrid_PLAN.json"),
                      ("base_probe_sha256", HERE / "probe.py"),
                      ("cases_sha256", directory / "cases.jsonl")]:
        assert summary[key] == hashlib.sha256(path.read_bytes()).hexdigest()
    expected = list(itertools.product(plan["scenarios"], plan["nominal_rows"], plan["data_seeds"]))
    assert [(c["scenario"], c["nominal_rows"], c["seed"]) for c in cases] == expected
    assert summary["cases"] == len(cases)
    structured = []
    differences = []
    by_scenario = {}
    for case in cases:
        assert "skipped" not in case
        banks, queries = make_case(case["scenario"], case["nominal_rows"], case["seed"])
        source = np.concatenate(banks)
        assert hashlib.sha256(source.tobytes()).hexdigest() == case["source_sha256"]
        assert hashlib.sha256(queries.tobytes()).hexdigest() == case["query_sha256"]
        assert case["counts"] == [len(bank) for bank in banks]
        assert case["raw_bytes"] == source.nbytes > plan["phase_bytes"]["total"]
        distances = np.empty((32, len(source)))
        for start in range(0, len(queries), 8):
            distances[start:start + 8] = ((queries[start:start + 8, None] - source[None]) ** 2).sum(axis=2)
        truth = np.argsort(distances, axis=1, kind="stable")[:, :10]
        assert truth.tolist() == case["top10_ids"]
        labels = np.repeat(np.arange(8), case["counts"])
        hits = np.array([np.bincount(labels[row], minlength=8) for row in truth])
        assert hits.tolist() == case["block_top10_hits"]
        witness_ids = [independent_witnesses(bank, 48) for bank in banks]
        assert witness_ids == case["geometry_witness_ids"]
        rng = np.random.default_rng(plan["reservoir_seed"] + case["seed"])
        reservoir_ids = [rng.choice(len(bank), 48, replace=False).tolist() for bank in banks]
        assert reservoir_ids == case["reservoir_ids"]
        reference = {"oracle_hit_count": hits.astype(float)}
        for name, ids, size in [("geometry48", witness_ids, 48), ("geometry36", witness_ids, 36),
                                ("reservoir48", reservoir_ids, 48)]:
            reference[name] = np.column_stack([
                -((queries[:, None] - bank[np.array(index[:size])][None]) ** 2).sum(axis=2).min(axis=1)
                for bank, index in zip(banks, ids)])
        centers = np.vstack([bank.mean(axis=0) for bank in banks])
        reference["centroid"] = -((queries[:, None] - centers[None]) ** 2).sum(axis=2)
        for seed in plan["projection_seeds"]:
            omega = np.random.default_rng(seed).normal(size=(32, 64))
            field = np.vstack([np.exp(1j * (bank @ omega / 1.5)).sum(axis=0) for bank in banks])
            reference[seed] = (np.exp(1j * (queries @ omega / 1.5)) @ field.conj().T).real / 64
        assert len(case["methods"]) == 11
        controls = {}
        hybrid_values = []
        for row in case["methods"]:
            assert row.get("bytes_total", 0) <= plan["total_budget_bytes"]
            method = row["method"]
            if method == "hybrid":
                assert row["bytes_total"] == 98368
                witness_rank = np.argsort(-reference["geometry36"], axis=1, kind="stable")
                phase_rank = np.argsort(-reference[row["projection_seed"]], axis=1, kind="stable")
            else:
                key = row["projection_seed"] if method == "phase64" else method
                order = np.argsort(-reference[key], axis=1, kind="stable")
                assert order.tolist() == row["order"]
            for fanout in plan["fanouts"]:
                if method == "hybrid":
                    selected = []
                    for q in range(32):
                        reserved = witness_rank[q, :(fanout + 1) // 2].tolist()
                        fill = [int(b) for b in phase_rank[q] if b not in reserved]
                        selected.append((reserved + fill)[:fanout])
                    assert selected == row["selected_by_fanout"][str(fanout)]
                else:
                    selected = order[:, :fanout].tolist()
                assert all(len(set(ids)) == fanout for ids in selected)
                retained = sum(int(hits[q, ids].sum()) for q, ids in enumerate(selected)) / 320
                assert row["coverage"][str(fanout)] == retained
            if method in ("geometry48", "reservoir48"):
                controls[method] = row["coverage"]["4"]
            if method == "hybrid":
                hybrid_values.append(row["coverage"]["4"])
        for value in hybrid_values:
            difference = value - max(controls.values())
            differences.append(difference)
            by_scenario.setdefault(case["scenario"], []).append(difference)
            if case["scenario"] != "mixed":
                structured.append(value)
    gains = {name: statistics.mean(values) for name, values in by_scenario.items()}
    decision = {"minimum_structured_coverage_at4": min(structured),
                "worst_delta_vs_better_control_at4": min(differences),
                "mean_delta_vs_better_control_by_scenario": gains,
                "eligible": min(structured) >= .9975 and min(differences) >= -.0025 and max(gains.values()) >= .02}
    assert decision == summary["decision"]
    return {"cases": len(cases), "validated_method_records": sum(len(c["methods"]) for c in cases),
            "coordinatewise_truth_matches": True, "witness_ids_independently_recomputed": True,
            "all_scores_and_routing_coverages_recomputed": True, "combined_memory_budget_matches": True,
            "frozen_decision_matches": True, "decision": decision}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    print(json.dumps(audit(parser.parse_args().directory), indent=2))
