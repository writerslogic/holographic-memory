# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * fraction) - 1]


def summarize():
    ledger = json.loads((HERE / "execution.json").read_text())
    assert ledger["complete"] and all(run["status"] == 0 for run in ledger["runs"])
    reports = {
        variant: [json.loads((HERE / f"{variant}-{n}.json").read_text()) for n in range(1, 6)]
        for variant in ["baseline", "candidate"]
    }
    checks = sum(group["oracle_checks"] for runs in reports.values() for run in runs for group in run["queries"])
    for phase in ["initial", "updated"]:
        fingerprints = {group["ordered_results_crc32"] for runs in reports.values() for run in runs for group in run["queries"] if group["phase"] == phase}
        assert len(fingerprints) == 1, f"Result checksum changed in {phase}"
    summary = {
        "baseline": json.loads((HERE / "build-baseline.json").read_text())["revision"],
        "execution_seconds": ledger["execution_seconds"],
        "exhaustive_ordered_result_equalities": checks,
        "independent_rounds_per_variant": 5,
        "samples_per_group_per_round": 128,
        "aggregation": "Median of five per-round nearest-rank quantiles; paired differences retained. Empirical p99 is not a population-tail guarantee.",
        "groups": {},
        "limitations": [
            "Background indexing and desktop activity; host-observation.json and per-run load averages record the lack of host isolation.",
            "One synthetic 10,000-row, 16-dimensional corpus, 32 mutations, fixed candidate limit and one runtime worker.",
            "Allocation counts and requested bytes include identical counting-allocator instrumentation; not allocator usable size or RSS.",
            "Comparison is against f28095e0, not the historical a736af02 baseline; it cannot establish closure of the earlier 11.5% hot-query regression.",
            "Exclusion bookkeeping grows by an optional owner identity per bounded delta entry; no reduction in peak requested query memory was measured.",
        ],
    }
    for phase in ["initial", "updated"]:
        for hot in [False, True]:
            group = {}
            for metric in ["capture_ns", "execute_ns", "total_ns", "allocations", "peak_requested_bytes"]:
                for quantile in ([0.5, 0.95, 0.99] if metric.endswith("_ns") else [0.5]):
                    rounds = {}
                    for variant, runs in reports.items():
                        rounds[variant] = [
                            percentile([sample[metric] for sample in next(g for g in run["queries"] if g["phase"] == phase)["samples"] if sample["hot"] == hot], quantile)
                            for run in runs
                        ]
                    baseline = statistics.median(rounds["baseline"])
                    candidate = statistics.median(rounds["candidate"])
                    group[f"{metric}_p{round(quantile * 100)}"] = {
                        "baseline": baseline, "candidate": candidate,
                        "change_percent": 100 * (candidate / baseline - 1),
                        "rounds": rounds,
                        "paired_change_percent": [100 * (c / b - 1) for b, c in zip(rounds["baseline"], rounds["candidate"])],
                    }
            summary["groups"][f"{phase}-{'hot' if hot else 'rare'}"] = group
    summary["lifecycle"] = {
        variant: {
            "initial_train_ns_median": statistics.median(run["initial_train_ns"] for run in runs),
            "maintenance_rebuilds": [run["maintenance_rebuilds"] for run in runs],
            "update_and_maintenance_ns_median": statistics.median(sum(m["update_ns"] + m["maintenance_ns"] for m in run["mutations"]) for run in runs),
            "reopen_ns_median": statistics.median(run["engine_lifecycle"]["reopen_ns"] for run in runs),
        }
        for variant, runs in reports.items()
    }
    return summary


if __name__ == "__main__":
    result = summarize()
    (HERE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checks": result["exhaustive_ordered_result_equalities"], "seconds": result["execution_seconds"], "lifecycle": result["lifecycle"]}, indent=2))
