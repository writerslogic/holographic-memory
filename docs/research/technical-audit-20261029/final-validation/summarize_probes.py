# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0

"""Summarize retained local observations without rerunning benchmarks."""

import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def percentile(values, fraction):
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def summarize(directory):
    execution = json.loads((directory / "probe-execution.json").read_text())
    assert execution["completed"]
    variants = {}
    for variant in ["baseline", "eligibility", "delta", "combined"]:
        rounds = []
        for number in [1, 2, 3]:
            data = json.loads((directory / f"probe-{variant}-{number}.json").read_text())
            report = {
                "round": number,
                "ingest_ms": data["ingest_ns"] / 1e6,
                "initial_train_ms": data["initial_train_ns"] / 1e6,
                "maintenance_rebuilds": data["maintenance_rebuilds"],
                "mutation_and_maintenance_ms": sum(
                    m["update_ns"] + m["maintenance_ns"] for m in data["mutations"]
                ) / 1e6,
                "engine_lifecycle": data["engine_lifecycle"],
                "query_groups": {},
            }
            usage = next(r for r in execution["rounds"] if r["log"] == f"probe-{variant}-{number}.log")
            report["peak_rss_bytes_macos"] = usage["peak_rss_platform_units"]
            report["cpu_seconds"] = usage["user_seconds"] + usage["system_seconds"]
            report["oracle_checks"] = sum(q["oracle_checks"] for q in data["queries"])
            report["ordered_results_crc32"] = [q["ordered_results_crc32"] for q in data["queries"]]
            for phase in data["queries"]:
                for hot in [False, True]:
                    samples = [s for s in phase["samples"] if s["hot"] == hot]
                    group = {
                        metric: {
                            f"p{int(p * 100)}": percentile([s[metric] for s in samples], p)
                            for p in [.5, .95, .99]
                        }
                        for metric in ["prepare_ns", "capture_ns", "execute_ns", "total_ns"]
                    }
                    for metric in ["allocations", "peak_requested_bytes", "captured_chunks"]:
                        group[metric] = {"min": min(s[metric] for s in samples), "max": max(s[metric] for s in samples)}
                    group["samples"] = len(samples)
                    group["sum_total_ns"] = sum(s["total_ns"] for s in samples)
                    report["query_groups"][phase["phase"] + ("_hot" if hot else "_rare")] = group
            rounds.append(report)
        variants[variant] = rounds
    comparisons = {}
    for group in variants["baseline"][0]["query_groups"]:
        comparisons[group] = {}
        for metric, p in [("capture_ns", "p95"), ("total_ns", "p50"), ("total_ns", "p99")]:
            baseline = statistics.median(r["query_groups"][group][metric][p] for r in variants["baseline"])
            combined = statistics.median(r["query_groups"][group][metric][p] for r in variants["combined"])
            comparisons[group][f"{metric}_{p}"] = {
                "baseline": baseline, "combined": combined,
                "change_percent": 100 * (combined / baseline - 1),
            }
    return {"execution_seconds_cumulative": execution["execution_seconds"], "variants": variants, "median_round_comparisons": comparisons}


def temporal():
    variants = {}
    for variant in ["baseline", "combined"]:
        rounds = []
        for number in [1, 2, 3]:
            records = []
            for line in (HERE / f"temporal-{variant}-{number}.log").read_text().splitlines():
                if '{"checks":' in line:
                    data = json.loads(line[line.index('{"checks":'):])
                    data["failed_steps"] = len({error.split(": /", 1)[0] for error in data["failures"]})
                    data["failed_assertions"] = len(data.pop("failures"))
                    records.append(data)
            assert len(records) == 6
            rounds.append(records)
        variants[variant] = rounds
    return variants


if __name__ == "__main__":
    result = {
        "method": "Nearest-rank percentiles of 32 queries per group/round; comparison is median of three round percentiles. p99 is the sample maximum, not an estimated population tail. No inferential confidence interval from three rounds.",
        "limitations": "One macOS arm64 host, 10k synthetic 16D rows, fixed budgets, allocator instrumentation and other host workloads. No extrapolation to model QA, high-dimensional capacity or all selectivities. Actual ordered JSON equality is asserted for every query; CRC is only an artifact cross-check.",
        "initial": summarize(HERE),
        "repaired": summarize(HERE / "repaired-retrieval"),
        "temporal": temporal(),
        "fixture_sha256": {name: hashlib.sha256((ROOT / "tests/fixtures" / name).read_bytes()).hexdigest()
                           for name in ["native_memory_evidence_v1.json", "native_memory_lifecycle_v1.json", "scitt_receipt_python_v1.json"]},
    }
    (HERE / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")
