"""Analyze the frozen three-block batch-pool development ablation."""

import argparse
import itertools
import json
import math
from pathlib import Path
import statistics

import analyze_glove_development as common


ORDERS = [["baseline", "disabled", "enabled"], ["enabled", "baseline", "disabled"], ["disabled", "enabled", "baseline"]]
EFS = [10, 20, 40, 80, 120, 160, 240, 400, 640]
MATCH = ["construction_count", "query_count", "dim", "k", "rerank", "efs", "repeats", "order_seed", "warmup_passes_per_ef", "truth", "truth_method", "residency", "timing_scope", "input_declaration", "binding_crc32", "knob_mode", "knobs", "logical_index_bytes", "serialized_index_bytes"]


def ratio_summary(ratios):
    common.require(len(ratios) == 3 and all(math.isfinite(r) and r > 0 for r in ratios), "three positive process ratios required")
    logs = [math.log(r) for r in ratios]
    center = statistics.fmean(logs)
    samples = [math.exp(statistics.fmean(xs)) for xs in itertools.product(logs, repeat=3)]
    delta = 4.302652729911275 * statistics.stdev(logs) / math.sqrt(3)
    return {
        "by_process_block": ratios,
        "geometric_mean": math.exp(center),
        "paired_process_block_bootstrap95": [common.quantile(samples, p) for p in [0.025, 0.975]],
        "paired_log_ratio_student_t95_sensitivity": [math.exp(center - delta), math.exp(center + delta)],
    }


def collapse(raw, cells, ef):
    trials = [cells[(ef, r)] for r in range(7)]
    latencies = [n for trial in trials for n in trial["latencies_ns"]]
    return {
        "query_only_qps": len(latencies) * 1e9 / sum(latencies),
        "batch_qps": len(latencies) * 1e9 / sum(t["batch_ns"] for t in trials),
        "recall_at_10": statistics.fmean(t["recall_at_10"] for t in trials),
        "latency_ns": {f"p{p:g}": common.quantile(latencies, p / 100) for p in [50, 95, 99, 99.9]},
        "distance_evaluations_mean": statistics.fmean(v for t in trials for v in t["distance_evaluations"]),
    }


def analyze(root):
    plan = json.loads((root / "plan.json").read_text())
    bindings = json.loads((root / "execution-manifest.json").read_text())
    common.require(plan["block_orders"] == ORDERS and plan["efs"] == EFS and plan["repeats_per_process"] == 7, "unexpected frozen schedule")
    common.require(plan["primary_gate"] == {"ef": 80, "recall_floor": 0.95, "minimum_geomean_qps_ratio": 1.10, "bootstrap_lower_strictly_above": 1.0, "contrasts": ["enabled_over_baseline", "enabled_over_disabled"]}, "unexpected acceptance gate")
    common.require(bindings["plan_sha256"] == common.fingerprints(root / "plan.json")["sha256"], "plan SHA mismatch")
    for relative, expected in bindings["frozen_files"].items():
        path = Path(relative)
        common.require(not path.is_absolute() and ".." not in path.parts, "unsafe archived artifact path")
        common.require(common.fingerprints(root / path)["sha256"] == expected, "frozen artifact SHA mismatch")
    for item in bindings["profile_evidence"].values():
        path = Path(item["path"])
        common.require(not path.is_absolute() and ".." not in path.parts, "unsafe profile artifact path")
        common.require(common.fingerprints(root / path)["sha256"] == item["sha256"], "profile artifact SHA mismatch")
    results, hashes, memory = {}, {}, {}
    reference = None
    reference_cells = None
    semantic_checks = {}
    previous_end = 0.0
    for block, order in enumerate(ORDERS):
        for position, arm in enumerate(order):
            name = f"block{block}-{arm}"
            expected = bindings["timed_runs"][name]
            common.require(expected["block"] == block and expected["position"] == position and expected["arm"] == arm, "run schedule mismatch")
            common.require(previous_end <= expected["started_monotonic_s"] < expected["finished_monotonic_s"], "timed processes overlap or violate schedule")
            previous_end = expected["finished_monotonic_s"]
            path = root / f"{name}.json"
            common.require(common.fingerprints(path)["sha256"] == expected["result_sha256"], "raw result SHA mismatch")
            receipt = json.loads((root / f"{name}.receipt.json").read_text())
            common.require(receipt["returncode"] == 0 and receipt["error"] is None, "failed timed process")
            common.require(receipt["command"] == expected["command"], "command mismatch")
            binary = plan["arms"][arm]["binary"]
            common.require(expected["binary_sha256"] == bindings["frozen_files"][binary] and receipt["command"][0] == str(root / binary), "binary binding mismatch")
            common.require(expected["environment"]["HMS_QGRAPH_BATCH_POOL"] == ("1" if arm == "enabled" else "0"), "mechanism switch mismatch")
            raw, cells, details = common.load(path)
            common.require(raw["efs"] == EFS and raw["repeats"] == 7 and raw["knob_mode"] == "env", "raw configuration mismatch")
            common.require(raw["knobs"] == {"prefetch": 0, "batch_est": False, "frontier": 1.0, "block_screen": False, "rerank_z": 0.0}, "unrelated search knob changed")
            if reference is None:
                reference, reference_cells = raw, cells
            for key in MATCH:
                common.require(raw[key] == reference[key], f"unmatched {key}")
            ids_equal = all(trial["ids"] == reference_cells[key]["ids"] for key, trial in cells.items())
            work_equal = all(trial["distance_evaluations"] == reference_cells[key]["distance_evaluations"] for key, trial in cells.items())
            common.require(all(trial["query_order"] == reference_cells[key]["query_order"] for key, trial in cells.items()), "unmatched query order")
            semantic_checks[name] = {"ordered_ids_equal_reference": ids_equal, "distance_evaluations_equal_reference": work_equal, "ordered_ids_stable_all_repeats": all(cells[(ef, r)]["ids"] == cells[(ef, 0)]["ids"] for ef in EFS for r in range(7))}
            results[name] = {str(ef): collapse(raw, cells, ef) for ef in EFS}
            hashes[name] = {"raw_sha256": expected["result_sha256"], "receipt_sha256": common.fingerprints(root / f"{name}.receipt.json")["sha256"], "input_and_index": details["bindings"], "binary_sha256": expected["binary_sha256"]}
            memory[name] = {"logical_index_bytes": raw["logical_index_bytes"], "serialized_index_bytes": raw["serialized_index_bytes"], "sampled_summed_process_tree_rss_kib": receipt["sampled_summed_process_tree_rss_kib"]}
    curves = {}
    for ef in EFS:
        curve = {}
        for left, right in [("baseline", "disabled"), ("baseline", "enabled"), ("disabled", "enabled")]:
            pairs = [(results[f"block{b}-{left}"][str(ef)], results[f"block{b}-{right}"][str(ef)]) for b in range(3)]
            curve[f"{right}_over_{left}"] = {
                "query_only_qps": ratio_summary([y["query_only_qps"] / x["query_only_qps"] for x, y in pairs]),
                "batch_qps": ratio_summary([y["batch_qps"] / x["batch_qps"] for x, y in pairs]),
                "recall_difference_by_block": [y["recall_at_10"] - x["recall_at_10"] for x, y in pairs],
                "p99_latency_right_over_left_by_block": [y["latency_ns"]["p99"] / x["latency_ns"]["p99"] for x, y in pairs],
            }
        curves[str(ef)] = curve
    semantic_pass = all(all(check.values()) for check in semantic_checks.values())
    recall_pass = all(result["80"]["recall_at_10"] >= 0.95 for result in results.values())
    primary = {contrast: curves["80"][contrast]["query_only_qps"] for contrast in plan["primary_gate"]["contrasts"]}
    speed_pass = all(row["geometric_mean"] >= 1.10 and row["paired_process_block_bootstrap95"][0] > 1.0 for row in primary.values())
    return {
        "schema": "hms-glove-batch-pool-ablation-analysis-v1", "development_only": True,
        "plan_sha256": bindings["plan_sha256"], "execution_manifest_sha256": common.fingerprints(root / "execution-manifest.json")["sha256"],
        "analyzer_sha256": common.fingerprints(Path(__file__))["sha256"], "shared_analyzer_sha256": common.fingerprints(Path(common.__file__))["sha256"],
        "process_results": results, "curves": curves, "raw_bindings": hashes, "memory_metrics": memory,
        "semantic_checks": semantic_checks,
        "primary": {"ef": 80, "recall_floor": 0.95, "contrasts": primary, "semantic_pass": semantic_pass, "recall_pass": recall_pass, "speed_pass": speed_pass, "numeric_development_gate_pass": semantic_pass and recall_pass and speed_pass, "profile_contrast_gate": "Requires parent review of baseline/candidate query-only sampled call sites; numeric pass alone is insufficient.", "leaderboard_claim": False},
        "uncertainty_scope": "Seven within-process repeats collapse to total queries divided by total native-query latency for each fresh process. Three matched process blocks are the only resampling units. Paired log-ratio bootstrap enumerates all27 ordered resamples of3blocks;95% percentile endpoints use linear quantiles. Student-t df2 sensitivity is also reported. Three blocks have low power; neither interval establishes coverage across builds/hardware/days or final comparative uncertainty. No perquery pseudoreplication; tail quantiles are descriptive.",
        "memory_scope": "Logical index bytes and serialized file bytes are static. Sampled summed child-tree RSS includes exact-truth preparation and allocator retention; it is neither OS peak RSS nor unique/live index memory. All arms use the same index; no memory-improvement claim.",
        "profile_evidence": bindings["profile_evidence"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    common.require(not args.out.exists(), "analysis output exists")
    result = analyze(args.root.resolve())
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
