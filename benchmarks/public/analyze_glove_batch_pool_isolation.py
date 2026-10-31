"""Extend the immutable v1 analysis with predeclared default-control gates."""

import argparse
import json
from pathlib import Path

import analyze_glove_batch_pool as previous


CANDIDATE_SHA = "cf05a410e46491f6e141c199c8613dee4c3cd65711ae407a1bcc76e4b6e5b50e"
DEFAULT_GATE = {"ef": 80, "contrast": "disabled_over_baseline", "minimum_geomean_qps_ratio": 0.99, "bootstrap_lower_at_least": 0.98}
SECONDARY_GATE = {"contrast": "disabled_over_baseline", "efs": previous.EFS, "minimum_geomean_qps_ratio": 0.98}


def control_gates(curves):
    primary = curves["80"]["disabled_over_baseline"]["query_only_qps"]
    point_pass = primary["geometric_mean"] >= 0.99
    lower_pass = primary["paired_process_block_bootstrap95"][0] >= 0.98
    secondary = {str(ef): {"geometric_mean": curves[str(ef)]["disabled_over_baseline"]["query_only_qps"]["geometric_mean"], "pass": curves[str(ef)]["disabled_over_baseline"]["query_only_qps"]["geometric_mean"] >= 0.98} for ef in previous.EFS}
    return {
        "primary": {"gate": DEFAULT_GATE, "measured": primary, "point_pass": point_pass, "lower_bound_pass": lower_pass, "pass": point_pass and lower_pass},
        "secondary": {"gate": SECONDARY_GATE, "by_ef": secondary, "pass": all(row["pass"] for row in secondary.values())},
    }


def analyze(root):
    require = previous.common.require
    fingerprint = previous.common.fingerprints
    plan = json.loads((root / "plan.json").read_text())
    manifest = json.loads((root / "execution-manifest.json").read_text())
    require(plan["schema"] == "warm-glove-batch-pool-ablation-plan-v2", "wrong isolation plan schema")
    require(plan["candidate_vertex_sha256"] == CANDIDATE_SHA, "isolation source SHA mismatch")
    require(plan["default_control_gate"] == DEFAULT_GATE and plan["secondary_default_control_gate"] == SECONDARY_GATE, "default-control gate changed")
    require(manifest["frozen_files"]["candidate-source/src/core/qgraph/vertex.rs"] == CANDIDATE_SHA, "archived candidate source differs from declared source")
    result = previous.analyze(root)
    controls = control_gates(result["curves"])
    previous_numeric_pass = result["primary"]["numeric_development_gate_pass"]
    result["primary"].update({
        "enabled_and_semantic_gate_pass": previous_numeric_pass,
        "default_control_gate_pass": controls["primary"]["pass"],
        "secondary_default_control_gate_pass": controls["secondary"]["pass"],
        "numeric_development_gate_pass": previous_numeric_pass and controls["primary"]["pass"] and controls["secondary"]["pass"],
    })
    costs = manifest["compile_cost"]
    require(costs["v1"] == plan["previous_candidate"], "v1 compile-cost declaration changed")
    for version, name in [("v1", "v1-compile.receipt.json"), ("v2", "compile.receipt.json")]:
        receipt = json.loads((root / name).read_text())
        require(receipt["returncode"] == 0 and receipt["error"] is None, "failed compile receipt")
        require(fingerprint(root / name)["sha256"] == costs[version]["compile_receipt_sha256"], "compile receipt SHA mismatch")
        require(receipt["elapsed_s"] == costs[version]["compile_elapsed_s"] and receipt["sampled_summed_process_tree_rss_kib"] == costs[version]["compile_sampled_summed_process_tree_rss_kib"], "compile metric mismatch")
    require(costs["v2"]["vertex_sha256"] == CANDIDATE_SHA and costs["v2"]["binary_sha256"] == manifest["frozen_files"]["candidate-binary"], "v2 cost/source binding mismatch")
    require(costs["v2"]["binary_bytes"] == (root / "candidate-binary").stat().st_size, "v2 binary size mismatch")
    previous_binary = Path(costs["v1"]["root"]) / "candidate-binary"
    require(fingerprint(previous_binary)["sha256"] == costs["v1"]["binary_sha256"] and previous_binary.stat().st_size == costs["v1"]["binary_bytes"], "preserved v1 binary binding mismatch")
    result.update({
        "schema": "warm-glove-batch-pool-isolation-analysis-v2",
        "analyzer_sha256": fingerprint(Path(__file__))["sha256"],
        "reused_v1_analyzer_sha256": fingerprint(Path(previous.__file__))["sha256"],
        "default_control_gates": controls,
        "compile_and_binary_cost": {
            **costs,
            "binary_bytes_v2_minus_v1": costs["v2"]["binary_bytes"] - costs["v1"]["binary_bytes"],
            "binary_bytes_v2_over_v1": costs["v2"]["binary_bytes"] / costs["v1"]["binary_bytes"],
            "interpretation": "Measured release binaries include unchanged harness and debug information (stripnone). Size growth is a code-generation cost, not a speed or memory win. Compile RSS is sampled summed child-tree RSS; compile wall times reuse caches and are not matched clean-build evidence.",
        },
        "isolation_scope": "Addresses measured v1 disabled-arm regression by moving the unchanged8-key mechanism dispatch outside the expansion loop. New defaultcontrol gates were declared beforev2data. Threshold, query workload, index, toolchain, repeats and balanced order are unchanged. Three process blocks remain lowpower, conditionalhardware evidence.",
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    previous.common.require(not args.out.exists(), "analysis output exists")
    result = analyze(args.root.resolve())
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
