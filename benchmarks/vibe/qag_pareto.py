# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Export every frozen point, Pareto frontiers, and eligible threshold comparisons."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path


def frontier(rows):
    return [row for row in rows if not any(
        other["recall_ids_min"] >= row["recall_ids_min"]
        and other["qps_median"] >= row["qps_median"]
        and (other["recall_ids_min"] > row["recall_ids_min"]
             or other["qps_median"] > row["qps_median"])
        for other in rows
    )]


def export(root):
    spec = importlib.util.spec_from_file_location("qag_saved_report", root / "generate_report.py")
    saved = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(saved)
    official = {key: saved.full_report(root, key, True) for key in saved.DATASETS}
    extra = saved.supplemental_mlann(root, official)
    phases = {}
    for phase in ("official", "validation"):
        datasets = {}
        for key, (name, _) in saved.DATASETS.items():
            report, path = official[key] if phase == "official" else saved.full_report(root, key, False)
            if report is None:
                continue
            inputs = [(report, path)]
            if phase == "official" and key in extra:
                inputs.append(extra[key])
            rows = []
            for source, source_path in inputs:
                checksum = hashlib.sha256(source_path.read_bytes()).hexdigest()
                for point in source["rows"]:
                    attrs = point["repetitions"][0]["resource_attrs"]
                    arm = point["arm"]
                    prefix = {0: "baseline4", 1: "candidate8", 2: "roargraph48",
                              3: "roargraph96", 4: "mlann_rf10_depth14"}[arm]
                    if arm == 4:
                        trees, depth = saved.mlann_shape(point)
                        prefix = f"mlann_rf{trees}_depth{depth}"
                    row = {k: v for k, v in point.items() if k != "repetitions"}
                    row.update(method=prefix, configuration=saved.config(point),
                               source_report=str(source_path.relative_to(root)), source_sha256=checksum,
                               qps_repetitions=[p["qps"] for p in point["repetitions"]],
                               degree_capacity=64 if arm == 0 else 128 if arm == 1 else None,
                               traversal_bits=4 if arm == 0 else 8 if arm == 1 else None,
                               residual_bits=8 if arm in (0, 1) else None,
                               exact_original_metric_rerank=False if arm in (0, 1) else None,
                               encoded_index_bytes=attrs.get(f"qag_{prefix}_index_bytes"),
                               build_seconds=attrs.get(f"qag_{prefix}_build_seconds"),
                               isolated_index_peak_rss_kib=None,
                               process_peak_rss_kib=max(p["resource_attrs"].get("qag_process_peak_rss_kib", 0)
                                                        for p in point["repetitions"]) or None)
                    rows.append(row)
            methods = sorted({row["method"] for row in rows})
            comparisons = []
            for target in saved.TARGETS:
                candidate, _ = saved.select({"rows": rows}, {1}, target)
                competitor, _ = saved.select({"rows": rows}, {2, 3, 4}, target)
                comparisons.append({"target": target, "hms": candidate, "competitor": competitor,
                                    "hms_qps_delta_percent": 100 * (candidate["qps_median"] / competitor["qps_median"] - 1)
                                    if candidate and competitor else None})
            datasets[name] = {"source": report["source"], "protocol": report["protocol"],
                              "rows": rows, "frontier_all_measured": frontier(rows),
                              "frontier_by_method": {method: frontier([r for r in rows if r["method"] == method])
                                                     for method in methods}, "thresholds": comparisons}
        phases[phase] = datasets
    return {"schema_version": 1, "scope": "Frozen v1 results only; validation and official kept separate",
            "competitor_scope": "Measured RoarGraph M48/M96 and MLANN RF10/depth14, plus RF80/depth13 when complete; not the entire VIBE roster",
            "hardware": "writerslogic-02; Intel Core Ultra 9 185H; Linux/WSL2; AVX2; no AVX-512",
            "hms_build": {"rustc": "1.96.0", "profile": "release", "rustflags": "-C target-cpu=native",
                          "baseline_sha256": "91beaaf58f5bced0060e6337052edad5aa1dd8c6733fece137b927c9d0788b7e",
                          "candidate_sha256": "31988dc57c8998e12f2daabb8e6b77086775d8532d60f475ac12095d7d2d0a26"},
            "resource_limits": "Main process RSS aggregates all resident arms; encoded bytes are not persisted file size. Build times are single observations.",
            "phases": phases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results/query_aware_graph_v1")
    args = parser.parse_args()
    root = args.evidence_dir.resolve(strict=True)
    report = export(root)
    (root / "pareto.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    fields = ("phase", "dataset", "method", "configuration", "recall_ids_min", "qps_median", "qps_min", "qps_max")
    with (root / "pareto.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for phase, datasets in report["phases"].items():
            for dataset, data in datasets.items():
                for row in data["rows"]:
                    writer.writerow({"phase": phase, "dataset": dataset, **{k: row[k] for k in fields[2:]}})


if __name__ == "__main__":
    main()
