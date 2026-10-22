# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Export the frozen exact byte-VNNI validation comparison without official claims."""

import argparse
import json
from pathlib import Path

from qag_cost_report import fingerprint, read, timing_summary


def qps(values):
    return f"{values[0]:.2f} [{values[1]:.2f}–{values[2]:.2f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/query_aware_graph_v3")
    root = parser.parse_args().evidence_dir.resolve(strict=True)
    summary = {"scope": "Reserved learn validation only; frozen v1 official results are unchanged",
               "control": "Fixed adjacency, same 8-bit codes and 8-bit residuals, graph/search settings and construction rows; v2 versus v3 source",
               "protocol": "writerslogic-02, Rust1.96.0/native, one VIBE-selected CPU, one thread, three alternating paired blocks, saved fifth pass after four warm passes",
               "datasets": {}}
    lines = ["# Exact byte-VNNI validation", "", summary["scope"] + ".", "",
             summary["control"] + ". " + summary["protocol"] + ".", "",
             "| Dataset / ef/rerank | Actual recall | Reference QPS median [min–max] | Byte-VNNI QPS median [min–max] | Median delta | Paired deltas | Observed result |",
             "|---|---:|---:|---:|---:|---|---|"]
    resource_lines = []
    for dataset in ("llama", "yi"):
        directory = root / "runs" / (dataset + "-kernel-v3")
        completion = read(directory / "completion.json")
        if fingerprint(directory / "report.json")["sha256"] != completion["report"]["sha256"]:
            raise ValueError("kernel report completion hash differs")
        report = read(directory / "report.json")
        points = timing_summary(report, read(directory / "gate.json"))
        for point in points:
            point["reference_qps"] = point.pop("fixed_qps")
            point["byte_vnni_qps"] = point.pop("compact_qps")
            deltas = ", ".join(f"{v:+.2f}%" for v in point["paired_delta_percent"])
            lines.append(f"| {dataset} / {point['ef']}/{point['rerank']} | {point['recall']:.8f} | {qps(point['reference_qps'])} | {qps(point['byte_vnni_qps'])} | {point['median_delta_percent']:+.2f}% | {deltas} | {point['status']} |")
        attrs = [run["resource_attrs"] for row in report["rows"] for run in row["repetitions"]]
        resources = {}
        for name, prefix in (("reference", "qag_baseline4"), ("byte_vnni", "qag_candidate8")):
            sizes = {a[prefix + "_index_bytes"] for a in attrs}
            seconds = {a[prefix + "_build_seconds"] for a in attrs}
            if len(sizes) != 1 or len(seconds) != 1:
                raise ValueError("kernel resource attributes differ across repetitions")
            resources[name] = {"encoded_index_bytes": sizes.pop(), "build_seconds": seconds.pop()}
        if resources["reference"]["encoded_index_bytes"] != resources["byte_vnni"]["encoded_index_bytes"]:
            raise ValueError("integer kernel changed encoded size")
        resources["co_resident_process_peak_rss_kib"] = max(a["qag_process_peak_rss_kib"] for a in attrs)
        resource_lines.append(f"| {dataset} | {resources['reference']['build_seconds']:.3f} | {resources['byte_vnni']['build_seconds']:.3f} | {resources['reference']['encoded_index_bytes']:,} | {resources['co_resident_process_peak_rss_kib']:,} |")
        summary["datasets"][dataset] = {"points": points, "resources": resources,
                                        "sources": [fingerprint(directory / name) for name in ("report.json", "gate.json", "benchmark.json", "raw.hdf5")]}
    decision_path = root / "decision.json"
    if decision_path.is_file():
        summary["decision"] = read(decision_path)
        verdict = ("The kernel passes the predeclared six-point validation criterion and is selected for final Rust validation."
                   if summary["decision"]["kernel_accepted"] else
                   "The kernel is rejected by the predeclared six-point criterion; final Rust validation selects the unchanged v2 kernel. The frozen patch and all results are retained.")
        lines[2:2] = [verdict, ""]
    lines.extend(["", "Ordered returned IDs match in every paired group. Observed ranges are not statistical confidence intervals. No combined compact-layout/VNNI timing or new official-test timing was performed.", "",
                  "| Dataset | Reference build seconds | Byte-VNNI build seconds | Encoded bytes per index | Co-resident process peak RSS, KiB |",
                  "|---|---:|---:|---:|---:|", *resource_lines, "",
                  "Build times are single observations. Process RSS includes both indexes and input/runtime buffers; isolated index peak RSS is unmeasured.", "",
                  "Reproduce from signed2d4967e plus [kernel.patch](kernel.patch), [kernel-v3.sh](kernel-v3.sh), and the source/binary/compiler receipts. Selection and the sole final Cargo gate use [final-gate.sh](final-gate.sh). Decisions and limits: [ledger.md](ledger.md). Machine-readable points: [summary.json](summary.json).", ""])
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (root / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
