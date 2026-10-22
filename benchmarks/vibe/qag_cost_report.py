# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Summarize frozen validation-only production profiles and paired layout timings."""

import argparse
import hashlib
import json
from pathlib import Path


STAGES = ("query_encoding_ns", "entry_ns", "adjacency_visited_prefetch_ns",
          "scoring_pool_ns", "rerank_sort_ns")
COUNTS = ("expansions", "neighbors_read", "entry_evaluations",
          "traversal_evaluations", "rerank_evaluations", "retained")


def read(path):
    return json.loads(path.read_text())


def fingerprint(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def profile_summary(profile):
    if not profile["returned_ids_identical"] or not profile["persisted_bytes_identical"]:
        raise ValueError("layout identity requirements failed")
    layouts = profile["layouts"]
    if [row["compact"] for row in layouts] != [False, True]:
        raise ValueError("expected fixed then compact production profiles")
    a, b = layouts
    if a["recall"] != b["recall"] or len(a["queries"]) != 256 or len(b["queries"]) != 256:
        raise ValueError("profile recall or reserved validation count differs")
    if a["persisted"]["bytes"] != b["persisted"]["bytes"]:
        raise ValueError("persisted sizes differ")
    for x, y in zip(a["queries"], b["queries"]):
        if any(x[key] != y[key] for key in COUNTS):
            raise ValueError("compaction changed production search work")
    output = []
    for layout in layouts:
        queries = layout["queries"]
        total = sum(q["total_ns"] for q in queries)
        stages = {name: sum(q[name] for q in queries) for name in STAGES}
        operations = {key: sum(q[key] for q in queries) / len(queries) for key in COUNTS}
        if total <= 0 or sum(stages.values()) > total:
            raise ValueError("invalid profile stage accounting")
        output.append({
            "layout": "compact8" if layout["compact"] else "fixed8",
            "recall": layout["recall"], "queries": len(queries),
            "encoded_index_bytes": layout["index_bytes"],
            "compaction_seconds": layout["compaction_seconds"],
            "persisted": layout["persisted"],
            "mean_instrumented_microseconds": total / len(queries) / 1000,
            "stage_percent_of_instrumented_total": {key: 100 * value / total for key, value in stages.items()},
            "unattributed_percent": 100 * (total - sum(stages.values())) / total,
            "mean_operations": operations,
            "edges_per_expansion": operations["neighbors_read"] / operations["expansions"],
        })
    return {"settings": profile["settings"], "graph_hash": profile["graph_hash"],
            "identical_ids_bytes_work": True,
            "encoded_saving_percent": 100 * (1 - b["index_bytes"] / a["index_bytes"]),
            "layouts": output}


def timing_summary(report, gate):
    if not gate["identical_returned_ids"] or report["source"]["official_test"]:
        raise ValueError("layout timing requires validation-only identical results")
    rows = {(r["arm"], r["ef"], r["rerank"]): r for r in report["rows"]}
    points = []
    for point in gate["points"]:
        ef, rr = point["ef"], point["rerank"]
        a, b = rows[(0, ef, rr)], rows[(1, ef, rr)]
        runs = [{r["repetition"]: r for r in row["repetitions"]} for row in (a, b)]
        if any(set(run) != {0, 1, 2} for run in runs):
            raise ValueError("layout timing requires three complete paired blocks")
        if any(runs[0][i]["ids_sha256"] != runs[1][i]["ids_sha256"] for i in range(3)):
            raise ValueError("layout timing IDs differ")
        if a["recall_ids_min"] != b["recall_ids_min"]:
            raise ValueError("matched layout recall differs")
        deltas = [100 * (runs[1][i]["qps"] / runs[0][i]["qps"] - 1) for i in range(3)]
        status = ("faster in all observed blocks with separated ranges"
                  if min(deltas) > 0 and b["qps_min"] > a["qps_max"]
                  else "slower in all observed blocks" if max(deltas) < 0 else "inconclusive")
        points.append({"ef": ef, "rerank": rr, "recall": a["recall_ids_min"],
                       "fixed_qps": [a["qps_median"], a["qps_min"], a["qps_max"]],
                       "compact_qps": [b["qps_median"], b["qps_min"], b["qps_max"]],
                       "median_delta_percent": 100 * (b["qps_median"] / a["qps_median"] - 1),
                       "paired_delta_percent": deltas, "status": status})
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/query_aware_graph_v2")
    root = parser.parse_args().evidence_dir.resolve(strict=True)
    summary = {"scope": "Reserved learn validation only; no new official evaluation",
               "profile_limits": "Unpinned instrumented aggregate stages; no production QPS, cache, allocation, bandwidth or instruction inference",
               "timing_limits": "One harness-selected CPU, five passes, saved fifth pass, three alternating paired blocks; observed ranges are not confidence intervals",
               "datasets": {}}
    lines = ["# Production search cost and adjacency layout", "",
             summary["scope"] + ". The frozen v1 official results remain separate.", "",
             summary["profile_limits"] + ".", "",
             "| Dataset / layout | Actual recall | Encoded bytes | Persisted bytes | Encode query % | Entry % | Adjacency / visited / prefetch % | Scoring / pool % | Rerank / sort % | Other % |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    timings, resources = [], []
    for dataset in ("llama", "yi"):
        path = root / "profiles" / (dataset + "-high.json")
        profile = profile_summary(read(path))
        profile["source"] = fingerprint(path)
        for layout in profile["layouts"]:
            percentages = [layout["stage_percent_of_instrumented_total"][key] for key in STAGES]
            cells = " | ".join(f"{value:.2f}" for value in [*percentages, layout["unattributed_percent"]])
            lines.append(f"| {dataset} / {layout['layout']} | {layout['recall']:.8f} | {layout['encoded_index_bytes']:,} | {layout['persisted']['bytes']:,} | {cells} |")
        directory = root / "runs" / (dataset + "-layout-v2")
        if (directory / "completion.json").is_file():
            report = read(directory / "report.json")
            profile["timings"] = timing_summary(report, read(directory / "gate.json"))
            profile["timing_sources"] = [fingerprint(directory / name) for name in ("report.json", "gate.json", "benchmark.json", "raw.hdf5")]
            attrs = [r["resource_attrs"] for row in report["rows"] for r in row["repetitions"]]
            builds = []
            for arm, prefix in enumerate(("qag_baseline4", "qag_candidate8")):
                sizes = {r[prefix + "_index_bytes"] for r in attrs}
                seconds = {r[prefix + "_build_seconds"] for r in attrs}
                if sizes != {profile["layouts"][arm]["encoded_index_bytes"]} or len(seconds) != 1:
                    raise ValueError("paired production resources differ from frozen diagnostic layout")
                builds.append({"layout": profile["layouts"][arm]["layout"], "seconds": seconds.pop()})
            peak = max(r["qag_process_peak_rss_kib"] for r in attrs)
            profile["paired_resources"] = {"builds": builds, "co_resident_process_peak_rss_kib": peak}
            resources.append(f"| {dataset} | {builds[0]['seconds']:.3f} | {builds[1]['seconds']:.3f} | {peak:,} |")
            for point in profile["timings"]:
                def qps(values):
                    return f"{values[0]:.2f} [{values[1]:.2f}–{values[2]:.2f}]"
                deltas = ", ".join(f"{v:+.2f}%" for v in point["paired_delta_percent"])
                timings.append(f"| {dataset} / {point['ef']}/{point['rerank']} | {point['recall']:.8f} | {qps(point['fixed_qps'])} | {qps(point['compact_qps'])} | {point['median_delta_percent']:+.2f}% | {deltas} | {point['status']} |")
        summary["datasets"][dataset] = profile
    lines.extend(["", "Exact ordered returned IDs, persisted bytes and operation counts match between layouts on all profiled queries.", "",
                  "| Dataset / ef/rerank | Actual recall | Fixed QPS median [min–max] | Compact QPS median [min–max] | Median delta | Paired deltas | Observed timing result |",
                  "|---|---:|---:|---:|---:|---|---|", *timings, "", summary["timing_limits"] + ".", "",
                  "| Dataset | Fixed build seconds | Compact build seconds | Co-resident process peak RSS, KiB |",
                  "|---|---:|---:|---:|", *resources, "",
                  "Compaction saves live encoded adjacency capacity. Save expands the original padding, so persisted-file size does not shrink. Save/open/compaction times in [summary.json](summary.json) are single diagnostic observations. Isolated index RSS/peak build RSS remain unmeasured.", "",
                  "Sources, settings, stage totals and work counts: [summary.json](summary.json). Drivers: [profile-v2.sh](profile-v2.sh), [layout-v2.sh](layout-v2.sh). Decisions: [ledger.md](ledger.md).", ""])
    points = [p for dataset in summary["datasets"].values() for p in dataset.get("timings", [])]
    if len(points) == 6:
        summary["repeatable_speedup_observed"] = any(p["status"].startswith("faster") for p in points)
        if not summary["repeatable_speedup_observed"]:
            lines[2:2] = ["No tested point shows a repeatable speedup with separated observed ranges. Compact adjacency is retained as an explicit memory option, default off.", ""]
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (root / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
