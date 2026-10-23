# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


lines = [
    "# Direct IP serving and independent candidate retention",
    "",
    ("Reserved learn validation only: 256 queries per full dataset, 32768 disjoint construction queries. "
    "No untouched final set is certified; these results do not establish a new official result."),
    "",
    ("The measured source is reconstructed by applying [measured-source.patch](measured-source.patch) "
    "to commit 4d5168f7ce8253b15aefe2f7ce24f0e248de4deb; final-source.sha256 and "
    "final-callers.sha256 identify its files. The PR integrates later main changes separately; "
    "its functional checks are recorded in [pr-integration.json](pr-integration.json). "
    "Timings were not rerun after integration. Large diagnostic JSONs and raw HDF5 files "
    "use Git LFS; fetch them with `git lfs pull --include='benchmarks/results/query_aware_graph_v4/**'` "
    "before regenerating this report."),
    "",
    ("Both changes are opt-in. Construction remains the lifted Euclidean graph; direct serving encodes "
    "the original 128 coordinates with explicit negative-inner-product semantics. The independent "
    "reservoir receives every row score before beam admission, including deduplicated descent scores. "
    "The existing ef >= k rule remains in force; screens and neighbor blocks reject independent retention."),
    "",
    "```python",
    'index = holographic_memory.VectorIndex("ip", code_bits=8, direct_ip=True)',
    "index.fit(base, training_queries=construction_queries)",
    "index.set_query_arguments(ef=100, rerank=400, reservoir=400)",
    "```",
    "",
    ("Omitting direct_ip and reservoir preserves legacy serving and retention. Rust callers use "
    "VGraph::from_lifted_ip_graph and VSearchParams::reservoir. VGraph::metric now returns VMetric; "
    "construction Metric converts with Into. Serving metric word 2 identifies IP in HMSVGR01, "
    "which older readers reject. Lifted HMSGRF04 construction caches are unchanged."),
    "",
    "## Storage and quantization",
    "",
    "| Dataset | Lifted encoded bytes | Direct encoded bytes | Saving | Lift dominates quantization scale | Mean lifted/direct scale ratio |",
    "|---|---:|---:|---:|---:|---:|",
]
summary = {"schema_version": 1, "scope": "reserved validation only", "datasets": {}}
datasets = {}
direct_gains = []
for name in ("llama", "yi"):
    diagnostic_path = ROOT / "diagnostics" / f"{name}.json"
    report_path = ROOT / "runs" / name / "report.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    report = json.loads(report_path.read_text())
    assert diagnostic["retention_preserves_traversal"]
    assert not report["source"]["official_test"] and report["source"]["queries"] == 256
    datasets[name] = (diagnostic, report)
    enc = {r["direct_ip"]: r["diagnostic"] for r in diagnostic["encodings"]}
    before, after = enc[False]["index_bytes"], enc[True]["index_bytes"]
    assert before - after == 148 * diagnostic["input"]["rows"] + 64
    scale = diagnostic["lift_scale_diagnostic"]
    lines.append(f"| {name} | {before:,} | {after:,} | {100*(1-after/before):.4f}% | "
        f"{100*scale['tail_dominates_fraction']:.4f}% | {scale['mean_scale_ratio_nonzero_original']:.6f} |")
    summary["datasets"][name] = {"diagnostic_sha256": digest(diagnostic_path),
        "report_sha256": digest(report_path), "encoded_bytes": {"lifted": before, "direct": after},
        "diagnostic_aggregate": diagnostic["aggregate"], "qualifying_points": report["targets"]}
    indexed = {(r["arm"], r["ef"]): r for r in report["rows"]}
    paired = []
    for ef in (100, 256, 512):
        a, b = indexed[0, ef], indexed[2, ef]
        gain = 100 * (b["qps_median"] / a["qps_median"] - 1)
        direct_gains.append(gain)
        ratios = [y["qps"] / x["qps"] for x, y in zip(a["repetitions"], b["repetitions"])]
        paired.append({"ef": ef, "median_gain_percent": gain, "paired_gain_percent": [100*(r-1) for r in ratios],
            "separated_observed_ranges": b["qps_min"] > a["qps_max"]})
    summary["datasets"][name]["direct_vs_lifted_same_effort"] = paired
lines[4:4] = [(f"Direct IP improves median QPS by {min(direct_gains):.2f}–{max(direct_gains):.2f}% "
    "across the six tested dataset/beam points and saves about 18.4% of encoded bytes. "
    "The reservoir recovers low-beam recall at a measurable cost; at high effort, direct serving "
    "without the reservoir is the faster tested variant. Both features remain independently selectable."), ""]
lines.extend(["", ("The measured 8-bit primary stride is 256→128 bytes; residual stride 144→128; "
    "norm storage 4→0 bytes per vector. Total savings are 148 bytes per vector plus 64 mean-array "
    "bytes. Encoded bytes exclude query scratch and runtime overhead and are not isolated RSS. "
    "Coordinate reconstruction errors are recorded in the diagnostic JSON; the lifted and direct "
    "coordinate RMSEs use different widths and are not a query-score error comparison."), "",
    "## Loss breakdown", "",
    ("Coverage and recall are percentages. These rows use lifted serving; direct rows and "
    "all original-float oracle reranks are retained in the JSON. Oracle work is diagnostic only."), "",
    "| Dataset | ef | Reservoir | Evaluated coverage | Beam coverage | Retained coverage | Rerank-set coverage | Returned recall | Float oracle over evaluated IDs |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
for name, (diagnostic, _) in datasets.items():
    for row in diagnostic["aggregate"]:
        if row["direct_ip"]:
            continue
        fields = ("evaluated_coverage", "beam_coverage", "retained_coverage", "rerank_candidate_coverage", "recall", "oracle_evaluated_recall")
        values = " | ".join(f"{100*row[f]:.4f}%" for f in fields)
        lines.append(f"| {name} | {row['ef']} | {row['reservoir']} | {values} |")
lines.extend(["", ("Changing only retention preserves every evaluated-ID set, ordered traversal beam "
    "and expansion count in all full-data diagnostic queries. At ef100, the 400-candidate reservoir "
    "recovers encountered true neighbors that the beam discards. At ef256/512, most misses are "
    "unencountered neighbors; original-float reranking adds at most 0.02 percentage points. "
    "Yi's low-beam discovery ceiling remains about 91.8%. Direct serving makes small, mixed "
    "recall changes; a precision improvement is not established."), "", "## Paired query timings", "",
    ("writerslogic-02, Core Ultra 9 185H, one VIBE-selected CPU (0) and one worker, "
    "8-bit traversal and 8-bit residual codes; each representation "
    "is built once and reused across all settings. Four arms, three alternating blocks, five passes "
    "per block; QPS uses only the saved fifth pass. All arms share two resident indexes. "
    "Ranges are observed minima/maxima, not confidence intervals. Rerank=400, patience=0, k=100; "
    "the legacy beam therefore refines only 100/256/400 candidates at ef100/256/512."), "",
    ("Default ordered IDs match all frozen v2 blocks (2304 lists per dataset); every measured "
    "four-arm result and encoded size matches the native diagnostic. The raw files, source/binary "
    "hashes, frozen input manifests, CPU/thread attributes and exact schedules are retained."), ""])
for name, (_, report) in datasets.items():
    lines.extend([f"### {name}", "", "| Serving / retention | ef | Actual recall | QPS median [min–max] |",
        "|---|---:|---:|---:|"])
    for row in report["rows"]:
        lines.append(f"| {row['method']} | {row['ef']} | {100*row['recall_ids_min']:.4f}% | "
            f"{row['qps_median']:.2f} [{row['qps_min']:.2f}–{row['qps_max']:.2f}] |")
    lines.extend(["", "| Required recall | Legacy: fastest sampled qualifying QPS | Direct | Direct + reservoir |",
        "|---|---:|---:|---:|"])
    points = {(r["arm"], r["target"]): r for r in report["targets"]}
    for target in (0.90, 0.95, 0.99):
        cells = []
        for arm in (0, 2, 3):
            p = points[arm, target]
            cells.append(f"{p['qps_median']:.2f} (ef={p['ef']}, recall={100*p['recall_ids_min']:.4f}%)" if p["reached"] else "Unreached")
        lines.append(f"| {100*target:.0f}% | {' | '.join(cells)} |")
    lines.append("")
gate = json.loads((ROOT / "final-gate.json").read_text())
summary["verification"] = gate
integration = json.loads((ROOT / "pr-integration.json").read_text())
summary["integration_verification"] = integration
lines.extend(["## Verification and limits", "",
    (f"Measured-source Linux all-features gate: {gate['passed']} Rust tests passed, {gate['ignored']} ignored; "
    f"{gate['native_python_passed']} native Python tests passed. Rust 1.89 MSRV library gate: "
    "603 passed, 4 ignored. Formatting passes. The measured-source Clippy receipt records two "
    "baseline lints in documents/postings.rs and triple_store.rs; no lint was suppressed."), "",
    (f"After integration with main at {integration['base_commit'][:7]}, the fresh Linux gate passed "
    f"{integration['linux_all_features']['passed']} Rust tests "
    f"({integration['linux_all_features']['ignored']} ignored) and "
    f"{integration['native_python']['passed']} native Python tests. The Rust 1.89 library gate "
    f"passed {integration['msrv_library']['passed']} tests "
    f"({integration['msrv_library']['ignored']} ignored). Source hashes, commands, logs and "
    "the separate passing Rust 1.96 all-features Clippy result are recorded in "
    "pr-integration.json. These checks do not "
    "remeasure the frozen timings."), "",
    ("Tests cover recovery after beam rejection, upper-layer deduplication/epoch wrap, unchanged "
    "traversal with patience/frontier/batched scoring, 127/128/129 storage boundaries, original "
    "magnitudes/IDs/ties/zero queries, and mapped/prefaulted/heap persistence. Compile-failure "
    "logs from updating complete parameter literals are preserved separately."), "",
    ("No topology or kernel change is included. Discovery remains the main high-recall limit; "
    "these measurements do not select a new degree policy. The three sampled ef values are "
    "not an exhaustive effort sweep. Validation margins remain narrow at 99%, and historical "
    "official results cannot certify the new variants. No new official or competitor-superiority "
    "claim is made; the previous inconclusive/stopped in-distribution regression evidence remains unchanged."), "",
    ("Reproduction: build-diagnose.sh, final-gate.sh, timing.sh and generate_report.py in this directory; "
    "the VIBE adapter/driver are benchmarks/public/qag_serving_{module,execute}.py. "
    "Decisions and execution history: [ledger.md](ledger.md)."), ""])
(ROOT / "README.md").write_text("\n".join(lines))
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
