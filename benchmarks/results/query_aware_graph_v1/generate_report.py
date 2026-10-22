# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Regenerate README.md from saved evidence without running benchmarks or gates.

Usage: uv run --no-project python generate_report.py [--checks checks.json]
Checks JSON optionally records fmt/cargo/native_python/benchmark_python/wrapper_native
entries with command, status, exit_code, attempts and a local log path. Missing
evidence remains unmeasured or unrun.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile


DATASETS = {"llama": ("llama-128-ip", 256921), "yi": ("yi-128-ip", 187843)}
REGRESSIONS = ("sift-128-euclidean", "glove-100-angular", "yahoo-minilm-384-normalized")
TARGETS = (0.90, 0.95, 0.99)
CHECKS = {
    "fmt": ("Formatting", "cargo fmt --check"),
    "cargo": ("Required Rust gate", "cargo +1.89.0 test --all-features"),
    "native_python": ("Native Python contracts", None),
    "benchmark_python": ("Benchmark Python contracts", None),
    "wrapper_native": ("Native benchmark wrapper", None),
}
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")


def load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("evidence contains a nonfinite or nonnumeric measurement")
    return value


def recall(value):
    if not 0 <= number(value) <= 1:
        raise ValueError("recall/coverage is outside 0..1")
    return f"{number(value):.10f}".rstrip("0").rstrip(".")


def qps(row):
    values = [number(row[key]) for key in ("qps_median", "qps_min", "qps_max")]
    if not 0 < values[1] <= values[0] <= values[2]:
        raise ValueError("QPS median/min/max are inconsistent")
    return f"{values[0]:,.2f} [{values[1]:,.2f}–{values[2]:,.2f}]"


def relative(path, root):
    return str(path.resolve().relative_to(root.resolve()))


def link(path, root, label=None):
    name = relative(path, root)
    return f"[{label or name}]({name})"


def validate_report(report, rows, queries, official):
    source, protocol = report["source"], report["protocol"]
    if (source.get("base_rows") != rows or source.get("queries") != queries
            or source.get("dimension") != 128 or source.get("metric") != "ip"
            or source.get("official_test") is not official
            or protocol.get("count") != 100 or protocol.get("repetitions") != [0, 1, 2]
            or protocol.get("passes_per_repetition") != 5 or protocol.get("timed_pass") != 5):
        raise ValueError("full-report dimensions, split semantics or timing protocol differ")
    if not report.get("rows"):
        raise ValueError("a measured full report has no configurations")
    for row in report["rows"]:
        if not 0 <= number(row["recall_ids_min"]) <= 1:
            raise ValueError("strict recall is outside 0..1")
        repetitions = row.get("repetitions", [])
        if sorted(item["repetition"] for item in repetitions) != [0, 1, 2]:
            raise ValueError("a measured configuration lacks three repetitions")
        if row["recall_ids_min"] != min(item["recall_ids"] for item in repetitions):
            raise ValueError("configuration recall is not the conservative repetition minimum")
        qps(row)


def full_report(root, key, official):
    name = f"{key}-official-v1" if official else f"{key}-full-validation-v1"
    path = root / "runs" / name / "report.json"
    report = load(path)
    if report is not None:
        validate_report(report, DATASETS[key][1], 1000 if official else 256, official)
    return report, path


def single_attribute(report, field):
    values = [item["resource_attrs"].get(field) for row in report["rows"]
              for item in row["repetitions"]]
    if not values or values[0] is None or any(value != values[0] for value in values):
        raise ValueError(f"missing or inconsistent comparison policy: {field}")
    return values[0]


def execution_policy(report):
    metadata = report["provenance"]["execution_receipt"]["metadata"]
    host = metadata["host"]
    cpu = {entry["field"].rstrip(":"): entry["data"] for entry in host["lscpu"]["lscpu"]}
    cpu_fields = ("Architecture", "Vendor ID", "Model name", "CPU family", "Model",
                  "Thread(s) per core", "Core(s) per socket", "Socket(s)", "Stepping", "Flags")
    if any(not cpu.get(field) for field in cpu_fields):
        raise ValueError("comparison CPU identity is incomplete")
    threads = dict(part.split("=", 1) for part in
                   single_attribute(report, "qag_thread_environment").split(","))
    if any(threads.get(name) != "1" for name in THREADS):
        raise ValueError("comparison thread environment is not one thread")
    affinity = single_attribute(report, "qag_cpu_affinity")
    if not isinstance(affinity, list) or len(affinity) != 1 or type(affinity[0]) is not int:
        raise ValueError("comparison runner affinity is not one CPU")
    if single_attribute(report, "qag_construction_queries") != 32768:
        raise ValueError("comparison construction pool differs from 32768 rows")
    pool = single_attribute(report, "qag_candidate_training_query_sha256")
    harness = report["provenance"]["harness_sha256"]
    if not isinstance(pool, str) or not re.fullmatch(r"[0-9a-f]{64}", pool):
        raise ValueError("comparison construction query hash is missing")
    if not isinstance(harness, str) or not re.fullmatch(r"[0-9a-f]{64}", harness):
        raise ValueError("comparison harness hash is missing")
    return (host["hostname"], host["platform"], tuple(cpu[field] for field in cpu_fields), tuple(affinity),
            tuple(threads[name] for name in THREADS), pool, harness)


def mlann_shape(row):
    descriptions = {item["resource_attrs"].get("qag_mlann_construction")
                    for item in row["repetitions"]}
    if len(descriptions) != 1:
        raise ValueError("MLANN construction differs across repetitions")
    description = next(iter(descriptions))
    match = re.match(r"RF, n_trees=(\d+), depth=(\d+);", description or "")
    shape = tuple(map(int, match.groups())) if match else None
    if shape not in ((10, 14), (80, 13)):
        raise ValueError("MLANN construction is outside the frozen comparator scope")
    return shape


def supplemental_mlann(root, official):
    supplements = {}
    for key, (name, rows) in DATASETS.items():
        path = root / "competitors" / "mlann-rf80" / name / "report.json"
        report = load(path)
        if report is None:
            continue
        validate_report(report, rows, 1000, True)
        main, main_path = official[key]
        if main is None:
            raise ValueError("RF80 comparison requires the completed main official report")
        if (report["source"] != main["source"]
                or report["provenance"]["source_sha256"] != main["provenance"]["source_sha256"]
                or report["protocol"] != main["protocol"]
                or execution_policy(report) != execution_policy(main)):
            raise ValueError("RF80 source, harness, CPU, thread, query-pool or cache policy differs")
        if (len(report["rows"]) != 3 or {row["ef"] for row in report["rows"]} != {1, 2, 3}
                or any(row["arm"] != 4 or mlann_shape(row) != (80, 13) for row in report["rows"])):
            raise ValueError("RF80 report differs from the frozen three-threshold schedule")
        freeze_path = main_path.parent / "frozen.json"
        freeze = load(freeze_path)
        if freeze is None:
            freeze_path = root / "sources" / f"{key}-official-v1-frozen.json"
            freeze = load(freeze_path)
        if freeze is None:
            raise ValueError("RF80 comparison requires the main official freeze receipt")
        expected = main["provenance"]["execution_receipt"]["metadata"]["freeze"]["sha256"]
        if hashlib.sha256(freeze_path.read_bytes()).hexdigest() != expected:
            raise ValueError("main official freeze does not match its execution receipt")
        bench = [item["sha256"] for item in freeze["inputs"] if item["path"] == freeze["bench"]]
        metadata = report["provenance"]["execution_receipt"]["metadata"]
        supplemental_bench = metadata.get("bench", metadata.get("files", {}).get("bench", {}))
        if len(bench) != 1 or supplemental_bench.get("sha256") != bench[0]:
            raise ValueError("RF80 comparison bench.sh policy hash differs or is missing")
        supplements[key] = (report, path)
    return supplements


def select(report, arms, target):
    if report is None:
        return None, "unmeasured"
    measured = [row for row in report["rows"] if row["arm"] in arms]
    if not measured:
        return None, "unmeasured"
    eligible = [row for row in measured if row["recall_ids_min"] >= target]
    if not eligible:
        maximum = max(row["recall_ids_min"] for row in measured)
        return None, f"unreached (max {recall(maximum)})"
    return max(eligible, key=lambda row: (row["qps_median"], -row["ef"], -row["rerank"])), None


def config(row):
    arm = row["arm"]
    if arm in (0, 1):
        return f"ef={row['ef']},rr={row['rerank']},p={row.get('patience', 0)}"
    if arm in (2, 3):
        return f"M={48 if arm == 2 else 96},L={row['ef']}"
    thresholds = {1: "5e-4", 2: "5e-5", 3: "5e-6"}
    trees, depth = mlann_shape(row)
    return f"RF{trees},d{depth},t={thresholds.get(row['ef'], 'unknown')}"


def cell(report, arms, target):
    row, missing = select(report, arms, target)
    if row is None:
        return missing
    return f"{recall(row['recall_ids_min'])}; {qps(row)}; {config(row)}"


def performance(root, reports, supplements=None):
    lines = [
        "| Dataset | Recall target | Before: strict recall; QPS [min–max]; config | After: strict recall; QPS [min–max]; config | RoarGraph: strict recall; QPS [min–max]; config | MLANN: strict recall; QPS [min–max]; config | After vs best measured CPU |",
        "|---|---:|---|---|---|---|---|",
    ]
    statuses = []
    for key, (name, _) in DATASETS.items():
        report, path = reports[key]
        supplement = (supplements or {}).get(key)
        comparisons = report
        if supplement:
            comparisons = {"rows": report["rows"] + supplement[0]["rows"]}
        dataset = link(path, root, name) if report else name
        for target in TARGETS:
            candidate, _ = select(report, {1}, target)
            comparator, _ = select(comparisons, {2, 3, 4}, target)
            delta = "unmeasured"
            if candidate and comparator:
                percent = 100 * (candidate["qps_median"] / comparator["qps_median"] - 1)
                method = comparator["method"]
                if comparator["arm"] == 4:
                    trees, depth = mlann_shape(comparator)
                    method += f" RF{trees}/d{depth}"
                delta = f"{method}: {percent:+.2f}%"
                if target in (0.90, 0.95):
                    statuses.append(candidate["qps_median"] >= 0.9 * comparator["qps_median"])
            elif target in (0.90, 0.95):
                statuses.append(None)
            lines.append(
                f"| {dataset} | {target:.2f} | {cell(report, {0}, target)} | "
                f"{cell(report, {1}, target)} | {cell(report, {2, 3}, target)} | "
                f"{cell(comparisons, {4}, target)} | {delta} |"
            )
    return lines, statuses


def diagnosis(root):
    lines = [
        "| Held-out learn split | B | O | V during B | E | OE | B retained coverage | B rerank-candidate coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, filename, rows in (
        ("llama 200k prefix", "llama-prefix-diagnosis.json", 200000),
        ("llama full base", "llama-full-b4.json", 256921),
        ("yi full base", "yi-full-b4.json", 187843),
    ):
        path = root / filename
        report = load(path)
        if report is None:
            lines.append(f"| {name} | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |")
            continue
        if (report["input"]["rows"] != rows or report["input"]["validation_queries"] != 256
                or report["input"]["official_test_read"] is not False
                or report["search"]["ef"] != 2000 or report["search"]["rerank"] != 1000
                or report["search"]["bits"] != 4 or report["search"]["k"] != 100
                or report["build"]["degree"] != 64 or report["build"]["build_ef"] != 500
                or report["build"]["seed"] != 24301):
            raise ValueError("diagnostic source does not match the frozen held-out protocol")
        values = report["aggregate"]
        b = values["B"]
        numbers = [values["B"]["recall"], values["O"]["recall"], b["evaluated_coverage"],
                   values["E"]["recall"], values["OE"]["recall"],
                   b["retained_coverage"], b["rerank_candidate_coverage"]]
        lines.append(f"| {link(path, root, name)} | " + " | ".join(recall(value) for value in numbers) + " |")
    return lines


def projected_ablation(root):
    lines = [
        "| Projected prefix config | B | V during B | B retained coverage | B rerank-candidate coverage | E | OE | E − B |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    identity = None
    for ef, rerank in ((100, 100), (160, 200)):
        path = root / f"llama-prefix-proj-q32768-d32-ef{ef}.json"
        report = load(path)
        if report is None:
            lines.append(f"| ef={ef},rr={rerank} | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |")
            continue
        source, build, search = report["input"], report["build"], report["search"]
        projection = build["projection"]
        if (source["rows"] != 200000 or source["validation_queries"] != 256
                or source["source_metric"] != "ip" or source["official_test_read"] is not False
                or build["degree"] != 64 or build["build_ef"] != 500 or build["seed"] != 24301
                or projection["stats"]["training_queries"] != 32768
                or projection["stats"]["projected_degree"] != 32
                or projection["input"]["official_test_read"] is not False
                or search["bits"] != 8 or search["residual_bits"] != 8
                or search["k"] != 100 or search["ef"] != ef or search["rerank"] != rerank
                or search.get("patience", 0) != 0):
            raise ValueError("projected ablation differs from the fixed held-out prefix protocol")
        current = (source, build["graph_hash"], projection["input"])
        if identity is not None and identity != current:
            raise ValueError("projected ablation changes the graph or held-out query rows")
        identity = current
        aggregate = report["aggregate"]
        baseline = aggregate["B"]
        values = [baseline["recall"], baseline["evaluated_coverage"], baseline["retained_coverage"],
                  baseline["rerank_candidate_coverage"], aggregate["E"]["recall"], aggregate["OE"]["recall"]]
        gain = number(aggregate["E"]["recall"]) - number(baseline["recall"])
        lines.append(f"| {link(path, root, f'ef={ef},rr={rerank}')} | "
                     + " | ".join(recall(value) for value in values) + f" | {gain:+.10f} |")
    return lines


def regression_tables(root):
    lines = [
        "| Dataset / original metric | Gate | Matched strict recall / config | Before QPS [min–max] | After QPS [min–max] | Median delta | Paired deltas |",
        "|---|---|---|---|---|---:|---|",
    ]
    statuses = []
    for name in REGRESSIONS:
        directory = root / "regression" / "default4-v1" / name
        gate, report = load(directory / "gate.json"), load(directory / "report.json")
        if gate is None or report is None:
            lines.append(f"| {name} | inconclusive | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |")
            statuses.append("inconclusive")
            continue
        if gate.get("gate") not in ("pass", "fail", "inconclusive"):
            raise ValueError("unknown recorded regression gate")
        source = report["source"]
        expected = {
            "sift-128-euclidean": (1000000, 10000, 128, "euclidean"),
            "glove-100-angular": (1183514, 10000, 100, "cosine"),
            "yahoo-minilm-384-normalized": (677305, 1000, 384, "normalized"),
        }[name]
        if (source["base_rows"], source["queries"], source["dimension"], source["metric"]) != expected:
            raise ValueError("regression report is not the full original dataset/metric")
        statuses.append(gate["gate"])
        lookup = {(row["arm"], row["ef"], row["rerank"]): row for row in report["rows"]}
        if len(gate.get("points", [])) != 3:
            raise ValueError("regression gate lacks its three frozen configurations")
        for point in gate["points"]:
            key = point["ef"], point["rerank"]
            baseline, candidate = lookup[(0, *key)], lookup[(1, *key)]
            if candidate["method"] != "candidate4":
                raise ValueError("regression report did not use the legacy four-bit candidate")
            for row in (baseline, candidate):
                if sorted(item["repetition"] for item in row["repetitions"]) != [0, 1, 2]:
                    raise ValueError("regression point lacks three repetitions")
            baseline_runs = {row["repetition"]: row for row in baseline["repetitions"]}
            candidate_runs = {row["repetition"]: row for row in candidate["repetitions"]}
            identical = all(baseline_runs[rep]["ids_sha256"] == candidate_runs[rep]["ids_sha256"]
                            for rep in range(3))
            ratios = [candidate_runs[rep]["qps"] / baseline_runs[rep]["qps"] for rep in range(3)]
            expected_gate = ("inconclusive" if not identical else "pass" if min(ratios) >= 0.97
                             else "fail" if max(ratios) < 0.97 else "inconclusive")
            if point["gate"] != expected_gate or point["identical_returned_ids"] != identical:
                raise ValueError("saved regression gate conflicts with the paired report")
            label = link(directory / "gate.json", root, name)
            matched = f"{recall(point['strict_recall'])}; ef={key[0]},rr={key[1]}"
            if not identical:
                matched += f"; candidate {recall(candidate['recall_ids_min'])}, IDs differ"
            deltas = ", ".join(f"{number(value):+.2f}%" for value in point["paired_delta_percent"])
            lines.append(
                f"| {label} | {point['gate']} | {matched} | {qps(baseline)} | {qps(candidate)} | "
                f"{number(point['median_qps_delta_percent']):+.2f}% | {deltas} |"
            )
    return lines, statuses


def resource_tables(root, official, validation, supplements=None):
    lines = [
        "| Dataset / phase | Arm | Build seconds (one build) | Encoded index bytes | Size delta vs baseline |",
        "|---|---|---:|---:|---:|",
    ]
    processes = [
        "| Dataset / phase | Process peak RSS KiB | VIBE USS delta KiB |",
        "|---|---:|---:|",
    ]
    arm_names = ("baseline4", "candidate8", "roargraph48", "roargraph96", "mlann_rf10_depth14")
    for key, (name, _) in DATASETS.items():
        report, path = official[key]
        phase = "official"
        if report is None:
            report, path = validation[key]
            phase = "validation"
        if report is None:
            lines.append(f"| {name} | all arms | unmeasured | unmeasured | unmeasured |")
            processes.append(f"| {name} | unmeasured | unmeasured |")
            continue
        attrs = [item["resource_attrs"] for row in report["rows"] for item in row["repetitions"]]
        def values(field):
            return {entry[field] for entry in attrs if field in entry}
        def single(field):
            found = values(field)
            if len(found) > 1:
                raise ValueError(f"inconsistent single-build resource measurement: {field}")
            return next(iter(found)) if found else None
        baseline_bytes = single("qag_baseline4_index_bytes")
        dataset = link(path, root, f"{name} / {phase}")
        for arm in arm_names:
            seconds = single(f"qag_{arm}_build_seconds")
            size = single(f"qag_{arm}_index_bytes")
            delta = f"{100 * (size / baseline_bytes - 1):+.2f}%" if size is not None and baseline_bytes else "unmeasured"
            lines.append(
                f"| {dataset} | {arm} | {'unmeasured' if seconds is None else f'{number(seconds):.3f}'} | "
                f"{'unmeasured' if size is None else f'{int(size):,}'} | {delta} |"
            )
        rss, uss = values("qag_process_peak_rss_kib"), values("index_size")
        processes.append(
            f"| {dataset} | {'unmeasured' if not rss else f'{int(max(rss)):,}'} | "
            f"{'unmeasured' if not uss else f'{max(uss):,.0f}'} |"
        )
        supplement = (supplements or {}).get(key)
        if supplement:
            extra, extra_path = supplement
            attrs = [item["resource_attrs"] for row in extra["rows"] for item in row["repetitions"]]
            seconds = single("qag_mlann_rf80_depth13_build_seconds")
            label = link(extra_path, root, f"{name} / separate RF80 official")
            lines.append(
                f"| {label} | mlann_rf80_depth13 | "
                f"{'unmeasured' if seconds is None else f'{number(seconds):.3f}'} | unmeasured | unmeasured |"
            )
            rss, uss = values("qag_process_peak_rss_kib"), values("index_size")
            processes.append(
                f"| {label} | {'unmeasured' if not rss else f'{int(max(rss)):,}'} | "
                f"{'unmeasured' if not uss else f'{max(uss):,.0f}'} |"
            )
    return lines + ["", "Build times are single observations. Main-run process RSS and USS include all co-resident arms. Separate RF80 process measurements include its input buffers and runtime; isolated index-only peak RSS and competitor serialized sizes are unmeasured.", ""] + processes


def check_table(root, path):
    checks = load(path) if path else None
    checks = checks or {}
    lines = ["| Check | Exact command / evidence | Status |", "|---|---|---|"]
    statuses = {}
    for key, (label, required_command) in CHECKS.items():
        entry = checks.get(key, {})
        status = entry.get("status", "unrun")
        status = {"passed": "pass", "failed": "fail"}.get(status, status)
        if status not in ("pass", "fail", "unrun"):
            raise ValueError("verification status must be pass, fail or unrun")
        command = entry.get("command", required_command)
        if command is not None and (not isinstance(command, str) or not command.strip()):
            raise ValueError("verification command must be a nonempty string")
        if required_command is not None and (command is None or command.strip() != required_command):
            raise ValueError("verification receipt names a different required command")
        if status != "unrun" and command is None:
            raise ValueError("executed verification requires the exact recorded command")
        log = entry.get("log")
        if status == "pass" and (entry.get("exit_code") != 0 or not log):
            raise ValueError("passing verification requires exit_code0 and a saved log")
        if status == "fail" and (entry.get("exit_code") in (None, 0) or not log):
            raise ValueError("failed verification requires a nonzero exit code and a saved log")
        if key == "cargo" and status != "unrun" and entry.get("attempts", 1) != 1:
            raise ValueError("the final Rust gate receipt must record exactly one attempt")
        evidence = f"`{command}`" if command else "No execution receipt"
        if log:
            log_path = Path(log)
            log_path = log_path if log_path.is_absolute() else root / log_path
            if not log_path.is_file():
                raise ValueError("verification log is missing")
            evidence += " — " + link(log_path, root, "log")
        lines.append(f"| {label} | {evidence} | {status} |")
        statuses[key] = status
    lines.append("| Additional Cargo suites | Clippy, default-feature suite, feature-matrix sweeps | unrun under the requested validation budget |")
    return lines, statuses


def render(root, checks=None):
    official = {key: full_report(root, key, True) for key in DATASETS}
    validation = {key: full_report(root, key, False) for key in DATASETS}
    supplements = supplemental_mlann(root, official)
    official_table, throughput = performance(root, official, supplements)
    validation_table, _ = performance(root, validation)
    regressions, regression_status = regression_tables(root)
    checks_table, check_status = check_table(root, checks)
    recall_pass = all(select(official[key][0], {1}, 0.99)[0] is not None for key in DATASETS)
    if not all(report is not None for report, _ in official.values()):
        outcome = "Official full-dataset performance remains unmeasured."
    elif recall_pass and all(value is True for value in throughput):
        outcome = "Both full datasets reach strict recall@100 ≥0.99, and the measured medians at 0.90/0.95 reach 90% of the fastest measured frozen CPU comparator."
    else:
        outcome = "The required recall/throughput goals remain incomplete within the frozen measured configurations."
    overall_regression = "fail" if "fail" in regression_status else "pass" if set(regression_status) == {"pass"} else "inconclusive"
    mlann_scope = ("RF80 depth13 is included only for datasets with a linked, matching separate official report: "
                   + ", ".join(link(path, root, DATASETS[key][0]) for key, (_, path) in supplements.items())
                   + ".") if supplements else "RF80 depth13 remains unrun in the saved evidence."
    lines = [
        "# Query-aware graph evidence", "",
        outcome,
        f"Regression gate: **{overall_regression}**. Formatting: **{check_status['fmt']}**. Required Rust gate: **{check_status['cargo']}**.",
        "",
        "Baseline SHA: `08d37778a66b58709461456e5b9a7ab6a05b399f`. Source decisions, configurations, seeds and commands are retained in [ledger.md](ledger.md); dataset/runtime fingerprints are in [provenance.json](provenance.json).",
        "",
        "B is ordinary entry with coded routing; O starts at the true nearest base vector; V is true-top100 coverage among all unique distance-evaluated vertices during B; E uses exact routing; OE combines oracle entry and exact routing. All diagnostic rows use 256 held-out learn queries, degree=64, build_ef=500, seed=24301, ef=2000, rerank=1000 and exact original-IP ground truth.",
        "",
        *diagnosis(root), "",
        "Observed prefix oracle-entry gain is 0.0002734375; exact-routing gain is 0.0259765625. The prefix supports routing/retention precision as the larger loss at that operating point. Full-base diagnostic rows retain the narrower evidence for each dataset.",
        "",
        "**Conditional projected-graph ablation, untimed.** Both rows use the same llama 200k graph, 8-bit codes plus 8-bit residuals, 32,768 construction queries, quota 32, seed 24301 and the same 256 held-out learn queries. Only ef/rerank changes between rows; E/OE replace coded routing within each row. These are prefix diagnostics, not official-test results.",
        "",
        *projected_ablation(root), "",
        "At ef=100, evaluated top-100 coverage exceeds retained coverage and exact routing raises recall. This supports testing whether distance refinement before admission helps at small beams. No E/OE QPS or refinement speedup was measured.",
        "",
        "**Official full-dataset Pareto points.** Every measured QPS entry is the saved fifth pass from each of three alternating blocks: median [min–max], one thread, writerslogic-02, common VIBE harness/affinity/cache policy. Selection uses actual minimum strict ID recall across repetitions ≥target, with no interpolation. Config p is the stale-expansion patience; RoarGraph L is its query width.",
        "",
        *official_table, "",
        "Comparator deltas cover only the frozen measured RoarGraph M48/M96 and MLANN RF10 depth14 configurations, plus matching separate RF80 depth13 reports where available. " + mlann_scope + " These measurements establish neither the global best CPU competitor nor cross-machine parity. The frozen construction policy uses the same 32,768 query rows for all trained arms. MLANN has no exposed seed control; each run reuses one forest across repetitions.",
        "",
        "**Full-base validation only.** These 256 queries come from reserved learn rows and are excluded from construction; this table does not substitute for official-test results.",
        "",
        *validation_table, "",
        "**In-distribution regression gate.** Both arms use the legacy 4-bit path with no training queries at frozen ef/rerank settings. The gate requires identical returned IDs and uses the three observed paired QPS ratios: all ≥0.97 pass, all <0.97 fail, mixed evidence is inconclusive. The observed paired range is not a statistical confidence interval. Glove's VIBE cosine alias preserves original angular coordinates and saved neighbor IDs.",
        "",
        *regressions, "",
        "**Resource observations.**", "",
        *resource_tables(root, official, validation, supplements), "",
        "**Persisted formats and compatibility.**", "",
        "| Format / contract | Change | Verification |",
        "|---|---|---|",
        "| Graph cache `HMSGRF04` | no layout change | required Rust gate covers compatibility tests; current status above |",
        "| Query-trained Graph cache writes | rejected before touching the target, because the cache key omits training queries | rejection contract belongs to the required Rust gate |",
        "| Encoded VGraph cache `HMSVGR01` | existing persisted layout, including trained adjacency | round-trip contracts belong to the required Rust gate; no layout version bump or new persisted fields |",
        "| Legacy no-training path/default 4-bit codes | preserved by optional API | None/default byte-parity and binding contracts belong to the recorded validation gates |",
        "",
        "**Verification.**", "", *checks_table, "",
        "**Historical target, separate protocol.** The supplied llama standing file reports VIBE epsilon-based recall and best-of-five QPS under concurrent pinned workers. These sourced values are not new paired medians and are excluded from the comparisons above.",
        "",
        "| llama target | Historical HMS QPS | Historical RoarGraph QPS |",
        "|---:|---:|---:|",
        "|0.90|700.8253394103064|5174.602867638178|",
        "|0.95|290.3855707947149|2947.0522964206903|",
        "|0.99|unreached (max recall 0.9585)|519.6949572300871|",
        "",
        "Source: [supplied standing JSON](../vibe_v1/llama-128-ip-standing-wl02.json); current audit and strict-ID reconstructions are in [provenance.json](provenance.json).",
        "",
        "Regenerate this file with `uv run --no-project python benchmarks/results/query_aware_graph_v1/generate_report.py` after pulling completed run reports; add `--checks benchmarks/results/query_aware_graph_v1/checks.json` when the sole final-gate receipt exists.",
        "",
    ]
    return "\n".join(lines)


def self_test():
    def row(arm, accuracy, speed):
        return {"arm": arm, "method": "test", "recall_ids_min": accuracy,
                "qps_median": speed, "qps_min": speed, "qps_max": speed,
                "ef": 100, "rerank": 100}
    report = {"rows": [row(1, 0.949999999999, 1000), row(1, 0.95, 10)]}
    chosen, missing = select(report, {1}, 0.95)
    assert chosen["qps_median"] == 10 and missing is None
    assert select(report, {1}, 0.99)[1].startswith("unreached")
    assert select(None, {1}, 0.99)[1] == "unmeasured"
    invalid = {"source": {"base_rows": 200000, "queries": 1000, "dimension": 128,
                          "metric": "ip", "official_test": True},
               "protocol": {"count": 100, "repetitions": [0, 1, 2],
                            "passes_per_repetition": 5, "timed_pass": 5}, "rows": [row(1, 1, 1)]}
    try:
        validate_report(invalid, 256921, 1000, True)
    except ValueError:
        pass
    else:
        raise AssertionError("prefix data were accepted as a full official result")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _, statuses = check_table(root, None)
        assert statuses == {key: "unrun" for key in CHECKS}
        (root / "native.log").write_text("passed\n")
        checks = root / "checks.json"
        checks.write_text(json.dumps({"native_python": {
            "command": "uv run python -m pytest tests/python/test_vector_index.py",
            "status": "pass", "exit_code": 0, "log": "native.log"}}))
        lines, statuses = check_table(root, checks)
        assert statuses["native_python"] == "pass" and statuses["cargo"] == "unrun"
        assert any("native.log" in line for line in lines)
        cpu_fields = ("Architecture", "Vendor ID", "Model name", "CPU family", "Model",
                      "Thread(s) per core", "Core(s) per socket", "Socket(s)", "Stepping", "Flags")
        host = {"hostname": "writerslogic-02", "platform": "linux",
                "lscpu": {"lscpu": [{"field": field + ":", "data": field} for field in cpu_fields]}}
        source = {"base_rows": 256921, "queries": 1000, "dimension": 128,
                  "metric": "ip", "official_test": True}
        protocol = {"count": 100, "repetitions": [0, 1, 2],
                    "passes_per_repetition": 5, "timed_pass": 5}
        freeze = root / "runs" / "llama-official-v1" / "frozen.json"
        freeze.parent.mkdir(parents=True)
        freeze.write_text(json.dumps({"bench": "/bench.sh", "inputs": [
            {"path": "/bench.sh", "sha256": "d" * 64}]}))
        metadata = {"host": host, "bench": {"sha256": "d" * 64},
                    "freeze": {"sha256": hashlib.sha256(freeze.read_bytes()).hexdigest()}}
        def mlann_report(trees, depth, speed):
            rows = []
            for effort in (1, 2, 3):
                result = row(4, 0.95, speed)
                result.update(ef=effort, rerank=0, repetitions=[{
                    "repetition": repetition, "recall_ids": 0.95, "resource_attrs": {
                        "qag_thread_environment": ",".join(name + "=1" for name in THREADS),
                        "qag_cpu_affinity": [7], "qag_construction_queries": 32768,
                        "qag_candidate_training_query_sha256": "c" * 64,
                        "qag_mlann_construction": f"RF, n_trees={trees}, depth={depth}; frozen"}}
                    for repetition in (0, 1, 2)])
                rows.append(result)
            return {"source": source, "protocol": protocol, "rows": rows,
                    "provenance": {"source_sha256": "a" * 64, "harness_sha256": "b" * 64,
                                   "execution_receipt": {"metadata": metadata}}}
        main = mlann_report(10, 14, 100)
        extra = mlann_report(80, 13, 200)
        extra_path = root / "competitors" / "mlann-rf80" / "llama-128-ip" / "report.json"
        extra_path.parent.mkdir(parents=True)
        extra_path.write_text(json.dumps(extra))
        official = {"llama": (main, freeze.parent / "report.json"), "yi": (None, root / "absent")}
        supplements = supplemental_mlann(root, official)
        selected, _ = select({"rows": main["rows"] + supplements["llama"][0]["rows"]}, {4}, 0.95)
        assert config(selected) == "RF80,d13,t=5e-4"
        extra["rows"][0]["repetitions"][0]["resource_attrs"]["qag_cpu_affinity"] = [8]
        extra_path.write_text(json.dumps(extra))
        try:
            supplemental_mlann(root, official)
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched RF80 CPU affinity was accepted")
    print("report selection/scope/verification/comparator self-checks: pass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--checks", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    root = args.evidence_dir.resolve(strict=True)
    if args.checks is not None and not args.checks.is_file():
        raise FileNotFoundError("explicit verification receipt is missing")
    checks = args.checks or (root / "checks.json")
    content = render(root, checks if checks.is_file() else None)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                     prefix=".README-", suffix=".tmp", delete=False) as output:
        output.write(content)
        temporary = Path(output.name)
    temporary.replace(root / "README.md")
    print(root / "README.md")


if __name__ == "__main__":
    main()
