# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run all three frozen default-encoding regressions within one outer timing lock."""

import argparse
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

from qag_execute import THREADS, host, now, record, resolved, verify_inputs, write_new


SETTINGS = ((100, 100), (300, 400), (1000, 1000))
SCHEDULE = [
    [arm, ef, rerank, repetition]
    for repetition in range(3)
    for ef, rerank in SETTINGS
    for arm in ((0, 1) if repetition % 2 == 0 else (1, 0))
]


def copy_new(source, target):
    with Path(source).open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    target.chmod(0o444)
    return record(target)


def definitions_for(vibe, folder, metric, dim):
    sys.path.insert(0, str(vibe))
    from vibe.definitions import get_definitions

    definitions = get_definitions(dim, "float", metric, 100, str(folder))
    if len(definitions) != 1:
        raise ValueError("regression requires exactly one VIBE definition")
    definition = definitions[0]
    if (definition.module != "vibe.algorithms.qag_pair"
            or definition.constructor != "QAGPair"
            or definition.arguments != [metric, 64, 500, 4]
            or definition.ood or definition.disabled or definition.gpu):
        raise ValueError("regression must use the two legacy four-bit arms without training")
    schedule = []
    for group in definition.query_argument_groups:
        if len(group) == 1 and isinstance(group[0], (list, tuple)):
            group = group[0]
        if any(type(value) is not int for value in group):
            raise ValueError("regression query arguments must be integers")
        schedule.append(list(group))
    if schedule != SCHEDULE:
        raise ValueError("regression schedule differs from the frozen alternating 18 groups")
    return definition


def matched_gate(report):
    rows = {(row["arm"], row["ef"], row["rerank"]): row for row in report["rows"]}
    if len(report["rows"]) != 6 or set(rows) != {(arm, ef, rr) for arm in (0, 1) for ef, rr in SETTINGS}:
        raise ValueError("report does not contain exactly the six frozen configurations")
    points = []
    for ef, rerank in SETTINGS:
        baseline, candidate = rows[(0, ef, rerank)], rows[(1, ef, rerank)]
        baseline_runs = {run["repetition"]: run for run in baseline["repetitions"]}
        candidate_runs = {run["repetition"]: run for run in candidate["repetitions"]}
        matched = all(
            baseline_runs[rep]["ids_sha256"] == candidate_runs[rep]["ids_sha256"]
            for rep in range(3)
        )
        paired_ratios = [
            candidate_runs[rep]["qps"] / baseline_runs[rep]["qps"]
            for rep in range(3)
        ]
        if not matched:
            status = "inconclusive"
        elif min(paired_ratios) >= 0.97:
            status = "pass"
        elif max(paired_ratios) < 0.97:
            status = "fail"
        else:
            status = "inconclusive"
        points.append({
            "ef": ef, "rerank": rerank, "gate": status,
            "strict_recall": baseline["recall_ids_min"], "identical_returned_ids": matched,
            "baseline_qps_median": baseline["qps_median"],
            "baseline_qps_range": [baseline["qps_min"], baseline["qps_max"]],
            "candidate_qps_median": candidate["qps_median"],
            "candidate_qps_range": [candidate["qps_min"], candidate["qps_max"]],
            "median_qps_delta_percent": 100 * (candidate["qps_median"] / baseline["qps_median"] - 1),
            "paired_delta_percent": [100 * (ratio - 1) for ratio in paired_ratios],
        })
    statuses = {point["gate"] for point in points}
    status = "fail" if "fail" in statuses else "pass" if statuses == {"pass"} else "inconclusive"
    return {
        "gate": status, "points": points,
        "rule": "identical returned IDs at each matched configuration; all three paired losses <=3% passes, all >3% fails, mixed evidence is inconclusive",
        "limits": "paired min-max is an observed range, not a statistical confidence interval",
    }


def prepare_source(dataset, directory):
    import h5py
    from qag_native_hdf5 import export

    source = dataset["source"]
    if dataset["native"]:
        view = directory / (dataset["alias"] + ".hdf5")
        metadata = export(source, view)
        export_receipt = record(view.with_suffix(".json"))
        dependencies = [item for item in metadata["inputs"].values()]
    else:
        view = source
        export_receipt = None
        dependencies = [record(source)]
    with h5py.File(view, "r") as handle:
        if (handle.attrs.get("distance") != dataset["metric"]
                or int(handle.attrs.get("dimension", 0)) != dataset["dimension"]
                or handle["train"].shape != (dataset["rows"], dataset["dimension"])
                or handle["test"].shape != (dataset["queries"], dataset["dimension"])
                or handle["neighbors"].shape != (dataset["queries"], 100)
                or "learn" in handle or "learn_neighbors" in handle):
            raise ValueError("regression source must have the complete original base/test and no learn input")
    return view, export_receipt, dependencies


def run_dataset(dataset, frozen, directory):
    import h5py
    from qag_report import write_report
    from vibe.results import build_result_filepath

    directory.mkdir()
    view, export_receipt, dependencies = prepare_source(dataset, directory)
    vibe = Path(frozen["vibe"])
    definition = definitions_for(vibe, Path(frozen["definitions"]),
                                 dataset["metric"], dataset["dimension"])
    alias = dataset["alias"]
    link = vibe / "data" / (alias + ".hdf5")
    raw_root = vibe / "results" / alias
    if link.exists() or link.is_symlink() or raw_root.exists():
        raise FileExistsError("refusing to reuse an existing VIBE alias or raw result directory")
    link.symlink_to(view.resolve())
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("WARM_QGRAPH_", "WARM_QAG_"))}
    environment.update({name: "1" for name in THREADS})
    environment.update(WARM_QAG_BASELINE_SO=frozen["extensions"]["baseline"]["path"],
                       WARM_QAG_CANDIDATE_SO=frozen["extensions"]["candidate"]["path"])
    command = [frozen["bench"], alias, "1", "vibe", frozen["definitions"], "nolock"]
    before = {
        "schema_version": 1, "created_utc": now(), "dataset": dataset["name"],
        "alias": alias, "host": host(), "command": command,
        "outer_flock": str(Path.home() / "warm-timing.lock"),
        "harness_sha256": frozen["harness_sha256"], "extensions": frozen["extensions"],
        "dataset_view": record(view), "export_receipt": export_receipt,
        "inputs": [*frozen["inputs"], *dependencies, record(view)],
        "environment": {name: environment[name] for name in environment
                        if name in THREADS or name.startswith("WARM_QAG_")},
        "policy": frozen["policy"],
    }
    write_new(directory / "before.json", before)
    verify_inputs(before)
    process = subprocess.run(command, cwd=vibe, env=environment, capture_output=True, text=True)
    with (directory / "bench-stdout.txt").open("x") as stream:
        stream.write(process.stdout + process.stderr)
    log = Path.home() / "vibe-warm/logs" / (alias + "-" + Path(frozen["definitions"]).name + ".log")
    if log.exists():
        copy_new(log, directory / "bench.log")
    printed_status = re.findall(r"\brc=(\d+)\b", process.stdout)
    if process.returncode or printed_status != ["0"]:
        raise RuntimeError(f"bench.sh failed: process={process.returncode}, printed={printed_status}")
    relative, _ = build_result_filepath(alias, 100, definition)
    raw = vibe / relative
    expected = {"_".join(map(str, group)) for group in SCHEDULE}
    with h5py.File(raw, "r") as handle:
        if set(handle) != expected:
            raise ValueError("raw results do not contain exactly the 18 frozen groups")
        for group in handle.values():
            if (group.attrs.get("run_count") != 5
                    or group.attrs.get("qag_candidate_code_bits") != 4
                    or group.attrs.get("qag_candidate_training_queries") != 0):
                raise ValueError("raw results violate five-pass/default-four-bit/no-training policy")
    verify_inputs(before)
    saved_raw = directory / "raw.hdf5"
    copy_new(raw, saved_raw)
    receipt = directory / "benchmark.json"
    write_new(receipt, {
        **before, "completed_utc": now(), "status": "benchmark_complete",
        "raw_results": record(saved_raw), "vibe_raw_results": record(raw),
        "before": record(directory / "before.json"),
    })
    report_path = directory / "report.json"
    report = write_report(saved_raw, view, report_path, receipt)
    report_path.chmod(0o444)
    gate = matched_gate(report)
    write_new(directory / "gate.json", gate)
    completed = {
        "dataset": dataset["name"], "status": "complete", "gate": gate["gate"],
        "report": record(report_path), "gate_report": record(directory / "gate.json"),
        "raw_results": record(saved_raw), "benchmark_receipt": record(receipt),
    }
    write_new(directory / "completion.json", completed)
    print(report_path, flush=True)
    return completed


def run(args):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}", args.run_id):
        raise ValueError("run ID must be a simple unique basename of at most 48 characters")
    root = args.runs.expanduser().resolve() / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    try:
        snapshots = root / "snapshots"
        snapshots.mkdir()
        extensions = {}
        for arm, path in (("baseline", args.baseline), ("candidate", args.candidate)):
            original = record(path)
            saved = copy_new(original["path"], snapshots / (arm + "-writerslogic_warm.so"))
            extensions[arm] = {**saved, "original_path": original["path"]}
        if extensions["baseline"]["sha256"] == extensions["candidate"]["sha256"]:
            raise ValueError("baseline and candidate must be distinct extension builds")
        vibe, bench = resolved(args.vibe), resolved(args.bench)
        definitions = root / "definitions" / "qag_pair"
        definitions.mkdir(parents=True)
        config = copy_new(resolved(args.config), definitions / "config.yml")
        for metric, dim in (("euclidean", 128), ("cosine", 100), ("normalized", 384)):
            definitions_for(vibe, definitions.parent, metric, dim)
        source_files = [
            Path(__file__), Path(__file__).with_suffix(".sh"),
            *(Path(__file__).with_name(name) for name in
              ("qag_execute.py", "qag_report.py", "qag_native_hdf5.py")),
            bench, vibe / "run.py",
            *(vibe / "vibe" / name for name in
              ("main.py", "runner.py", "workers.py", "numa.py", "distance.py", "results.py", "definitions.py")),
            vibe / "vibe/algorithms/qag_pair/module.py",
        ]
        inputs = [config, *extensions.values()]
        sources = []
        for index, path in enumerate(source_files):
            original = record(path)
            inputs.append(original)
            saved = copy_new(path, snapshots / f"source-{index:02d}-{path.name}")
            sources.append({**saved, "original_path": original["path"]})
        prefix = "qagreg-run_" + args.run_id.replace("-", "_")
        datasets = [
            {"name": "sift-128-euclidean", "metric": "euclidean", "dimension": 128,
             "rows": 1000000, "queries": 10000, "native": True,
             "source": resolved(args.sift_native), "alias": prefix + "-sift-128-euclidean"},
            {"name": "glove-100-angular", "metric": "cosine", "dimension": 100,
             "rows": 1183514, "queries": 10000, "native": True,
             "source": resolved(args.glove_native), "alias": prefix + "-glove-100-cosine"},
            {"name": "yahoo-minilm-384-normalized", "metric": "normalized", "dimension": 384,
             "rows": 677305, "queries": 1000, "native": False,
             "source": resolved(args.yahoo_source), "alias": prefix + "-yahoo-minilm-384-normalized"},
        ]
        frozen = {
            "schema_version": 1, "created_utc": now(), "run_id": args.run_id,
            "host": host(), "vibe": str(vibe), "bench": str(bench),
            "definitions": str(definitions.parent.resolve()), "extensions": extensions,
            "inputs": inputs, "source_snapshots": sources,
            "harness_sha256": record(vibe / "vibe/runner.py")["sha256"],
            "vibe_sha": subprocess.check_output(["git", "-C", str(vibe), "rev-parse", "HEAD"], text=True).strip(),
            "schedule": SCHEDULE,
            "datasets": [{**item, "source": str(item["source"])} for item in datasets],
            "policy": {
                "candidate_code_bits": 4, "training_queries": None, "parallelism": 1,
                "threads": {name: "1" for name in THREADS},
                "cache": "both indexes co-resident; four preceding warm passes; saved fifth pass reported",
                "repetitions": [0, 1, 2], "settings": SETTINGS,
                "lock": "one outer flock across all exports, benchmarks and reports",
            },
        }
        write_new(root / "frozen.json", frozen)
        results = []
        for dataset in datasets:
            results.append(run_dataset(dataset, frozen, root / dataset["name"]))
        statuses = {item["gate"] for item in results}
        overall = "fail" if "fail" in statuses else "pass" if statuses == {"pass"} else "inconclusive"
        write_new(root / "completion.json", {"created_utc": now(), "status": "complete",
                                            "gate": overall, "datasets": results})
        print(root / "completion.json", flush=True)
    except BaseException as error:
        write_new(root / "failure.json", {"created_utc": now(), "status": "failed",
                                         "error_type": type(error).__name__, "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline", type=Path,
                        default=Path.home() / "sol-qag/artifacts/baseline-writerslogic_warm.so")
    parser.add_argument("--candidate", type=Path,
                        default=Path.home() / "sol-qag/artifacts/trained-writerslogic_warm.so")
    parser.add_argument("--runs", type=Path, default=Path.home() / "sol-qag/regression-runs")
    parser.add_argument("--vibe", type=Path, default=Path.home() / "vibe-warm/vibe")
    parser.add_argument("--bench", type=Path, default=Path.home() / "vibe-warm/bench.sh")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "regression/qag_pair/config.yml")
    parser.add_argument("--sift-native", type=Path,
                        default=Path.home() / "warm-bench-decisions/data/sift-128-euclidean")
    parser.add_argument("--glove-native", type=Path,
                        default=Path.home() / ".cache/warm-bench/glove-100-angular")
    parser.add_argument("--yahoo-source", type=Path,
                        default=Path.home() / "vibe-warm/vibe/data/yahoo-minilm-384-normalized.hdf5")
    args = parser.parse_args()
    if (sys.platform != "linux" or socket.gethostname().split(".")[0] != "writerslogic-02"
            or os.environ.get("WARM_QAG_OUTER_LOCK") != "1"
            or any(os.environ.get(name) != "1" for name in THREADS)):
        parser.error("execute on writerslogic-02 through qag_regression.sh and its outer flock")
    run(args)


if __name__ == "__main__":
    main()
