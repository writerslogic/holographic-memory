# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run one frozen validation layout/revision pair, or a fixed legacy regression pair."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

from qag_execute import THREADS, digest, host, now, record, verify_inputs, write_new


def run(args):
    import h5py
    from qag_report import extract

    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}", args.run_id):
        raise ValueError("invalid run ID")
    vibe = Path.home() / "vibe-hms/vibe"
    bench = Path.home() / "vibe-hms/bench.sh"
    with h5py.File(args.view, "r") as source:
        metric, dim = str(source.attrs["distance"]), int(source.attrs["dimension"])
        if not args.legacy and (source.attrs.get("qag_scope") != "validation"
                                or len(source["test"]) != 256 or "learn" not in source):
            raise ValueError("new layout development requires reserved learn validation")
        if args.legacy and ("learn" in source or metric not in ("euclidean", "cosine", "normalized")):
            raise ValueError("legacy gate requires a complete original ID regression view")
        dependencies = {args.view.resolve()}
        for name in source:
            link = source.get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                dependencies.add((args.view.parent / link.filename).resolve())
            for filename, _, _ in source[name].external or []:
                dependencies.add(Path(filename).resolve())
    snapshots = args.out / "snapshots"
    snapshots.mkdir()
    label = "qag_cost_v3" if args.revision else "qag_layout"
    module = vibe / "vibe/algorithms" / label / "module.py"
    module.parent.mkdir(exist_ok=True)
    incoming = Path(__file__).with_name("qag_layout_module.py")
    if module.exists() and digest(module) != digest(incoming):
        raise ValueError("refusing to replace another layout adapter")
    if not module.exists():
        shutil.copyfile(incoming, module)
        (module.parent / "__init__.py").touch(exist_ok=True)
    extensions = {}
    inputs = []
    for arm, original in (("baseline", args.baseline), ("candidate", args.candidate)):
        path = snapshots / (arm + "-holographic_memory.so")
        shutil.copyfile(original, path)
        path.chmod(0o444)
        extensions[arm] = record(path)
        inputs.append(record(path))
    same_source = extensions["baseline"]["sha256"] == extensions["candidate"]["sha256"]
    if same_source == (args.legacy or args.revision):
        raise ValueError("layout controls need one binary; revision/legacy pairs need distinct binaries")
    settings = ((100, 100), (300, 400), (1000, 1000)) if args.legacy else ((100, 100), (256, 400), (512, 400))
    schedule = [[arm, ef, rr, rep] for rep in range(3) for ef, rr in settings
                for arm in ((0, 1) if rep != 1 else (1, 0))]
    bits = 4 if args.legacy else 8
    algorithm = "qag-cost-v3" if args.revision else "qag-layout-v2"
    definition = {"name": algorithm, "module": "vibe.algorithms." + label,
                  "constructor": "QAGLayout", "singularity_image": "hms", "disabled": False,
                  "ood": not args.legacy, "base_args": ["@metric"],
                  "run_groups": {"fixed": {"args": [64, 500, bits, args.legacy, args.revision],
                                             "query_arg_groups": [schedule]}}}
    folder = args.out / "definitions" / label
    folder.mkdir(parents=True)
    config = folder / "config.yml"
    write_new(config, {"float": {metric: [definition]}})
    for path in [bench, vibe / "run.py",
                 *(vibe / "vibe" / name for name in ("main.py", "runner.py", "workers.py", "numa.py", "definitions.py", "results.py", "distance.py")),
                 vibe / "vibe/algorithms/qag_pair/module.py", module, Path(__file__),
                 *(Path(__file__).with_name(name) for name in ("qag_report.py", "qag_execute.py", "qag_regression.py"))]:
        inputs.append(record(path))
        target = snapshots / f"source-{len(inputs):02d}-{path.name}"
        shutil.copyfile(path, target)
    inputs.extend(record(path) for path in sorted(dependencies))
    inputs.append(record(config))
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("HMS_QGRAPH_", "HMS_QAG_"))}
    environment.update({name: "1" for name in THREADS})
    environment.update(HMS_QAG_BASELINE_SO=extensions["baseline"]["path"],
                       HMS_QAG_CANDIDATE_SO=extensions["candidate"]["path"])
    if not args.legacy:
        if args.training is None:
            raise ValueError("validation layout pair requires the frozen construction file")
        inputs.append(record(args.training))
        environment.update(HMS_QAG_TRAINING_QUERIES=str(args.training.resolve()),
                           HMS_QAG_TRAINING_QUERIES_SHA256=digest(args.training))
    alias = f"qag-{'cost' if args.revision else 'layout'}-{args.run_id}-{dim}-{metric}"
    link = vibe / "data" / (alias + ".hdf5")
    if link.exists() or link.is_symlink() or (vibe / "results" / alias).exists():
        raise FileExistsError("existing VIBE alias/results")
    link.symlink_to(args.view.resolve())
    command = [str(bench), alias, "1", "vibe", str(folder.parent), "nolock"]
    before = {"schema_version": 1, "created_utc": now(), "host": host(), "command": command,
              "scope": "legacy regression" if args.legacy else "validation-only revision comparison" if args.revision else "validation-only layout comparison",
              "inputs": inputs, "schedule": schedule, "extensions": extensions,
              "harness_sha256": digest(vibe / "vibe/runner.py"),
              "environment": {k: v for k, v in environment.items() if k in THREADS or k.startswith("HMS_QAG_")}}
    write_new(args.out / "frozen.json", before)
    verify_inputs(before)
    process = subprocess.run(command, cwd=vibe, env=environment, capture_output=True, text=True)
    (args.out / "bench-stdout.txt").write_text(process.stdout + process.stderr)
    if process.returncode or re.findall(r"\brc=(\d+)\b", process.stdout) != ["0"]:
        raise RuntimeError("bench.sh did not complete successfully")
    raw_paths = list((vibe / "results" / alias / "100" / algorithm).glob("*.hdf5"))
    if len(raw_paths) != 1:
        raise ValueError("expected one raw layout result")
    raw = args.out / "raw.hdf5"
    shutil.copyfile(raw_paths[0], raw)
    shutil.copyfile(Path.home() / "vibe-hms/logs" / (alias + "-definitions.log"), args.out / "bench.log")
    with h5py.File(raw, "r") as results:
        if set(results) != {"_".join(map(str, group)) for group in schedule}:
            raise ValueError("raw layout schedule is incomplete")
        affinities = {tuple(group.attrs["qag_cpu_affinity"]) for group in results.values()}
        if len(affinities) != 1 or len(next(iter(affinities))) != 1:
            raise ValueError("layout groups must share one VIBE-selected CPU")
        for group in results.values():
            recorded = set(str(group.attrs["qag_thread_environment"]).split(","))
            if not {name + "=1" for name in THREADS} <= recorded:
                raise ValueError("layout thread policy differs")
    verify_inputs(before)
    receipt = args.out / "benchmark.json"
    write_new(receipt, {**before, "completed_utc": now(), "raw_results": record(raw)})
    report = extract(raw, args.view, receipt)
    if args.legacy:
        from qag_regression import matched_gate
        gate = matched_gate(report)
    else:
        rows = {(r["arm"], r["ef"], r["rerank"]): r for r in report["rows"]}
        points = []
        for ef, rr in settings:
            a, b = rows[(0, ef, rr)], rows[(1, ef, rr)]
            aa, bb = a["repetitions"], b["repetitions"]
            if any(x["ids_sha256"] != y["ids_sha256"] for x, y in zip(aa, bb)):
                raise ValueError("compaction changed returned IDs")
            points.append({"ef": ef, "rerank": rr, "recall": a["recall_ids_min"],
                           "paired_delta_percent": [100 * (y["qps"] / x["qps"] - 1) for x, y in zip(aa, bb)],
                           "median_delta_percent": 100 * (b["qps_median"] / a["qps_median"] - 1)})
        gate = {"identical_returned_ids": True, "points": points,
                "timing_limits": "three observed paired blocks; ranges are not confidence intervals"}
        names = ("reference8", "avxvnni8") if args.revision else ("fixed8", "compact8")
        for row in [*report["rows"], *report["targets"]]:
            row["method"] = names[row["arm"]]
    write_new(args.out / "report.json", report)
    write_new(args.out / "gate.json", gate)
    write_new(args.out / "completion.json", {"status": "success", "report": record(args.out / "report.json"),
                                               "gate": record(args.out / "gate.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("view", "baseline", "candidate", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--training", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--legacy", action="store_true")
    modes.add_argument("--revision", action="store_true")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if (socket.gethostname() != "writerslogic-02" or os.environ.get("HMS_QAG_OUTER_LOCK") != "1"
            or any(os.environ.get(name) != "1" for name in THREADS)):
        parser.error("use the shared outer timing lock and one-thread environment")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as error:
        if args.out.is_dir():
            write_new(args.out / "failure.json", {"error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
