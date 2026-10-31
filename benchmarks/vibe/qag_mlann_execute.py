# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Run the locally frozen RF80/depth13 comparator on an unchanged official QAG view."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

sys.dont_write_bytecode = True

from qag_execute import THREADS, digest, host, now, record, resolved, verify_inputs, write_new


SCHEDULE = [[4, effort, 0, repetition] for repetition in range(3)
            for effort in ((3, 2, 1) if repetition == 1 else (1, 2, 3))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("view", "training-queries", "baseline", "candidate", "config", "freeze", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--vibe", type=Path, default=Path.home() / "vibe-warm/vibe")
    parser.add_argument("--bench", type=Path, default=Path.home() / "vibe-warm/bench.sh")
    args = parser.parse_args()
    if (sys.platform != "linux" or socket.gethostname().split(".")[0] != "writerslogic-02"
            or os.environ.get("WARM_QAG_OUTER_LOCK") != "1"
            or any(os.environ.get(name) != "1" for name in THREADS)):
        parser.error("run on writerslogic-02 through qag_mlann_execute.sh and its outer flock")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", args.run_id):
        parser.error("run ID must be a simple unique basename")
    vibe, view, training = map(resolved, (args.vibe, args.view, args.training_queries))
    bench, config, local_freeze = map(resolved, (args.bench, args.config, args.freeze))
    sys.path.insert(0, str(vibe))
    from vibe.definitions import get_definitions
    from vibe.results import build_result_filepath
    from qag_report import write_report
    import h5py

    pinned = json.loads(local_freeze.read_text())
    source_paths = {
        "mlann_pair/module.py": vibe / "vibe/algorithms/mlann_pair/module.py",
        "mlann_pair/__init__.py": vibe / "vibe/algorithms/mlann_pair/__init__.py",
        "mlann_pair/config.yml": config,
        "qag_mlann_execute.py": Path(__file__),
        "qag_mlann_execute.sh": Path(__file__).with_suffix(".sh"),
        "qag_pair/module.py": vibe / "vibe/algorithms/qag_pair/module.py",
        "qag_execute.py": Path(__file__).with_name("qag_execute.py"),
        "qag_report.py": Path(__file__).with_name("qag_report.py"),
    }
    if set(pinned["files"]) != set(source_paths) or pinned["query_schedule"] != SCHEDULE:
        raise ValueError("local comparator freeze has unexpected inputs or query schedule")
    for name, path in source_paths.items():
        if digest(path) != pinned["files"][name]:
            raise ValueError(f"locally frozen comparator source changed: {name}")
    dependencies = {view}
    with h5py.File(view, "r") as handle:
        if (handle.attrs.get("qag_scope") != "official_test" or handle.attrs.get("distance") != "ip"
                or handle.attrs.get("dimension") != 128 or handle["learn"].shape != (32768, 128)
                or handle["learn_neighbors"].shape != (32768, 100)):
            raise ValueError("requires the existing full official128-IP view and its32768 learn rows")
        original_sha = str(handle.attrs["qag_original_source_sha256"])
        for name in ("train", "test", "neighbors", "distances", "learn", "learn_neighbors"):
            link = handle.get(name, getlink=True)
            if not isinstance(link, h5py.ExternalLink):
                raise ValueError("requires the unchanged external-link QAG official view")
            target = Path(link.filename)
            dependencies.add(resolved(target if target.is_absolute() else view.parent / target))
        original_link = handle.get("train", getlink=True)
        original_path = Path(original_link.filename)
        original = resolved(original_path if original_path.is_absolute() else view.parent / original_path)
    if digest(original) != original_sha:
        raise ValueError("official view's original source hash differs")
    spec = importlib.util.find_spec("mlann")
    if spec is None or spec.origin is None:
        raise ValueError("MLANN native package is unavailable")
    binary_paths = {Path(spec.origin)}
    for folder in spec.submodule_search_locations or ():
        binary_paths.update(Path(folder).rglob("*.so"))
    runtime = [bench, vibe / "run.py", *(vibe / "vibe" / name for name in
               ("main.py", "runner.py", "results.py", "definitions.py", "datasets.py"))]
    inputs = [record(path) for path in {*source_paths.values(), *dependencies, *binary_paths,
                                       *runtime, training, args.baseline, args.candidate, local_freeze}]
    directory = args.out.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    try:
        definitions = directory / "definitions" / "mlann_pair"
        definitions.mkdir(parents=True)
        shutil.copyfile(config, definitions / "config.yml")
        (definitions / "config.yml").chmod(0o444)
        snapshots = directory / "sources"
        snapshots.mkdir()
        archived = []
        for index, path in enumerate([*source_paths.values(), *runtime]):
            target = snapshots / f"{index:02d}-{path.name}"
            shutil.copyfile(path, target)
            target.chmod(0o444)
            archived.append({**record(target), "original_path": str(path)})
        parsed = get_definitions(dimension=128, point_type="float", distance_metric="ip", count=100,
                                 base_dir=str(definitions.parent))
        if (len(parsed) != 1 or parsed[0].arguments != ["ip", 80, 13]
                or parsed[0].module != "vibe.algorithms.mlann_pair"
                or parsed[0].query_argument_groups != [[group] for group in SCHEDULE]):
            raise ValueError("VIBE parsed an unexpected RF80 comparator definition")
        alias = "qag-mlann-rf80-" + args.run_id + "-128-ip"
        relative, _ = build_result_filepath(alias, 100, parsed[0])
        raw = vibe / relative
        dataset_path = vibe / "data" / (alias + ".hdf5")
        if raw.parents[2].exists() or dataset_path.exists():
            raise FileExistsError("refusing to reuse an existing VIBE alias or raw results")
        dataset_path.symlink_to(view)
        reserved = directory / "reserved-validation-ids.u32"
        with reserved.open("xb"):
            pass
        reserved.chmod(0o444)
        environment = {name: value for name, value in os.environ.items()
                       if not name.startswith(("WARM_QGRAPH_", "WARM_QAG_"))}
        environment.update({name: "1" for name in THREADS})
        environment.update(WARM_QAG_BASELINE_SO=str(resolved(args.baseline)),
                           WARM_QAG_CANDIDATE_SO=str(resolved(args.candidate)),
                           WARM_QAG_VALIDATION_IDS=str(reserved),
                           WARM_QAG_TRAINING_QUERIES=str(training),
                           WARM_QAG_TRAINING_QUERIES_SHA256=digest(training))
        command = [str(bench), alias, "1", "vibe", str(definitions.parent), "nolock"]
        before = {"schema_version": 1, "created_utc": now(), "phase": "official_mlann_rf80",
                  "local_freeze": record(local_freeze), "host": host(), "bench": record(bench),
                  "harness_sha256": digest(vibe / "vibe/runner.py"), "command": command,
                  "source": record(view), "original_source": record(original), "inputs": inputs,
                  "source_snapshots": archived, "query_schedule": SCHEDULE,
                  "outer_flock": str(Path.home() / "warm-timing.lock"),
                  "construction_query_sha256": digest(training),
                  "environment": {name: environment[name] for name in environment
                                  if name in THREADS or name.startswith("WARM_QAG_")},
                  "policy": "same official view, construction pool, affinity policy and five-pass cache protocol; fixed RF80/depth13 comparator only"}
        write_new(directory / "before.json", before)
        verify_inputs(before)
        process = subprocess.run(command, cwd=vibe, env=environment, capture_output=True, text=True)
        with (directory / "bench-stdout.txt").open("x") as stream:
            stream.write(process.stdout + process.stderr)
        printed = re.findall(r"\brc=(\d+)\b", process.stdout)
        if process.returncode or printed != ["0"]:
            raise RuntimeError(f"bench failed: process={process.returncode}, printed={printed}")
        expected = {build_result_filepath(alias, 100, parsed[0], group)[1]
                    for group in parsed[0].query_argument_groups}
        with h5py.File(raw, "r") as handle:
            if set(handle) != expected:
                raise ValueError("MLANN raw results lack the exact nine frozen groups")
            affinities = set()
            for group in handle.values():
                attrs = group.attrs
                affinities.add(tuple(int(cpu) for cpu in attrs.get("qag_cpu_affinity", ())))
                if (attrs.get("run_count") != 5 or attrs.get("qag_arm") != 4
                        or attrs.get("qag_warm_indexes_built") != 0
                        or attrs.get("qag_construction_queries") != 32768
                        or attrs.get("qag_reserved_validation_queries") != 0
                        or attrs.get("qag_candidate_training_query_sha256") != environment["WARM_QAG_TRAINING_QUERIES_SHA256"]
                        or "n_trees=80, depth=13" not in str(attrs.get("qag_mlann_construction", ""))):
                    raise ValueError("MLANN raw metadata differs from the frozen comparator")
                if not {name + "=1" for name in THREADS} <= set(str(attrs.get("qag_thread_environment", "")).split(",")):
                    raise ValueError("MLANN thread policy differs")
            if len(affinities) != 1 or len(next(iter(affinities))) != 1:
                raise ValueError("MLANN groups must use one identical VIBE-selected CPU")
        verify_inputs(before)
        saved_raw = directory / "raw.hdf5"
        with raw.open("rb") as source, saved_raw.open("xb") as target:
            shutil.copyfileobj(source, target)
        saved_raw.chmod(0o444)
        log = Path.home() / "vibe-warm/logs" / (alias + "-definitions.log")
        if log.exists():
            with log.open("rb") as source, (directory / "bench.log").open("xb") as target:
                shutil.copyfileobj(source, target)
        receipt = directory / "benchmark.json"
        write_new(receipt, {**before, "completed_utc": now(), "status": "benchmark_complete",
                            "raw_results": record(saved_raw), "vibe_raw_results": record(raw)})
        report = directory / "report.json"
        write_report(saved_raw, view, report, receipt)
        report.chmod(0o444)
        write_new(directory / "completion.json", {"status": "success", "created_utc": now(),
                                                  "benchmark": record(receipt), "report": record(report)})
        print(report, flush=True)
    except BaseException as error:
        write_new(directory / "failure.json", {"status": "failed", "created_utc": now(),
                                               "error_type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
