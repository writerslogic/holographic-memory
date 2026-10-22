# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Freeze and execute one full-data validation or official-test run under qag_execute.sh."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys


THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def now():
    return datetime.now(timezone.utc).isoformat()


def resolved(path):
    return path.expanduser().resolve(strict=True)


def host():
    return {"hostname": socket.gethostname(), "platform": platform.platform(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "lscpu": json.loads(subprocess.check_output(["lscpu", "--json"], text=True)),
            "python": sys.executable}


def record(path):
    path = resolved(path)
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def checked_files(folder):
    manifest_path = folder / "meta.json"
    manifest = json.loads(manifest_path.read_text())
    files = [record(manifest_path)]
    for name, expected in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("manifest file must be a basename")
        item = record(folder / name)
        if item["sha256"] != expected:
            raise ValueError(f"matrix checksum mismatch: {name}")
        files.append(item)
    return manifest, files


def check_config(path):
    config = json.loads(path.read_text())
    if set(config) != {"float"} or set(config["float"]) != {"ip"}:
        raise ValueError("freeze requires an IP-only paired definition")
    definitions = config["float"]["ip"]
    if len(definitions) != 1:
        raise ValueError("freeze exactly one paired definition")
    definition = definitions[0]
    if (not re.fullmatch(r"[a-zA-Z0-9_-]+", definition["name"])
            or set(definition["run_groups"]) != {"fixed"}):
        raise ValueError("definition name and build groups must be fixed basenames")
    fixed = definition["run_groups"]["fixed"]
    if (definition["module"] != "vibe.algorithms.qag_pair"
            or definition["constructor"] != "QAGPair" or not definition.get("ood")
            or definition.get("disabled") or definition["base_args"] != ["@metric"]
            or fixed["args"] != {"degree": [64], "build_ef": [500]}):
        raise ValueError("unexpected paired full-data build definition")
    groups = fixed["query_arg_groups"]
    if len(groups) != 1 or not 6 <= len(groups[0]) <= 54 or len(groups[0]) % 3:
        raise ValueError("freeze requires a bounded three-repetition paired schedule")
    schedule = groups[0]
    repetitions = {}
    for group in schedule:
        if (len(group) not in (4, 5) or any(type(value) is not int for value in group)
                or group[0] not in range(5) or group[1] <= 0 or group[2] < 0):
            raise ValueError("invalid paired query group")
        if len(group) == 5 and (group[0] != 1 or group[4] <= 0):
            raise ValueError("only candidate configurations may select positive patience")
        repetitions.setdefault(tuple(group[:3]) + (group[4] if len(group) == 5 else 0,), []).append(group[3])
    if ({key[0] for key in repetitions} != set(range(5))
            or any(sorted(values) != [0, 1, 2] for values in repetitions.values())
            or len(set(map(tuple, schedule))) != len(schedule)):
        raise ValueError("each frozen configuration needs repetitions 0, 1 and 2")
    return definition, schedule


def verify_inputs(frozen):
    for item in frozen["inputs"]:
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"frozen input changed: {item['path']}")


def freeze(args):
    source, native, training = map(resolved, (args.source, args.native, args.training))
    vibe, bench = map(resolved, (args.vibe, args.bench))
    config = resolved(args.config)
    definition, schedule = check_config(config)
    meta, native_files = checked_files(native)
    learn, training_files = checked_files(training)
    source_record = record(source)
    if (meta["source_metric"] != "ip" or meta["rows"] != meta["original_rows"]
            or meta["source_sha256"] != source_record["sha256"]
            or learn["source_sha256"] != source_record["sha256"]
            or learn["data_manifest_sha256"] != digest(native / "meta.json")
            or learn["queries"] != 32768 or learn["rows"] != meta["rows"]
            or learn["dimension"] != meta["dimension"] or learn["neighbors"] != 100):
        raise ValueError("requires matching full-base manifests and exactly 32768 construction queries")
    validation_report = None
    if args.phase == "official":
        if args.validation_report is None:
            raise ValueError("official freeze requires a completed validation report")
        validation_report = record(args.validation_report)
        result = json.loads(Path(validation_report["path"]).read_text())
        if (result["source"]["official_test"] or result["source"]["base_rows"] != meta["rows"]
                or result["source"]["dimension"] != meta["dimension"]):
            raise ValueError("official freeze requires a matching full-base validation report")
    args.out.mkdir(parents=True, exist_ok=False)
    snapshots = args.out / "snapshots"
    snapshots.mkdir()
    definitions = args.out / "definitions" / "qag_pair"
    definitions.mkdir(parents=True)
    frozen_config = definitions / "config.yml"
    shutil.copyfile(config, frozen_config)
    frozen_config.chmod(0o444)
    inputs = [source_record, *native_files, *training_files, record(frozen_config)]
    extensions = {}
    for arm, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        original = record(path)
        target = snapshots / (arm + "-holographic_memory.so")
        shutil.copyfile(original["path"], target)
        target.chmod(0o444)
        extensions[arm] = {**record(target), "original_path": original["path"]}
        inputs.append(record(target))
    if extensions["baseline"]["sha256"] == extensions["candidate"]["sha256"]:
        raise ValueError("baseline and candidate extensions must differ")
    runtime_files = [bench, vibe / "run.py", *(vibe / "vibe" / name for name in
                     ("main.py", "runner.py", "results.py", "datasets.py", "definitions.py")),
                     vibe / "vibe/algorithms/qag_pair/module.py",
                     vibe / "vibe/algorithms/roargraph/module.py"]
    helpers = [Path(__file__), Path(__file__).with_suffix(".sh"),
               *(Path(__file__).with_name(name) for name in
                 ("qag_hdf5.py", "qag_prepare.py", "qag_report.py"))]
    source_snapshots = []
    for index, path in enumerate([*runtime_files, *helpers]):
        item = record(path)
        inputs.append(item)
        target = snapshots / f"source-{index:02d}-{path.name}"
        shutil.copyfile(path, target)
        target.chmod(0o444)
        source_snapshots.append({**record(target), "original_path": item["path"]})
    competitor_modules = []
    for name in ("RoarGraph", "mlann"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"missing comparator module: {name}")
        paths = {Path(spec.origin)}
        for folder in spec.submodule_search_locations or ():
            paths.update(Path(folder).rglob("*.so"))
        for path in sorted(paths):
            item = record(path)
            competitor_modules.append({"module": name, **item})
            inputs.append(item)
    frozen = {
        "schema_version": 1, "created_utc": now(), "phase": args.phase,
        "dataset": args.dataset, "source": source_record, "native": str(native),
        "training": str(training), "vibe": str(vibe), "bench": str(bench),
        "definitions": str(definitions.parent.resolve()), "algorithm": definition["name"],
        "extensions": extensions, "schedule": schedule, "inputs": inputs,
        "source_snapshots": source_snapshots, "competitor_modules": competitor_modules,
        "host": host(), "harness_sha256": digest(vibe / "vibe/runner.py"),
        "vibe_sha": subprocess.check_output(["git", "-C", str(vibe), "rev-parse", "HEAD"], text=True).strip(),
        "validation_report": validation_report,
        "policy": {"construction_queries": 32768, "reserved_ids": "empty: learn already excludes reserved validation",
                   "threads": {name: "1" for name in THREADS}, "parallelism": 1,
                   "cache": "same resident indexes, five passes per group, report saved fifth pass",
                   "repetitions": [0, 1, 2], "competitors": "best among frozen measured 32768-query configurations"},
    }
    write_new(args.out / "frozen.json", frozen)
    print(args.out / "frozen.json", flush=True)


def make_view(frozen, directory):
    import h5py
    from qag_hdf5 import export

    native, training = Path(frozen["native"]), Path(frozen["training"])
    exported = directory / "validation-export.hdf5"
    export(native, training, exported)
    view = directory / "evaluation.hdf5"
    source = Path(frozen["source"]["path"]) if frozen["phase"] == "official" else exported
    with h5py.File(view, "x") as handle, h5py.File(source, "r") as base:
        handle.attrs.update(dimension=int(base.attrs["dimension"]), distance="ip", point_type="float",
                            qag_scope="official_test" if frozen["phase"] == "official" else "validation",
                            qag_original_source_sha256=frozen["source"]["sha256"],
                            qag_construction_policy="identical 32768 construction learn rows in both phases")
        for name in ("train", "test", "neighbors", "distances"):
            handle[name] = h5py.ExternalLink(str(source), name)
        for name in ("learn", "learn_neighbors"):
            handle[name] = h5py.ExternalLink(str(exported), name)
    view.chmod(0o444)
    exported.chmod(0o444)
    return view, exported


def run(args):
    import h5py
    from qag_report import write_report

    frozen_path = resolved(args.freeze)
    frozen = json.loads(frozen_path.read_text())
    if socket.gethostname() != frozen["host"]["hostname"]:
        raise ValueError("frozen execution host differs")
    verify_inputs(frozen)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", args.run_id):
        raise ValueError("run ID must be a simple unique basename")
    directory = args.runs.expanduser().resolve() / args.run_id
    directory.mkdir(parents=True, exist_ok=False)
    try:
        view, exported = make_view(frozen, directory)
        reserved = directory / "reserved-validation-ids.u32"
        with reserved.open("xb"):
            pass
        reserved.chmod(0o444)
        dimension = json.loads((Path(frozen["native"]) / "meta.json").read_text())["dimension"]
        alias = f"qag-{frozen['phase']}-{args.run_id}-{dimension}-ip"
        vibe = Path(frozen["vibe"])
        raw_root = vibe / "results" / alias
        if raw_root.exists() or (vibe / "data" / (alias + ".hdf5")).exists():
            raise FileExistsError("refusing to reuse an existing VIBE dataset or result path")
        (vibe / "data" / (alias + ".hdf5")).symlink_to(view)
        environment = {name: value for name, value in os.environ.items()
                       if not name.startswith(("HMS_QGRAPH_", "HMS_QAG_"))}
        environment.update({name: "1" for name in THREADS})
        environment.update(HMS_QAG_BASELINE_SO=frozen["extensions"]["baseline"]["path"],
                           HMS_QAG_CANDIDATE_SO=frozen["extensions"]["candidate"]["path"],
                           HMS_QAG_VALIDATION_IDS=str(reserved),
                           HMS_QAG_TRAINING_QUERIES=str(Path(frozen["training"]) / "queries.f32"),
                           HMS_QAG_TRAINING_QUERIES_SHA256=digest(Path(frozen["training"]) / "queries.f32"))
        command = [frozen["bench"], alias, "1", "vibe", frozen["definitions"], "nolock"]
        before = {"schema_version": 1, "created_utc": now(), "phase": frozen["phase"],
                  "freeze": record(frozen_path), "host": host(), "command": command,
                  "outer_flock": str(Path.home() / "hms-timing.lock"),
                  "harness_sha256": frozen["harness_sha256"], "dataset_view": record(view),
                  "validation_export": record(exported), "reserved_ids": record(reserved),
                  "environment": {name: environment[name] for name in environment
                                  if name in THREADS or name.startswith("HMS_QAG_")}}
        write_new(directory / "before.json", before)
        process = subprocess.run(command, cwd=vibe, env=environment, capture_output=True, text=True)
        with (directory / "bench-stdout.txt").open("x") as stream:
            stream.write(process.stdout + process.stderr)
        printed_status = re.findall(r"\brc=(\d+)\b", process.stdout)
        if process.returncode or printed_status != ["0"]:
            raise RuntimeError(f"bench.sh failed: process={process.returncode}, printed={printed_status}")
        raw = raw_root / "100" / frozen["algorithm"] / "ip_64_500.hdf5"
        expected = {"_".join(map(str, group)) for group in frozen["schedule"]}
        with h5py.File(raw, "r") as result:
            if set(result) != expected or any(result[name].attrs.get("run_count") != 5 for name in result):
                raise ValueError("raw groups do not match the complete frozen five-pass schedule")
            affinities = set()
            for group in result.values():
                attrs = group.attrs
                affinities.add(tuple(int(cpu) for cpu in attrs.get("qag_cpu_affinity", ())))
                if (attrs.get("qag_candidate_training_queries") != 32768
                        or attrs.get("qag_construction_queries") != 32768
                        or attrs.get("qag_reserved_validation_queries") != 0
                        or attrs.get("qag_baseline_so") != frozen["extensions"]["baseline"]["path"]
                        or attrs.get("qag_candidate_so") != frozen["extensions"]["candidate"]["path"]):
                    raise ValueError("raw arm artifacts or construction policy differ from freeze")
                recorded = str(attrs.get("qag_thread_environment", "")).split(",")
                if not {name + "=1" for name in THREADS} <= set(recorded):
                    raise ValueError("raw thread policy differs from freeze")
            if len(affinities) != 1 or len(next(iter(affinities))) != 1:
                raise ValueError("all groups must use the same single CPU chosen by the VIBE runner")
        verify_inputs(frozen)
        if digest(view) != before["dataset_view"]["sha256"] or digest(exported) != before["validation_export"]["sha256"]:
            raise ValueError("evaluation inputs changed during timing")
        saved_raw = directory / "raw.hdf5"
        with raw.open("rb") as source, saved_raw.open("xb") as target:
            shutil.copyfileobj(source, target)
        saved_raw.chmod(0o444)
        log = Path.home() / "vibe-hms/logs" / (alias + "-" + Path(frozen["definitions"]).name + ".log")
        if log.exists():
            with log.open("rb") as source, (directory / "bench.log").open("xb") as target:
                shutil.copyfileobj(source, target)
        receipt = directory / "benchmark.json"
        write_new(receipt, {**before, "completed_utc": now(), "status": "benchmark_complete",
                            "before": record(directory / "before.json"), "raw_results": record(saved_raw),
                            "vibe_raw_results": record(raw)})
        report = directory / "report.json"
        write_report(saved_raw, view, report, receipt)
        report.chmod(0o444)
        write_new(directory / "completion.json", {"status": "success", "created_utc": now(),
                                                  "benchmark_receipt": record(receipt), "report": record(report)})
        print(report, flush=True)
    except BaseException as error:
        write_new(directory / "failure.json", {"status": "failed", "created_utc": now(),
                                               "error_type": type(error).__name__, "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("--phase", choices=("validation", "official"), required=True)
    prepare.add_argument("--dataset", choices=("llama-128-ip", "yi-128-ip"), required=True)
    for name in ("source", "native", "training", "config", "baseline", "candidate", "out"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--validation-report", type=Path)
    prepare.add_argument("--vibe", type=Path, default=Path.home() / "vibe-hms/vibe")
    prepare.add_argument("--bench", type=Path, default=Path.home() / "vibe-hms/bench.sh")
    execute = commands.add_parser("run")
    execute.add_argument("--freeze", type=Path, required=True)
    execute.add_argument("--run-id", required=True)
    execute.add_argument("--runs", type=Path, default=Path.home() / "sol-qag/runs")
    args = parser.parse_args()
    if (sys.platform != "linux" or socket.gethostname().split(".")[0] != "writerslogic-02"
            or os.environ.get("HMS_QAG_OUTER_LOCK") != "1"
            or any(os.environ.get(name) != "1" for name in THREADS)):
        parser.error("execute on writerslogic-02 through qag_execute.sh and its outer flock")
    freeze(args) if args.command == "freeze" else run(args)


if __name__ == "__main__":
    main()
