# Copyright 2024-2026 WritersLogic Contributors
# SPDX-License-Identifier: Apache-2.0
"""Measure the declared serving comparison with VIBE's saved fifth-pass timings."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys


THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")
NAMES = {0: "lifted", 1: "lifted+reservoir", 2: "direct", 3: "direct+reservoir"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(args):
    import h5py
    import numpy as np

    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}", args.run_id):
        raise ValueError("invalid run ID")
    repo = Path(__file__).resolve().parents[2]
    extractor = repo / "benchmarks/results/query_aware_graph_v1/sources/qag_report_full_evaluation_v1.py"
    if not extractor.exists():
        subprocess.run([sys.executable, str(repo / "scripts/restore-retired-inputs.py"),
                        "--path", str(extractor.relative_to(repo))], check=True)
    spec = importlib.util.spec_from_file_location("qag_serving_report", extractor)
    reporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporter)
    diagnostic = json.loads(args.diagnostic.read_text())
    if not diagnostic["retention_preserves_traversal"] or diagnostic["input"]["official_test_read"]:
        raise ValueError("requires the completed reserved validation diagnosis")
    expected = {(int(r["direct_ip"]) * 2 + int(r["reservoir"] > 0), r["ef"], r["query"]): r["ids"]
                for r in diagnostic["queries"]}
    encoded_bytes = {r["direct_ip"]: r["diagnostic"]["index_bytes"] for r in diagnostic["encodings"]}
    if len(expected) != 4 * 3 * 256:
        raise ValueError("incomplete diagnostic matrix")
    dependencies = {args.view.resolve(), args.training.resolve(), args.diagnostic.resolve(), args.extension.resolve()}
    baseline_raw = args.baseline / "raw.hdf5"
    baseline_report = args.baseline / "report.json"
    saved = json.loads(baseline_report.read_text())
    if digest(baseline_raw) != saved["provenance"]["raw_results_sha256"]:
        raise ValueError("frozen baseline raw results changed")
    with h5py.File(baseline_raw, "r") as results:
        for rep in range(3):
            for ef in (100, 256, 512):
                rr = 100 if ef == 100 else 400
                reference = np.asarray([expected[0, ef, i] for i in range(256)])
                if not np.array_equal(results[f"0_{ef}_{rr}_{rep}"]["neighbors"][:], reference):
                    raise ValueError("legacy serving differs from frozen v2 ordered IDs")
    dependencies.update((baseline_raw.resolve(), baseline_report.resolve()))
    write_new(args.out / "baseline-parity.json", {"ordered_ids_identical": True,
        "queries": 256, "settings": 3, "repetitions": 3, "raw": record(baseline_raw)})
    with h5py.File(args.view, "r") as source:
        if (source.attrs["distance"] != "ip" or source.attrs.get("qag_scope") != "validation"
                or source["test"].shape != (256, 128) or source["learn"].shape != (32768, 128)
                or source["train"].shape != (diagnostic["input"]["rows"], 128)):
            raise ValueError("view differs from the declared full-base validation split")
        for name in source:
            link = source.get(name, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                dependencies.add((args.view.parent / link.filename).resolve())
            for filename, _, _ in source[name].external or []:
                dependencies.add(Path(filename).resolve())
    vibe = Path.home() / "vibe-hms/vibe"
    bench = Path.home() / "vibe-hms/bench.sh"
    module = vibe / "vibe/algorithms/qag_serving/module.py"
    module.parent.mkdir(exist_ok=True)
    adapter = Path(__file__).with_name("qag_serving_module.py")
    if module.exists() and digest(module) != digest(adapter):
        raise ValueError("refusing to replace another serving adapter")
    if not module.exists():
        shutil.copyfile(adapter, module)
        (module.parent / "__init__.py").touch()
    schedule = [[arm, ef, 400, rep] for rep in range(3) for ef in (100, 256, 512)
                for arm in ((0, 1, 2, 3) if rep != 1 else (3, 2, 1, 0))]
    folder = args.out / "definitions/qag_serving"
    folder.mkdir(parents=True)
    config = folder / "config.yml"
    write_new(config, {"float": {"ip": [{"name": "qag-serving-v4", "module": "vibe.algorithms.qag_serving",
        "constructor": "QAGServing", "singularity_image": "hms", "disabled": False,
        "ood": True, "base_args": ["@metric"], "run_groups": {"fixed": {"args": [64, 500],
            "query_arg_groups": [schedule]}}}]}})
    sources = [Path(__file__), adapter, extractor, config, bench, vibe / "run.py", module,
        vibe / "vibe/algorithms/qag_pair/module.py", *(vibe / "vibe" / f"{n}.py" for n in
        ("main", "runner", "workers", "numa", "definitions", "results", "distance"))]
    snapshots = args.out / "snapshots"
    snapshots.mkdir()
    for i, path in enumerate(sources):
        shutil.copyfile(path, snapshots / f"{i:02d}-{path.name}")
    inputs = [record(p) for p in sources + sorted(dependencies)]
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("HMS_QGRAPH_", "HMS_QAG_"))}
    environment.update({name: "1" for name in THREADS})
    environment.update(HMS_QAG_CANDIDATE_SO=str(args.extension.resolve()),
        HMS_QAG_TRAINING_QUERIES=str(args.training.resolve()), HMS_QAG_TRAINING_QUERIES_SHA256=digest(args.training))
    alias = f"qag-serving-{args.run_id}-128-ip"
    link = vibe / "data" / f"{alias}.hdf5"
    if link.exists() or link.is_symlink() or (vibe / "results" / alias).exists():
        raise FileExistsError("existing VIBE serving results")
    link.symlink_to(args.view.resolve())
    command = [str(bench), alias, "1", "vibe", str(folder.parent), "nolock"]
    frozen = {"scope": "reserved learn validation only; no untouched final test", "host": socket.gethostname(),
        "inputs": inputs, "schedule": schedule, "command": command, "harness_sha256": digest(vibe / "vibe/runner.py"),
        "environment": {k: v for k, v in environment.items() if k in THREADS or k.startswith("HMS_QAG_")}}
    write_new(args.out / "frozen.json", frozen)
    process = subprocess.run(command, cwd=vibe, env=environment, capture_output=True, text=True)
    (args.out / "bench-stdout.txt").write_text(process.stdout + process.stderr)
    if process.returncode or re.findall(r"\brc=(\d+)\b", process.stdout) != ["0"]:
        raise RuntimeError("bench.sh failed; inspect its preserved log")
    raw_paths = list((vibe / "results" / alias / "100/qag-serving-v4").glob("*.hdf5"))
    if len(raw_paths) != 1:
        raise ValueError("expected one raw serving result")
    raw = args.out / "raw.hdf5"
    shutil.copyfile(raw_paths[0], raw)
    shutil.copyfile(Path.home() / "vibe-hms/logs" / f"{alias}-definitions.log", args.out / "bench.log")
    with h5py.File(raw, "r") as results:
        if set(results) != {"_".join(map(str, group)) for group in schedule}:
            raise ValueError("incomplete raw schedule")
        affinities = {tuple(g.attrs["qag_cpu_affinity"]) for g in results.values()}
        if len(affinities) != 1 or len(next(iter(affinities))) != 1:
            raise ValueError("all groups must use the same single VIBE CPU")
        for name, group in results.items():
            arm, ef, _, _ = map(int, name.split("_"))
            if not {f"{n}=1" for n in THREADS} <= set(str(group.attrs["qag_thread_environment"]).split(",")):
                raise ValueError("thread policy differs")
            for direct, label in ((False, "lifted"), (True, "direct")):
                if int(group.attrs[f"qag_{label}_index_bytes"]) != encoded_bytes[direct]:
                    raise ValueError("VIBE and native encoded sizes differ")
            actual = group["neighbors"][:]
            reference = np.asarray([expected[arm, ef, i] for i in range(256)])
            if not np.array_equal(actual, reference):
                raise ValueError(f"VIBE and native diagnostic IDs differ for {name}")
    if any(record(item["path"]) != item for item in inputs):
        raise ValueError("an input changed during timing")
    receipt = args.out / "benchmark.json"
    write_new(receipt, {**frozen, "raw_results": record(raw), "native_diagnostic_ids_identical": True})
    report = reporter.extract(raw, args.view, receipt)
    for row in report["rows"] + report["targets"]:
        row["method"] = NAMES[row["arm"]]
    write_new(args.out / "report.json", report)
    write_new(args.out / "completion.json", {"status": "complete", "report": record(args.out / "report.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("view", "training", "extension", "diagnostic", "baseline", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if (socket.gethostname() != "writerslogic-02" or os.environ.get("HMS_QAG_OUTER_LOCK") != "1"
            or any(os.environ.get(name) != "1" for name in THREADS)):
        parser.error("use the shared outer timing lock and one-thread environment")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as error:
        write_new(args.out / "failure.json", {"error_type": type(error).__name__, "error": str(error)})
        raise
